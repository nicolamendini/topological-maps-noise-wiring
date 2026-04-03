import gc
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from neuralsheet import NeuralSheet
from wiring_efficiency_utils import *
from map_plotting import *


# -----------------------------
# Fixed experiment configuration
# -----------------------------
crop_size = 24
batch_size = 32
num_workers = 4
root_dir = './input_stimuli'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
beta = 1 - 5e-5
loss_beta = 3e-3
R_rf = 7

dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)

# Keep these exactly as requested.
trialvar = [9]
sizesvar = [30]
noise_vals = [0.05]

sparsity_vals = torch.arange(0.8, 1, 0.005)

# 1 epoch is enough.
epochs = 1

# Parameter grids
p0_values = [0] * 5
p1_values = np.arange(0, 0.6, 0.1)
p2_values = np.arange(0.05, 0.35, 0.05)

runs = len(p0_values) * len(p1_values) * len(p2_values) * len(sizesvar) * 2
print('# EXPECTED RUNS: ' + str(runs))
print(p1_values)
print(p2_values)

# Evaluation settings
DIM_MAX_BATCHES = 151
ROBUSTNESS_SAMPLES = 1000
ROBUSTNESS_SIMILARITY_FLOOR = 0.1


def to_float(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().cpu().mean().item())
    return float(x)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def center_crop_for_robustness(x: torch.Tensor, margin: int) -> torch.Tensor:
    if margin <= 0:
        return x
    h, w = x.shape[-2:]
    h_end = h - margin - 1
    w_end = w - margin - 1
    if h_end <= margin or w_end <= margin:
        return x
    return x[:, :, margin:h_end, margin:w_end]


def train_single_model(microcolumnar, p0: float, p1: float, p2: float, sheet_size: int, r_long: float):
    model = NeuralSheet(
        crop_size,
        sheet_size,
        R_rf,
        R_long=r_long,
        device=device,
        microcolumnar=microcolumnar,
        p = [p0, p1, p2]
    ).to(device)

    lr = 1e-3
    network = init_nn(sheet_size, crop_size, 1)
    avg_loss = 0.0

    model.train()
    for _ in range(epochs):
        batch_progress = tqdm(dataloader, leave=False, desc=f"train R_long={r_long:.2f}")
        for batch in batch_progress:
            batch = batch.to(device)
            batch_responses = []
            batch_inputs = []

            for image in batch:
                image = image[0:1][None].flip(1)

                if image.mean() <= 0.15:
                    continue

                lr = max(lr * beta, 1e-4)
                model.hebbian_lr = lr * 1e2
                model.homeo_lr = lr

                model(image, adaptation=True)
                model.hebbian_step()

                batch_responses.append(model.current_response_l3.clone())
                batch_inputs.append(model.current_input.clone())

            if not batch_responses:
                continue

            batch_responses = torch.cat(batch_responses, dim=0)
            batch_inputs = torch.cat(batch_inputs, dim=0)

            reco_input = network['activ'](network['model'](batch_responses))
            targets = batch_inputs

            loss, _ = nn_loss(network, targets, reco_input)
            sim = cosim(targets.detach().cpu(), reco_input.detach().cpu(), True)
            avg_loss = (1 - loss_beta) * avg_loss + loss_beta * to_float(sim)

            network['optim'].zero_grad()
            loss.backward()
            network['optim'].step()

    return model, avg_loss


@torch.no_grad()
def compute_dimensionality(model: torch.nn.Module) -> float:
    model.eval()
    codes = []

    batch_progress = tqdm(dataloader, leave=False, desc='dimensionality')
    for b_idx, batch in enumerate(batch_progress):
        batch = batch.to(device)

        for image in batch:
            image = image[0:1][None].flip(1)
            if image.mean() <= 0.15:
                continue

            model(image, adaptation=False)
            codes.append(model.current_response_l3.detach().cpu())

        if b_idx >= DIM_MAX_BATCHES:
            break

    if not codes:
        return float('nan')

    codes = torch.cat(codes, dim=0)
    eff_dim, _ = get_pca_dimensions(codes, 3)
    return to_float(eff_dim)


@torch.no_grad()
def compute_robustness(model, s_idx) -> float:
    model.eval()
    sims = []
    margin = int(float(trialvar[s_idx]) // 2)

    batch_progress = tqdm(dataloader, leave=False, desc='robustness')
    for batch in batch_progress:
        batch = batch.to(device)

        for image in batch:
            img = image[0:1][None].flip(1)

            model(img, adaptation=False)
            dense = model.current_response_l3.detach().cpu()
            dense = center_crop_for_robustness(dense, margin)

            if dense.mean() <= 0.05:
                continue

            model(img, adaptation=False, noise_gamma=noise_vals[s_idx])
                
            perturbed = model.current_response_l3.detach().cpu()
            perturbed = center_crop_for_robustness(perturbed, margin)

            sims.append(to_float(cosim(dense, perturbed)))

            if len(sims) >= ROBUSTNESS_SAMPLES:
                break

        if len(sims) >= ROBUSTNESS_SAMPLES:
            break

    if not sims:
        return float('nan')

    sims = np.asarray(sims, dtype=np.float32)
    valid = sims[sims > ROBUSTNESS_SIMILARITY_FLOOR]
    if valid.size == 0:
        return float('nan')
    return float(valid.mean())


@torch.no_grad()
def compute_sparsity(model, s_idx) -> torch.Tensor:
    model.eval()
    total_sims = torch.zeros(len(sparsity_vals), device=device) # Stay on device
    counts = torch.zeros(len(sparsity_vals), device=device)
    
    margin = int(float(trialvar[s_idx]) // 2)
    processed_samples = 0

    for batch in tqdm(dataloader, leave=False, desc='sparsity sweep'):
        batch = batch.to(device)
        for image in batch:
            img = image[0:1][None].flip(1)

            model(img, adaptation=False)
            dense = model.current_response_l3.clone()
            dense = center_crop_for_robustness(dense, margin)

            if dense.mean() <= 0.05: continue

            for idx, spa_val in enumerate(sparsity_vals):
                model(img, adaptation=False, sparsity=float(spa_val))
                perturbed = model.current_response_l3
                perturbed = center_crop_for_robustness(perturbed, margin)
                
                # Keep math on GPU
                sim = cosim(dense, perturbed)
                if sim > ROBUSTNESS_SIMILARITY_FLOOR:
                    total_sims[idx] += sim
                    counts[idx] += 1

            processed_samples += 1
            if processed_samples >= ROBUSTNESS_SAMPLES: break
        if processed_samples >= ROBUSTNESS_SAMPLES: break

    return (total_sims / torch.clamp(counts, min=1)).cpu() # Move to CPU only at the end


@torch.no_grad()
def vertical_distance(model):
    model.eval()
    
    ori_map = compute_orientation_maps(model, model.current_input.shape[-1], device=model.device)
    sim = torch.cos(ori_map[0] - ori_map[1]).mean().cpu()
    return sim


def run_search(microcolumnar):

    shape_spa = (len(p0_values), len(p1_values), len(p2_values), len(sizesvar), len(sparsity_vals))
    sparsity = torch.full(shape_spa, torch.nan, dtype=torch.float32)
    
    shape = (len(p0_values), len(p1_values), len(p2_values), len(sizesvar))

    accuracy = torch.full(shape, torch.nan, dtype=torch.float32)
    dimensionality = torch.full(shape, torch.nan, dtype=torch.float32)
    robustness = torch.full(shape, torch.nan, dtype=torch.float32)
    columnarity = torch.full(shape, torch.nan, dtype=torch.float32)

    total_runs = len(p0_values) * len(p1_values) * len(p2_values)
    grid_progress = tqdm(total=total_runs, desc='p0/p1/p2 grid')

    for i, p0 in enumerate(p0_values):
        for j, p1 in enumerate(p1_values):
            for k, p2 in enumerate(p2_values):
                for s in range(len(sizesvar)):
                    clear_memory()
                
                    # Note: Ensure trialvar is the same length as sizesvar or this will IndexError
                    model, avg_loss = train_single_model(
                        p0=float(p0),
                        p1=float(p1),
                        p2=float(p2),
                        sheet_size=int(sizesvar[s]),
                        r_long=float(trialvar[s]),
                        microcolumnar=microcolumnar
                    )
                
                    eff_dim = compute_dimensionality(model)
                    rob = compute_robustness(model, s)
                    ver_sim = vertical_distance(model)
                    spa = compute_sparsity(model, s)
                
                    # Store directly into the 4th dimension
                    accuracy[i, j, k, s] = avg_loss
                    dimensionality[i, j, k, s] = eff_dim
                    robustness[i, j, k, s] = rob
                    columnarity[i, j, k, s] = ver_sim
                    sparsity[i, j, k, s] = spa
                
                    del model
                    clear_memory()

                    grid_progress.set_postfix(
                        p0=f'{p0:.2f}',
                        p1=f'{p1:.2f}',
                        p2=f'{p2:.2f}',
                        acc=f'{accuracy[i, j, k, s].item():.4f}',
                        dim=f'{dimensionality[i, j, k, s].item():.4f}',
                        rob=f'{robustness[i, j, k, s].item():.4f}',
                    )
                    grid_progress.update(1)

    grid_progress.close()

    results = {
        'p0_values': torch.tensor(p0_values, dtype=torch.float32),
        'p1_values': torch.tensor(p1_values, dtype=torch.float32),
        'p2_values': torch.tensor(p2_values, dtype=torch.float32),
        'accuracy': accuracy,
        'dimensionality': dimensionality,
        'robustness': robustness,
        'columnarity': columnarity,
        'sparsity': sparsity,
        'config': {
            'crop_size': crop_size,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'root_dir': root_dir,
            'device': device,
            'beta': beta,
            'loss_beta': loss_beta,
            'R_rf': R_rf,
            'epochs': epochs,
            'dim_max_batches': DIM_MAX_BATCHES,
            'robustness_samples': ROBUSTNESS_SAMPLES,
            'noise_values_used': noise_vals,
            'sparsity_sweep_range': sparsity_vals,
            'robustness_similarity_floor': ROBUSTNESS_SIMILARITY_FLOOR
        },
    }

    title = 'search_results_micro.pt' if microcolumnar else 'search_results_macro.pt'
    torch.save(results, Path(title))


if __name__ == '__main__':
    run_search(microcolumnar=False)
    run_search(microcolumnar=True)