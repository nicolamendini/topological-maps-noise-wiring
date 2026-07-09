import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Image as IPImage
from helpers.wiring_efficiency_utils import *
from neuralsheet import *
from helpers.map_plotting import *
import os
import gc
import time
from wakepy import keep

DEVICE = 'cuda'
AGGRESSIVE_CLEANUP = False


def _validate_n_reps(n_reps):
    n_reps = int(n_reps)
    if n_reps < 1:
        raise ValueError('n_reps must be at least 1.')
    return n_reps


def _run_model(model, image, **kwargs):
    """Run the current L4 collector path without requiring L2/3 parameters."""
    return model(image, layer_3=False, **kwargs)


def _cat_or_none(tensors):
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _require_codes(code_tracker, context):
    if not code_tracker:
        raise RuntimeError(f'No valid responses collected during {context}.')
    return torch.cat(code_tracker, dim=0)


def _as_list(values):
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def _cleanup(force=False):
    if not (AGGRESSIVE_CLEANUP or force):
        return

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# python3 -c 'import stats_collector; stats_collector.collect_stats()' -m wakepy
#@profile
def collect_stats(n_reps=3):
    n_reps = _validate_n_reps(n_reps)
    
    # Example usage
    crop_size = 30 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = './input_stimuli'  # Path to your image folder
    device = DEVICE  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 5
    beta = 1 - 5e-5
    loss_beta = 1e-2

    dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
    
    trials = 3
    epochs = 1
    
    trialvar = np.sqrt(np.linspace(6**2, 18**2, trials))
    sizesvar = [30,60]
    N_CODES = (sizesvar[-1]+1)**2
    sizes = len(sizesvar)
    output_file = 'data_l4/data_accdim.pt'
    lr_initial = 1e-3
    lr_floor = 2e-4
    hebbian_lr_scale = 1e2
    model_kwargs = {
        'microcolumnar': False,
        'layer_3': False,
    }
    reco_tracker = torch.zeros((n_reps, sizes, trials, len(dataloader)))
    se_tracker = torch.zeros((n_reps, sizes, trials))
    spectrum_tracker = torch.zeros((n_reps, sizes, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((n_reps, sizes, trials))

    se_pca_tracker = torch.zeros((n_reps, sizes, trials))

    #------------------------- running simulations
    code_tracker = []
    for s in range(sizes):
        
        for t in range(trials):

            for rep in range(n_reps):
                    
                dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
                print('cropsize: ', crop_size)

                print('running simulation rep: ' + str(rep + 1) + '/' + str(n_reps) \
                      + ' size: ' + str(float(sizesvar[s])) \
                      + ' interaction radius: ' + str(float(trialvar[t])))

                model = NeuralSheet(crop_size, sizesvar[s], R_rf, R_long=trialvar[t], device=device, microcolumnar=False).to(device)
                    
                lr = lr_initial
                network = init_nn(sizesvar[s], crop_size)
                avg_loss = 0
                code_tracker = []
                batch_responses = []
                batch_inputs = []
                _cleanup()
            
                for e in range(epochs):
            
                    batch_progress = tqdm(dataloader, leave=False)
                    del code_tracker
                    code_tracker = []
                    
                    for b_idx, batch in enumerate(batch_progress):
            
                        del batch_inputs, batch_responses
                        batch_responses = []
                        batch_inputs = []
                        _cleanup()
                        
                        batch = batch.to(device)  # Transfer the entire batch to GPU
            
                        for image in batch:
            
                            image = image[0:1][None].flip(1)
            
                            if image.mean() > 0.15:
            
                                limit = lr_floor
                                lr *= beta
                                lr = lr if lr>limit else limit
            
                                model.hebbian_lr = lr * hebbian_lr_scale
                                model.homeo_lr = lr
            
                                _run_model(model, image)
                                model.hebbian_step()
            
                                if model.current_response.sum():
            
                                    batch_responses.append(model.current_response+0)
                                    batch_inputs.append(model.current_input+0)
                                    code_tracker.append(model.current_response+0)
            
                        if len(batch_responses):
                            
                            batch_responses = torch.cat(batch_responses, dim=0)
                            batch_inputs = torch.cat(batch_inputs, dim=0)
            
                            reco_input = network['activ'](network['model'](batch_responses))[:,:,R_rf:-R_rf,R_rf:-R_rf]
            
                            targets = batch_inputs[:,:,R_rf:-R_rf,R_rf:-R_rf]
                            loss, loss_std = nn_loss(network, targets, reco_input)
            
                            sim = cosim(targets.detach(), reco_input.detach())
                            reco_tracker[rep, s, t, b_idx] = sim
            
                            avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim
            
                            network['optim'].zero_grad()
                            loss.backward()
                            network['optim'].step()
            
                            mean_activation = model.mean_activations.mean()
                            mean_std = model.mean_activations.std() / model.homeo_target
                            batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}'.format(
                                mean_activation, 
                                mean_std, 
                                avg_loss,
                                lr
                            ))

                #------------------------- accuracy-dimensionality measurements

                code_tracker = _require_codes(code_tracker, 'accuracy/dimensionality training')

                mask = torch.isnan(code_tracker).any(dim=(-2, -1))
                print('finished training, number of Nan found: ' + str(int(mask.sum())))

                eff_dims, spectrum, peak = get_effective_dims(code_tracker[-N_CODES:])
                eff_dims_pca, samp_components = get_pca_dimensions(code_tracker[-N_CODES:])

                se_tracker[rep, s, t] = eff_dims
                se_pca_tracker[rep, s, t] = eff_dims_pca

                print('training complete, accuracy: ' + str(float(reco_tracker[rep, s, t, -N_CODES:].mean())) + ' dimensionality: ' \
                      + str(eff_dims_pca))

                spectrum_tracker[rep,s,t,:sizesvar[s],:sizesvar[s]] = spectrum.cpu()
                peak_tracker[rep,s,t] = peak.cpu()                    

    config = {
        'experiment': 'accuracy_dimensionality',
        'output_file': output_file,
        'device': device,
        'root_dir': root_dir,
        'crop_size': crop_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'aggressive_cleanup': AGGRESSIVE_CLEANUP,
        'epochs': epochs,
        'n_reps': n_reps,
        'trials': trials,
        'sizesvar': _as_list(sizesvar),
        'trialvar': _as_list(trialvar),
        'N_CODES': int(N_CODES),
        'R_rf': R_rf,
        'beta': beta,
        'loss_beta': loss_beta,
        'lr_initial': lr_initial,
        'lr_floor': lr_floor,
        'hebbian_lr_scale': hebbian_lr_scale,
        'model_kwargs': model_kwargs,
        'decoder': {
            'init_fn': 'init_nn',
            'input_size': 'sheet_size',
            'output_size': crop_size,
        },
        'result_axes': {
            'reco_tracker': ['rep', 'size', 'radius', 'batch'],
            'se_tracker': ['rep', 'size', 'radius'],
            'spectrum_tracker': ['rep', 'size', 'radius', 'x', 'y'],
            'peak_tracker': ['rep', 'size', 'radius'],
            'se_pca_tracker': ['rep', 'size', 'radius'],
        },
    }

    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'se_pca_tracker': se_pca_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar,
        'n_reps': n_reps,
        'config': config
    }


    torch.save(data, output_file)
    time.sleep(5)

def collect_noise_stats(minicolumnar=True, n_reps=3):
    n_reps = _validate_n_reps(n_reps)
    
    # Example usage
    crop_size = 30 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = './input_stimuli'  # Path to your image folder
    device = DEVICE  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 5
    beta = 1 - 5e-5
    loss_beta = 1e-2

    dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
    
    trials = 4
    n_conditions = 15
    epochs = 2

    if minicolumnar:
        trialvar = np.sqrt(np.linspace(6**2, 18**2, trials))
    else:
        trialvar = np.sqrt(np.linspace(2**2, 18**2, trials))
    
    sizesvar = np.round(trialvar * 5).astype(int)
    noise_conditions = np.linspace(0, 0.1, n_conditions)
    N_CODES = int((sizesvar[-1]+1)**2)
    output_file = 'data_l4/data_noise_sp.pt' if minicolumnar else 'data_l4/data_noise_topo.pt'
    lr_initial = 1e-3
    lr_floor = 2e-4
    hebbian_lr_scale = 1e2
    model_kwargs = {
        'microcolumnar': bool(minicolumnar),
        'layer_3': False,
    }
    reco_tracker = torch.zeros((n_reps, trials, len(dataloader)))
    se_tracker = torch.zeros((n_reps, trials))
    spectrum_tracker = torch.zeros((n_reps, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((n_reps, trials))

    se_pca_tracker = torch.zeros((n_reps, trials))

    noise_acc_tracker = torch.zeros((n_reps, trials, n_conditions))
    noise_dim_tracker = torch.zeros((n_reps, trials, n_conditions))
    noise_rob_tracker = torch.zeros((n_reps, trials, n_conditions))

    #------------------------- running simulations
    code_tracker = []
        
    for t in range(trials):

        for rep in range(n_reps):
                
            dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
            print('cropsize: ', crop_size)

            print('running simulation rep: ' + str(rep + 1) + '/' + str(n_reps) \
                  + ' size: ' + str(float(sizesvar[t])) \
                  + ' interaction radius: ' + str(float(trialvar[t])))

            if minicolumnar:
                model = NeuralSheet(crop_size, int(sizesvar[t]), R_rf, R_long=trialvar[t], device=device, microcolumnar=True).to(device)
            else:
                model = NeuralSheet(crop_size, int(sizesvar[t]), R_rf, R_long=trialvar[t], device=device, microcolumnar=False).to(device)
                
            lr = lr_initial
            network = init_nn(int(sizesvar[t]), crop_size)
            avg_loss = 0
            code_tracker = []
            batch_responses = []
            batch_inputs = []
            _cleanup()
        
            for e in range(epochs):

                batch_progress = tqdm(dataloader, leave=False)
                for b_idx, batch in enumerate(batch_progress):


                    batch_responses = []
                    batch_inputs = []
                    batch = batch.to(device)  # Transfer the entire batch to GPU

                    for image in batch:

                        image = image[0:1][None].flip(1)

                        if image.mean()>0.15:

                            limit = lr_floor
                            lr *= beta
                            lr = lr if lr>limit else limit
                            model.hebbian_lr = lr * hebbian_lr_scale
                            model.homeo_lr = lr

                            _run_model(model, image, adaptation=True)
                            model.hebbian_step()
                            
                            batch_responses.append(model.current_response.clone())
                            batch_inputs.append(model.current_input.clone())
                            code_tracker.append(model.current_response.clone())

                    batch_responses = _cat_or_none(batch_responses)
                    batch_inputs = _cat_or_none(batch_inputs)

                    if batch_responses is None:
                        continue

                    reco_input = network['activ'](network['model'](batch_responses))[:,:,R_rf:-R_rf,R_rf:-R_rf]
                    targets = batch_inputs[:,:,R_rf:-R_rf,R_rf:-R_rf]
                    
                    loss, loss_std = nn_loss(network, targets, reco_input)
                    
                    sim = cosim(targets.detach(), reco_input.detach(), True)
                    reco_tracker[rep, t, b_idx] = sim
                                    
                    avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim
                    
                    network['optim'].zero_grad()
                    loss.backward()
                    network['optim'].step()

                    mean_activation = model.mean_activations.mean()
                    mean_std = model.mean_activations.std() / model.homeo_target
                    
                    batch_progress.set_description('M:{:.3f} STD:{:.3f} BCE:{:.3f} LR:{:.5f} SP:{:.3f} D:{:.3f} A:{:.3f}'.format(
                        mean_activation, 
                        mean_std, 
                        avg_loss,
                        lr,
                        model.aff_strength.mean(),
                        model.gains.mean(),
                        model.delta_mag.mean()
                    ))
                    
            #------------------------- accuracy-dimensionality measurements

            code_tracker = _require_codes(code_tracker, 'noise-stat training')

            mask = torch.isnan(code_tracker).any(dim=(-2, -1))
            print('finished training, number of Nan found: ' + str(int(mask.sum())))

            eff_dims, spectrum, peak = get_effective_dims(code_tracker[-N_CODES:])
            eff_dims_pca, samp_components = get_pca_dimensions(code_tracker[-N_CODES:])

            se_tracker[rep, t] = eff_dims
            se_pca_tracker[rep, t] = eff_dims_pca

            print('training complete, accuracy: ' + str(float(reco_tracker[rep, t, -N_CODES:].mean())) + ' dimensionality: ' \
                  + str(eff_dims_pca))

            sheet_size = int(sizesvar[t])
            spectrum_tracker[rep,t,:sheet_size,:sheet_size] = spectrum.cpu()
            peak_tracker[rep,t] = peak.cpu()           

            #------------------------- noise robustness measurements

            print('collecting noise robustness measurements!')

            input_tracker = []
            code_tracker = []
            perturbed_code_tracker = []
                
            for n_idx, noise_gamma in tqdm(enumerate(noise_conditions)):

                del input_tracker, perturbed_code_tracker, code_tracker
                input_tracker = []
                code_tracker = []
                perturbed_code_tracker = []
                _cleanup()

                robustness_dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)

                for b_idx, batch in enumerate(robustness_dataloader):

                    _cleanup()
                    
                    batch = batch.to(device)  # Transfer the entire batch to GPU

                    for image in batch:

                        image = image[0:1][None].flip(1)

                        if image.mean() > 0.15:

                            _run_model(model, image, adaptation=False)

                            p = int(trialvar[t]//2)
                            if model.current_response[:,:,p:-p-1,p:-p-1].mean() > 0.05:

                                code_tracker.append(model.current_response.clone())
                                input_tracker.append(model.current_input.clone())
                                
                                _run_model(
                                    model,
                                    image,
                                    noise_gamma=noise_gamma, 
                                    adaptation=False
                                )
                                perturbed_code_tracker.append(model.current_response.clone())

                    if len(perturbed_code_tracker)>N_CODES:  
                        break

                input_tracker = _cat_or_none(input_tracker)
                code_tracker = _cat_or_none(code_tracker)
                perturbed_code_tracker = _cat_or_none(perturbed_code_tracker)

                if perturbed_code_tracker is None:
                    print('no valid responses collected for noise: ' + str(noise_gamma))
                    continue

                # normalising perturbations to simulate homeostasis
                #perturbed_code_tracker /= perturbed_code_tracker.sum([-2, -1], keepdim=True) + 1e-11
                #perturbed_code_tracker *= code_tracker.sum([-2, -1], keepdim=True)
                
                mask = torch.isnan(perturbed_code_tracker).any(dim=(-2, -1))

                eff_dims_pca, samp_components = get_pca_dimensions(perturbed_code_tracker)
                
                noise_dim_tracker[rep, t, n_idx] = eff_dims_pca

                reco_input = network['activ'](network['model'](perturbed_code_tracker))
                accuracy = cosim(reco_input.detach(), input_tracker.detach())
                noise_acc_tracker[rep, t, n_idx] = accuracy

                robustness = cosim(code_tracker.detach()[:,:,p:-p-1,p:-p-1], perturbed_code_tracker.detach()[:,:,p:-p-1,p:-p-1])
                noise_rob_tracker[rep, t, n_idx] = robustness

                print('measuring noise robustness, noise: ' + str(noise_gamma) + ', robustness: ' + str(float(robustness)))

                if robustness < 0.9:
                    break

    config = {
        'experiment': 'noise_robustness',
        'output_file': output_file,
        'device': device,
        'root_dir': root_dir,
        'crop_size': crop_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'aggressive_cleanup': AGGRESSIVE_CLEANUP,
        'epochs': epochs,
        'n_reps': n_reps,
        'trials': trials,
        'n_conditions': n_conditions,
        'sizesvar': _as_list(sizesvar),
        'trialvar': _as_list(trialvar),
        'noise_conditions': _as_list(noise_conditions),
        'N_CODES': int(N_CODES),
        'R_rf': R_rf,
        'beta': beta,
        'loss_beta': loss_beta,
        'lr_initial': lr_initial,
        'lr_floor': lr_floor,
        'hebbian_lr_scale': hebbian_lr_scale,
        'minicolumnar': bool(minicolumnar),
        'model_kwargs': model_kwargs,
        'decoder': {
            'init_fn': 'init_nn',
            'input_size': 'sheet_size',
            'output_size': crop_size,
        },
        'robustness': {
            'activity_margin': 'int(trialvar[t] // 2)',
            'min_central_mean_response': 0.05,
            'early_stop_robustness': 0.9,
        },
        'result_axes': {
            'reco_tracker': ['rep', 'radius', 'batch'],
            'se_tracker': ['rep', 'radius'],
            'spectrum_tracker': ['rep', 'radius', 'x', 'y'],
            'peak_tracker': ['rep', 'radius'],
            'se_pca_tracker': ['rep', 'radius'],
            'noise_acc': ['rep', 'radius', 'noise_condition'],
            'noise_dim': ['rep', 'radius', 'noise_condition'],
            'noise_rob': ['rep', 'radius', 'noise_condition'],
        },
    }

    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'se_pca_tracker': se_pca_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar,
        'n_reps': n_reps,
        'noise_conditions' : noise_conditions,
        'noise_acc': noise_acc_tracker,
        'noise_dim': noise_dim_tracker,
        'noise_rob': noise_rob_tracker,
        'config': config
    }


    if minicolumnar:
        torch.save(data, output_file)
    else:
        torch.save(data, output_file)
        
    time.sleep(5)

            
def train_map(sheet_size, crop_size, epochs, dataloader, beta, model, reco_tracker=None, loss_beta=1e-2):
    lr = 1e-3
    network = init_nn(sheet_size, crop_size)
    avg_loss = 0
    code_tracker = []
    batch_responses = []
    batch_inputs = []
    _cleanup()

    for e in range(epochs):

        batch_progress = tqdm(dataloader, leave=False)
        del code_tracker
        code_tracker = []
        
        for b_idx, batch in enumerate(batch_progress):

            del batch_inputs, batch_responses
            batch_responses = []
            batch_inputs = []
            _cleanup()
            
            batch = batch.to(DEVICE)  # Transfer the entire batch to GPU

            for image in batch:

                image = image[0:1][None].flip(1)

                if image.mean()>0.15:

                    limit = 1e-4
                    lr *= beta
                    lr = lr if lr>limit else limit

                    model.hebbian_lr = lr
                    model.homeo_lr = lr

                    _run_model(model, image)
                    model.hebbian_step()

                    if model.current_response.sum():

                        batch_responses.append(model.current_response.clone())
                        batch_inputs.append(model.current_input.clone())
                        code_tracker.append(model.current_response.clone())

            if len(batch_responses):
                
                batch_responses = torch.cat(batch_responses, dim=0)
                batch_inputs = torch.cat(batch_inputs, dim=0)

                reco_input = network['activ'](network['model'](batch_responses))

                targets = batch_inputs
                loss, loss_std = nn_loss(network, targets, reco_input)

                sim = cosim(targets.detach(), reco_input.detach())
                if reco_tracker is not None:
                    reco_tracker[b_idx] = sim

                avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim

                network['optim'].zero_grad()
                loss.backward()
                network['optim'].step()

                if b_idx%50==0:
                    ori_map, phase_map, mean_tc = get_orientations(
                        model.afferent_weights, gabor_size=model.rf_size)

                mean_activation = model.mean_activations.mean()
                mean_std = model.mean_activations.std() / model.homeo_target
                batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}, AS:{:.3f}'.format(
                    mean_activation, 
                    mean_std, 
                    avg_loss,
                    lr,
                    model.aff_strength.mean()
                ))

    return model, network, code_tracker


def run_noise_sweeps(n_reps=1):
    with keep.running():
        collect_noise_stats(minicolumnar=True, n_reps=n_reps)
        collect_noise_stats(minicolumnar=False, n_reps=n_reps)

if __name__ == '__main__':
    run_noise_sweeps()
