import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Image as IPImage
from wiring_efficiency_utils import *
from neuralsheet import *
from map_plotting import *
import os
import gc
import time

# python3 -c 'import stats_collector; stats_collector.collect_dim_stats()' -m wakepy
#@profile
def collect_stats(salt_and_pepper=True, accdim=False):
    
    # Example usage
    crop_size = 28 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = './input_stimuli'  # Path to your image folder
    device = 'cuda'  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 5
    beta = 1 - 5e-5
    loss_beta = 3e-3

    dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
    
    trials = 10
    n_conditions = 10
    epochs = 1
    n_samples = 3
    
    trialvar = np.sqrt(np.linspace(5**2, 20**2, trials))
    sizesvar = [56]
    N_CODES = sizesvar[-1]**2 + 100
    noise_conditions = torch.linspace(0, 0.5, n_conditions)
    plastic_sparsity_vals = torch.linspace(1, 20, n_conditions)
    sizes = len(sizesvar)
    reco_tracker = torch.zeros((sizes, trials, len(dataloader)))
    se_tracker = torch.zeros((sizes, trials))
    map_tracker = torch.zeros((sizes, trials, sizesvar[-1], sizesvar[-1]))
    spectrum_tracker = torch.zeros((sizes, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((sizes, trials))

    noise_acc_tracker = torch.zeros((sizes, trials, n_conditions))
    noise_dim_tracker = torch.zeros((sizes, trials, n_conditions))
    noise_rob_tracker = torch.zeros((sizes, trials, n_conditions))
    
    sparsity_acc_tracker = torch.zeros((sizes, trials, 1, n_conditions))
    sparsity_dim_tracker = torch.zeros((sizes, trials, 1, n_conditions))
    sparsity_rob_tracker = torch.zeros((sizes, trials, 1, n_conditions))

    comp_tracker = torch.zeros((sizes, trials, n_samples*sizesvar[-1], n_samples*sizesvar[-1]))
    se_pca_tracker = torch.zeros((sizes, trials))

    rob_accu = torch.zeros((trials))
    rob_dims = torch.zeros((trials))

    print(trialvar, noise_conditions, plastic_sparsity_vals)

    #------------------------- running simulations
    code_tracker = []
    for s in range(sizes):
        
        for t in range(trials):

            if not accdim:
                a = 5
                b = 2
                crop_size = a * (sizesvar[s] / trialvar[t] + b)
                crop_size = round(crop_size)
                
            dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
            print('cropsize: ', crop_size)

            print('running simulation with size: ' + str(float(sizesvar[s])) \
                  + ' interaction radius: ' + str(float(trialvar[t])))

            if salt_and_pepper:
                model = NeuralSheet(sizesvar[s], crop_size, R_rf, R_long=trialvar[t], device=device).to(device)
                model.range_norm = 0.16
            else:
                model = NeuralSheet(sizesvar[s], crop_size, R_rf, R_pat=trialvar[t], device=device).to(device)
                model.range_norm = 0.10
                
            lr = 1e-3
            network = init_nn(sizesvar[s], crop_size)
            avg_loss = 0
            code_tracker = []
            batch_responses = []
            batch_inputs = []
            gc.collect()
        
            for e in range(epochs):
        
                batch_progress = tqdm(dataloader, leave=False)
                del code_tracker
                code_tracker = []
                
                for b_idx, batch in enumerate(batch_progress):

                    model.update_interactions(1,1,1)
        
                    del batch_inputs, batch_responses
                    batch_responses = []
                    batch_inputs = []
                    torch.cuda.empty_cache()
                    
                    batch = batch.to('cuda')  # Transfer the entire batch to GPU
        
                    for image in batch:
        
                        image = image[0:1][None].flip(1)
        
                        if image.mean()>0.15:
        
                            limit = 1e-4
                            lr *= beta
                            lr = lr if lr>limit else limit
        
                            model.hebbian_lr = lr
                            model.homeo_lr = lr
        
                            model(image, noise_lvl=0)
                            model.hebbian_step()
        
                            if model.current_response.sum():
        
                                batch_responses.append(model.current_response+0)
                                batch_inputs.append(model.current_input+0)
                                code_tracker.append(model.current_response+0)
        
                    if len(batch_responses):
                        
                        batch_responses = torch.cat(batch_responses, dim=0)
                        batch_inputs = torch.cat(batch_inputs, dim=0)
        
                        reco_input = network['activ'](network['model'](batch_responses))
        
                        targets = batch_inputs
                        loss, loss_std = nn_loss(network, targets, reco_input)
        
                        sim = cosim(targets.detach(), reco_input.detach())
                        reco_tracker[s, t, b_idx] = sim
        
                        avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim
        
                        network['optim'].zero_grad()
                        loss.backward()
                        network['optim'].step()
        
                        #if b_idx%50==0:
                        #    ori_map, phase_map, mean_tc = get_orientations(
                        #        model.afferent_weights, gabor_size=model.rf_size)
        
                        mean_activation = model.mean_activations.mean()
                        mean_std = model.mean_activations.std() / model.homeo_target
                        batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}, SP:{:.3f}, B:{:.3f} S:{:.3f} AS:{:.3f}'.format(
                            mean_activation, 
                            mean_std, 
                            avg_loss,
                            lr,
                            model.spread.mean(),
                            model.b.mean(),
                            model.gains.mean(),
                            model.aff_strength
                        ))

            #------------------------- accuracy-dimensionality measurements

            code_tracker = torch.cat(code_tracker, dim=0)

            mask = torch.isnan(code_tracker).any(dim=(-2, -1))
            print('finished training, number of Nan found: ' + str(int(mask.sum())))
            
            #ori_map, phase_map, mean_tc = get_orientations(model.afferent_weights, gabor_size=model.rf_size)
            #ori_map = ori_map.view(sizesvar[s], sizesvar[s]).cpu()

            eff_dims, spectrum, peak = get_effective_dims(code_tracker[-N_CODES:])
            eff_dims_pca, samp_components = get_pca_dimensions(code_tracker[-N_CODES:], n_samples)

            se_tracker[s, t] = eff_dims
            se_pca_tracker[s, t] = eff_dims_pca

            comp_size = n_samples * sizesvar[s]
            comp_tracker[s, t, :comp_size, :comp_size] = samp_components

            print('training complete, accuracy: ' + str(float(reco_tracker[s, t, -N_CODES:].mean())) + ' dimensionality: ' \
                  + str(eff_dims_pca))

            #map_tracker[s, t,:sizesvar[s],:sizesvar[s]] = ori_map
            spectrum_tracker[s,t,:sizesvar[s],:sizesvar[s]] = spectrum.cpu()
            peak_tracker[s,t] = peak.cpu()

            if s==(sizes-1) and not accdim:
    
                #------------------------- noise robustness measurements
    
                print('collecting noise robustness measurements!')
    
                input_tracker = []
                code_tracker = []
                perturbed_code_tracker = []
                    
                for n_idx, noise_lvl in tqdm(enumerate(noise_conditions)):

                    del input_tracker, perturbed_code_tracker, code_tracker
                    input_tracker = []
                    code_tracker = []
                    perturbed_code_tracker = []
                    gc.collect()

                    for b_idx, batch in enumerate(batch_progress):

                        model.update_interactions(1,1,1)
                        torch.cuda.empty_cache()
                        
                        batch = batch.to('cuda')  # Transfer the entire batch to GPU
    
                        for image in batch:
    
                            image = image[0:1][None].flip(1)
    
                            if image.mean()>0.15:

                                model(image, adaptation=False)

                                if model.current_response.mean() > 0.04:

                                    code_tracker.append(model.current_response.clone())
                                    input_tracker.append(model.current_input.clone())
                                    
                                    model(
                                        image,
                                        noise_lvl=noise_lvl, 
                                        adaptation=False
                                    )
                                    perturbed_code_tracker.append(model.current_response.clone())

                        if len(perturbed_code_tracker)>N_CODES:  
                            break

                    input_tracker = torch.cat(input_tracker, dim=0)
                    code_tracker = torch.cat(code_tracker, dim=0)
                    perturbed_code_tracker = torch.cat(perturbed_code_tracker, dim=0)

                    # normalising perturbations to simulate homeostasis
                    perturbed_code_tracker /= perturbed_code_tracker.sum([-2, -1], keepdim=True) + 1e-11
                    perturbed_code_tracker *= code_tracker.sum([-2, -1], keepdim=True)
                    
                    mask = torch.isnan(perturbed_code_tracker).any(dim=(-2, -1))

                    eff_dims_pca, samp_components = get_pca_dimensions(perturbed_code_tracker, n_samples)
                    
                    noise_dim_tracker[s, t, n_idx] = eff_dims_pca

                    reco_input = network['activ'](network['model'](perturbed_code_tracker))
                    accuracy = cosim(reco_input.detach(), input_tracker.detach())
                    noise_acc_tracker[s, t, n_idx] = accuracy

                    robustness = cosim(code_tracker.detach(), perturbed_code_tracker.detach())
                    noise_rob_tracker[s, t, n_idx] = robustness

                    print('measuring noise robustness, accuracy: ' + str(float(accuracy)))

                    if robustness < 0.9:
                        break
        
                #------------------------- connection sparsity measurements
    
                print('collecting sparsity measurements')
    
                input_tracker = []
                code_tracker = []
                perturbed_code_tracker = []
                    
                for p_idx, plastic_sparsity in tqdm(enumerate(plastic_sparsity_vals)): 

                    cutoff = model.long_cutoff if salt_and_pepper else model.mid_cutoff
                    
                    if (cutoff.sum([2,3]).min() / plastic_sparsity) < 1:
                        break

                    del input_tracker, perturbed_code_tracker, code_tracker
                    input_tracker = []
                    code_tracker = []
                    perturbed_code_tracker = []
                    gc.collect()

                    for b_idx, batch in enumerate(batch_progress):

                        model.update_interactions(1,1,1)
                        torch.cuda.empty_cache()
                        
                        batch = batch.to('cuda')  # Transfer the entire batch to GPU
    
                        for image in batch:
    
                            image = image[0:1][None].flip(1)
    
                            if image.mean()>0.15:

                                model(image, adaptation=False)

                                if model.current_response.mean() > 0.04:

                                    code_tracker.append(model.current_response.clone())
                                    input_tracker.append(model.current_input.clone())

                                    if salt_and_pepper:
                                        model(
                                            image, 
                                            phi_long=plastic_sparsity,
                                            adaptation=False
                                        )
                                    else:
                                        o = 10
                                        short_sparsity = (plastic_sparsity + o - 1) / o
                                        
                                        model(
                                            image, 
                                            phi_mid=plastic_sparsity,
                                            phi_short=short_sparsity,
                                            adaptation=False
                                        )
                                        
                                    perturbed_code_tracker.append(model.current_response.clone())
       
                            if len(perturbed_code_tracker)>N_CODES:  
                                break

                            if len(perturbed_code_tracker)%100==0:
                                model.update_short_sparsity(model.phi_short)
                                model.update_mid_sparsity(model.phi_mid)
                                model.update_long_sparsity(model.phi_long)
    
                    input_tracker = torch.cat(input_tracker, dim=0)
                    code_tracker = torch.cat(code_tracker, dim=0)
                    perturbed_code_tracker = torch.cat(perturbed_code_tracker, dim=0)

                    # normalising perturbations to simulate homeostasis
                    perturbed_code_tracker /= perturbed_code_tracker.sum([-2, -1], keepdim=True) + 1e-11
                    perturbed_code_tracker *= code_tracker.sum([-2, -1], keepdim=True)

                    mask = torch.isnan(perturbed_code_tracker).any(dim=(-2, -1))

                    eff_dims_pca, samp_components = get_pca_dimensions(perturbed_code_tracker, n_samples)
                    
                    sparsity_dim_tracker[s, t, 0, p_idx] = eff_dims_pca

                    reco_input = network['activ'](network['model'](perturbed_code_tracker))
                    accuracy = cosim(reco_input.detach(), input_tracker.detach())
                    sparsity_acc_tracker[s, t, 0, p_idx] = accuracy

                    robustness = cosim(code_tracker.detach(), perturbed_code_tracker.detach())
                    sparsity_rob_tracker[s, t, 0, p_idx] = robustness

                    print('sparsity_robustness', robustness)

                    if robustness < 0.9:
                        break
    
                    

    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'map_tracker' : map_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'comp_tracker': comp_tracker,
        'se_pca_tracker': se_pca_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar,
        'noise_conditions' : noise_conditions,
        'sparsity_conditions' : plastic_sparsity_vals,
        'noise_acc': noise_acc_tracker,
        'noise_dim': noise_dim_tracker,
        'noise_rob': noise_rob_tracker,
        'sparsity_acc': sparsity_acc_tracker,
        'sparsity_dim': sparsity_dim_tracker,
        'sparsity_rob': sparsity_rob_tracker
    }

    if salt_and_pepper:
        torch.save(data, 'data_sp.pt')
        time.sleep(5)
        collect_stats(salt_and_pepper=False)
        #os.system("shutdown -h 0")

    else:
        torch.save(data, 'data_topo.pt')
        time.sleep(5)
        os.system("shutdown -h 0")

def train_map(sheet_size, crop_size, epochs, dataloader, beta, model, reco_tracker):
    lr = 1e-3
    network = init_nn(sheet_size, crop_size)
    avg_loss = 0
    code_tracker = []
    batch_responses = []
    batch_inputs = []
    gc.collect()

    for e in range(epochs):

        batch_progress = tqdm(dataloader, leave=False)
        del code_tracker
        code_tracker = []
        
        for b_idx, batch in enumerate(batch_progress):

            del batch_inputs, batch_responses
            batch_responses = []
            batch_inputs = []
            torch.cuda.empty_cache()
            
            batch = batch.to('cuda')  # Transfer the entire batch to GPU

            for image in batch:

                image = image[0:1][None].flip(1)

                if image.mean()>0.15:

                    limit = 1e-4
                    lr *= beta
                    lr = lr if lr>limit else limit

                    model.hebbian_lr = lr
                    model.homeo_lr = lr

                    model(image)
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
                reco_tracker[s, t, b_idx] = sim

                avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim

                network['optim'].zero_grad()
                loss.backward()
                network['optim'].step()

                if b_idx%50==0:
                    ori_map, phase_map, mean_tc = get_orientations(
                        model.afferent_weights, gabor_size=model.rf_size)

                mean_activation = model.mean_activations.mean()
                mean_std = model.mean_activations.std() / model.homeo_target
                batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}, SP:{:.3f}, B:{:.3f} S:{:.3f} AS:{:.3f}'.format(
                    mean_activation, 
                    mean_std, 
                    avg_loss,
                    lr,
                    model.spread.mean(),
                    model.b,
                    model.strength,
                    model.aff_strength
                ))
       
#collect_stats()  #used for profiling