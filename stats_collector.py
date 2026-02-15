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
from wakepy import keep

# python3 -c 'import stats_collector; stats_collector.collect_dim_stats()' -m wakepy
#@profile
def collect_stats():
    
    # Example usage
    crop_size = 30 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = './input_stimuli'  # Path to your image folder
    device = 'cuda'  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 5
    beta = 1 - 5e-5
    loss_beta = 1e-2

    dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
    
    trials = 10
    epochs = 2
    
    trialvar = np.sqrt(np.linspace(6**2, 18**2, trials))
    sizesvar = [30,60,90]
    N_CODES = (sizesvar[-1]+1)**2
    sizes = len(sizesvar)
    reco_tracker = torch.zeros((sizes, trials, len(dataloader)))
    spectrum_tracker = torch.zeros((sizes, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((sizes, trials))

    se_pca_tracker = torch.zeros((sizes, trials))

    #------------------------- running simulations
    code_tracker = []
    for s in range(sizes):
        
        for t in range(trials):
                
            dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
            print('cropsize: ', crop_size)

            print('running simulation with size: ' + str(float(sizesvar[s])) \
                  + ' interaction radius: ' + str(float(trialvar[t])))

            model = NeuralSheet(crop_size, sizesvar[s], R_rf, R_long=trialvar[t], device=device, microcolumnar=False).to(device)
                
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
        
                    del batch_inputs, batch_responses
                    batch_responses = []
                    batch_inputs = []
                    torch.cuda.empty_cache()
                    
                    batch = batch.to('cuda')  # Transfer the entire batch to GPU
        
                    for image in batch:
        
                        image = image[0:1][None].flip(1)
        
                        if image.mean() > 0.15:
        
                            limit = 1e-4
                            lr *= beta
                            lr = lr if lr>limit else limit
        
                            model.hebbian_lr = lr * 1e2
                            model.homeo_lr = lr
        
                            model(image)
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
                        reco_tracker[s, t, b_idx] = sim
        
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

            code_tracker = torch.cat(code_tracker, dim=0)

            mask = torch.isnan(code_tracker).any(dim=(-2, -1))
            print('finished training, number of Nan found: ' + str(int(mask.sum())))

            _, spectrum, peak = get_effective_dims(code_tracker[-N_CODES:])
            eff_dims_pca, samp_components = get_pca_dimensions(code_tracker[-N_CODES:])

            se_pca_tracker[s, t] = eff_dims_pca

            print('training complete, accuracy: ' + str(float(reco_tracker[s, t, -N_CODES:].mean())) + ' dimensionality: ' \
                  + str(eff_dims_pca))

            spectrum_tracker[s,t,:sizesvar[s],:sizesvar[s]] = spectrum.cpu()
            peak_tracker[s,t] = peak.cpu()                    

    data = {
        'reco_tracker' : reco_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'se_pca_tracker': se_pca_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar
    }


    torch.save(data, 'data_accdim.pt')
    time.sleep(5)
    #os.system("shutdown -h 0")

def collect_noise_stats(minicolumnar=True):
    
    # Example usage
    crop_size = 30 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = './input_stimuli'  # Path to your image folder
    device = 'cuda'  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 5
    beta = 1 - 5e-5
    loss_beta = 1e-2

    dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
    
    trials = 15
    n_conditions = 15
    epochs = 2

    if minicolumnar:
        trialvar = np.sqrt(np.linspace(6**2, 18**2, trials))
    else:
        trialvar = np.sqrt(np.linspace(2**2, 18**2, trials))
    
    sizesvar = np.round(trialvar * 5.astype(int)
    noise_conditions = np.linspace(0, 0.1, n_conditions)
    N_CODES = (sizesvar[-1]+1)**2
    reco_tracker = torch.zeros((trials, len(dataloader)))
    se_tracker = torch.zeros((trials))
    spectrum_tracker = torch.zeros((trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((trials))

    se_pca_tracker = torch.zeros((trials))

    noise_acc_tracker = torch.zeros((trials, n_conditions))
    noise_dim_tracker = torch.zeros((trials, n_conditions))
    noise_rob_tracker = torch.zeros((trials, n_conditions))

    #------------------------- running simulations
    code_tracker = []
        
    for t in range(trials):
            
        dataloader = create_dataloader(root_dir, crop_size, batch_size, num_workers)
        print('cropsize: ', crop_size)

        print('running simulation with size: ' + str(float(sizesvar[t])) \
              + ' interaction radius: ' + str(float(trialvar[t])))

        if minicolumnar:
            model = NeuralSheet(crop_size, sizesvar[t], R_rf, R_long=trialvar[t], device=device, microcolumnar=True, range_norm=0.35).to(device)
        else:
            model = NeuralSheet(crop_size, sizesvar[t], R_rf, R_long=trialvar[t], device=device, microcolumnar=False, range_norm=0.32).to(device)
            
        lr = 1e-3
        network = init_nn(sizesvar[t], crop_size)
        avg_loss = 0
        code_tracker = []
        batch_responses = []
        batch_inputs = []
        gc.collect()
    
        for e in range(epochs):

            batch_progress = tqdm(dataloader, leave=False)
            for b_idx, batch in enumerate(batch_progress):


                batch_responses = []
                batch_inputs = []
                batch = batch.to('cuda')  # Transfer the entire batch to GPU

                for image in batch:

                    image = image[0:1][None].flip(1)

                    if image.mean()>0.15:

                        limit = 1e-4
                        lr *= beta
                        lr = lr if lr>limit else limit
                        model.hebbian_lr = lr * 1e2
                        model.homeo_lr = lr

                        model(image, adaptation=True)
                        model.hebbian_step()
                        
                        batch_responses.append(model.current_response.clone())
                        batch_inputs.append(model.current_input.clone())
                        code_tracker.append(model.current_response.clone())

                batch_responses = torch.cat(batch_responses, dim=0)
                batch_inputs = torch.cat(batch_inputs, dim=0)

                reco_input = network['activ'](network['model'](batch_responses))[:,:,R_rf:-R_rf,R_rf:-R_rf]
                targets = batch_inputs[:,:,R_rf:-R_rf,R_rf:-R_rf]
                
                loss, loss_std = nn_loss(network, targets, reco_input)
                
                sim = cosim(targets.detach().cpu(), reco_input.detach().cpu(), True)
                reco_tracker[t, b_idx] = sim
                                
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

        code_tracker = torch.cat(code_tracker, dim=0)

        mask = torch.isnan(code_tracker).any(dim=(-2, -1))
        print('finished training, number of Nan found: ' + str(int(mask.sum())))

        eff_dims, spectrum, peak = get_effective_dims(code_tracker[-N_CODES:])
        eff_dims_pca, samp_components = get_pca_dimensions(code_tracker[-N_CODES:])

        se_tracker[t] = eff_dims
        se_pca_tracker[t] = eff_dims_pca

        print('training complete, accuracy: ' + str(float(reco_tracker[t, -N_CODES:].mean())) + ' dimensionality: ' \
              + str(eff_dims_pca))

        spectrum_tracker[t,:sizesvar[t],:sizesvar[t]] = spectrum.cpu()
        peak_tracker[t] = peak.cpu()           

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
            gc.collect()

            for b_idx, batch in enumerate(batch_progress):

                torch.cuda.empty_cache()
                
                batch = batch.to('cuda')  # Transfer the entire batch to GPU

                for image in batch:

                    image = image[0:1][None].flip(1)

                    if image.mean() > 0.15:

                        model(image, adaptation=False)

                        p = int(trialvar[t]//2)
                        if model.current_response[:,:,p:-p-1,p:-p-1].mean() > 0.05:

                            code_tracker.append(model.current_response.clone())
                            input_tracker.append(model.current_input.clone())
                            
                            model(
                                image,
                                noise_gamma=noise_gamma, 
                                adaptation=False
                            )
                            perturbed_code_tracker.append(model.current_response.clone())

                if len(perturbed_code_tracker)>N_CODES:  
                    break

            input_tracker = torch.cat(input_tracker, dim=0)
            code_tracker = torch.cat(code_tracker, dim=0)
            perturbed_code_tracker = torch.cat(perturbed_code_tracker, dim=0)

            # normalising perturbations to simulate homeostasis
            #perturbed_code_tracker /= perturbed_code_tracker.sum([-2, -1], keepdim=True) + 1e-11
            #perturbed_code_tracker *= code_tracker.sum([-2, -1], keepdim=True)
            
            mask = torch.isnan(perturbed_code_tracker).any(dim=(-2, -1))

            eff_dims_pca, samp_components = get_pca_dimensions(perturbed_code_tracker)
            
            noise_dim_tracker[t, n_idx] = eff_dims_pca

            reco_input = network['activ'](network['model'](perturbed_code_tracker))
            accuracy = cosim(reco_input.detach(), input_tracker.detach())
            noise_acc_tracker[t, n_idx] = accuracy

            robustness = cosim(code_tracker.detach()[:,:,p:-p-1,p:-p-1], perturbed_code_tracker.detach()[:,:,p:-p-1,p:-p-1])
            noise_rob_tracker[t, n_idx] = robustness

            print('measuring noise robustness, noise: ' + str(noise_gamma) + ', robustness: ' + str(float(robustness)))

            if robustness < 0.9:
                break

    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'se_pca_tracker': se_pca_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar,
        'noise_conditions' : noise_conditions,
        'noise_acc': noise_acc_tracker,
        'noise_dim': noise_dim_tracker,
        'noise_rob': noise_rob_tracker
    }


    if minicolumnar:
        torch.save(data, 'data_noise_sp.pt')
    else:
        torch.save(data, 'data_noise_topo.pt')
        
    time.sleep(5)

            
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

with keep.running():
    collect_noise_stats(minicolumnar=True)
    collect_noise_stats(minicolumnar=False)
    #collect_stats() 
os.system("shutdown -h 0")