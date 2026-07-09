import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

from helpers.wiring_efficiency_utils import *

class NeuralSheet(nn.Module):
    def __init__(
        self, 
        input_size,
        sheet_size, 
        R_rf,
        R_long,
        homeo_target=0.05,
        lat_norm=0.3,
        aff_norm=0.7,
        iterations=30,
        lr=1e-3,
        microcolumnar=True,
        device='cuda',
        p = []
    ):
        super().__init__()

        self.lat_norm = lat_norm
        self.aff_norm = aff_norm
        self.homeo_lr = lr
        self.homeo_target = homeo_target
        self.iterations = iterations        
        self.sheet_size = sheet_size  
        self.input_size = input_size 
        self.device = device
        self.microcolumnar = microcolumnar

        self.rf_size = oddenise(R_rf*2)
        self.rf_size_l3 = oddenise(sheet_size/input_size*R_rf*2)
        self.rf_size_l3 = oddenise(R_long*2/1.8)
        self.aff_pad = self.rf_size
        self.R_long = R_long

        self.aff_cutoff = get_circle(self.rf_size, self.rf_size/2).float().to(device)
        self.aff_cutoff *= torch.rand(self.aff_cutoff.shape).to(device) > 0.
        self.aff_cutoff_l3 = get_circle(self.rf_size_l3, self.rf_size_l3/2).float().to(device)

        a=1.7
        b=1.5
        c=3

        if microcolumnar:
            self.se_cutoff = generate_circles(sheet_size, sheet_size, R_long/a/b/c).to(device)
            self.i_cutoff = generate_circles(sheet_size, sheet_size, R_long/a/b).to(device)
            self.le_cutoff = generate_circles(sheet_size, sheet_size, R_long/a).to(device) 
            #self.le_cutoff *= generate_gaussians(sheet_size, sheet_size, R_long/a).to(device)
            self.exc_b = 0.3

        else:
            self.se_cutoff = generate_circles(sheet_size, sheet_size, R_long/c).to(device)
            self.i_cutoff = generate_circles(sheet_size, sheet_size, R_long).to(device)
            
        
        afferent_weights = torch.rand((sheet_size**2, 2, self.rf_size, self.rf_size), device=device)
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)        
        self.afferent_weights = afferent_weights

        afferent_weights_l3 = torch.rand((sheet_size**2, 1, self.rf_size_l3, self.rf_size_l3), device=device)
        afferent_weights_l3 /= afferent_weights_l3.sum([2,3], keepdim=True)        
        self.afferent_weights_l3 = afferent_weights_l3
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations + 0
        self.lateral_correlations_exc = lateral_correlations + 0
        self.lateral_correlations_l3 = lateral_correlations + 0
        self.lateral_correlations_exc_l3 = lateral_correlations + 0
        self.lateral_correlations_l4_l3 = lateral_correlations + 0
        self.lateral_correlations_exc_l4_l3 = lateral_correlations + 0

        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)   
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)
        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.gains = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_aff = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_strength = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.avg_hist = torch.zeros(10) 
        self.noise = 0

        self.current_response_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)        
        self.response_tracker_l3 = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)
        self.mean_activations_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.thresholds_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_fr_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.gains_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_aff_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_strength_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.avg_hist_l3 = torch.zeros(10) 
        self.noise_l3 = 0

        r = int(self.R_long//a+1 if self.microcolumnar else self.R_long)
        self.window = oddenise(r*2)

        self.init_indices()
        self.rf_grids = get_grids(input_size, input_size, self.rf_size, sheet_size, jitter=0, device=device)
        self.rf_grids_l3 = get_grids(sheet_size+self.rf_size_l3, sheet_size, self.rf_size_l3, sheet_size, device=device)
        
        self.slicing_var = torch.zeros(
            (self.sheet_size**2, 5, sheet_size+r*2, sheet_size+r*2), 
            device=device
        )

        self.delta_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.eye = torch.eye(self.sheet_size**2).view(self.lateral_correlations.shape).to(device)

        if len(p):
            self.p1 = p[1]
            self.p2 = p[2]

    def forward(
        self, 
        input_crop,
        noise_gamma=0, 
        noise_beta=0.8,
        adaptation=True,
        sparsity=0,
        loc_sparsity=0,
        layer_3=False
    ):

        self._last_layer_3 = layer_3
        self.current_response *= 0
        self.response_tracker *= 0

        if layer_3:
            self.current_response_l3 *= 0
            self.response_tracker_l3 *= 0
            self.current_tiles_l3 = 0
        
        net_afferent = 0
        net_afferent_l3 = 0
        break_flag = False
        
        current_input = input_crop
        self.current_input = input_crop
        self.current_tiles = extract_patches(current_input, self.rf_grids)       
        afferent = self.current_tiles * self.get_aff_weights()
        afferent = afferent.sum([1,2,3])
        self.current_afferent = afferent.view(self.current_response.shape) 
        net_afferent = self.current_afferent * self.aff_strength

        if sparsity:
            self.lat_sparsity = torch.rand(self.lateral_correlations.shape, device=self.device) < (1-sparsity)
        else:
            self.lat_sparsity = 1

        if loc_sparsity:
            self.loc_lat_sparsity = torch.rand(self.lateral_correlations.shape, device=self.device) < (1-loc_sparsity)
        else:
            self.loc_lat_sparsity = 1
            
        self.update_interactions(layer_3=layer_3)
        
        pad_amount = self.window // 2

        se_padded = F.pad(self.s_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
        self.slicing_var[:, 0:1] = se_padded 
        
        i_padded = F.pad(self.inh, (pad_amount, pad_amount, pad_amount, pad_amount))
        self.slicing_var[:, 1:2] = i_padded

        if self.microcolumnar:
                    
            le_padded = F.pad(self.l_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 2:3] = le_padded / le_padded.sum([1,2,3], keepdim=True)

        if layer_3:

            i_padded_l4_l3 = F.pad(self.inh_l4_l3, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 3:4] = i_padded_l4_l3

            if self.microcolumnar:

                le_padded_l4_l3 = F.pad(self.l_exc_l4_l3, (pad_amount, pad_amount, pad_amount, pad_amount))
                self.slicing_var[:, 4:5] = le_padded_l4_l3

        n_channels = 3 if self.microcolumnar else 2
        if layer_3:
            n_channels = 5 if self.microcolumnar else 4

        sliced_interactions = self.slicing_var[self.batch_indices, :n_channels, self.final_row_indices, self.final_col_indices]

        se_crops = sliced_interactions[:,:,:,:,0]
        i_crops = sliced_interactions[:,:,:,:,1]

        if self.microcolumnar:
            
            le_crops = sliced_interactions[:,:,:,:,2]
            #lat_interactions = (self.exc_b*se_crops + (1-self.exc_b)*le_crops) - i_crops
            lat_interactions = self.exc_b*se_crops + (1-self.exc_b)*le_crops - i_crops

        else:

            lat_interactions = se_crops - i_crops

        if layer_3:

            i_crops_l4_l3 = sliced_interactions[:,:,:,:,3]
            local_int = self.s_exc_l3 - self.inh_l3

            if self.microcolumnar:
                l4_l3_int = 1.15*(self.exc_b*se_crops + (1-self.exc_b)*le_crops) - i_crops_l4_l3
            else:
                l4_l3_int = 1.5*se_crops - i_crops_l4_l3


        for i in range(self.iterations):

            if noise_gamma:

                if i==0:
                    self.noise = torch.randn(self.current_response.shape, device=self.device) 

                    if layer_3:
                        self.noise_l3 = torch.randn(self.current_response.shape, device=self.device) 

                else:
                    curr_noise = torch.randn(self.current_response.shape, device=self.device) 
                    self.noise = self.noise * noise_beta + curr_noise * (1-noise_beta)
                    self.noise /= (self.noise**2).mean()**0.5

                    if layer_3:
                        curr_noise_l3 = torch.randn(self.current_response.shape, device=self.device) 
                        self.noise_l3 = self.noise_l3 * noise_beta + curr_noise_l3 * (1-noise_beta)
                        self.noise_l3 /= (self.noise_l3**2).mean()**0.5

            else:
                self.noise = 0

                if layer_3:
                    self.noise_l3 = 0

            padded_response = F.pad(self.current_response, (self.window//2, self.window//2, self.window//2, self.window//2))
            res_tiles = F.unfold(padded_response, self.window)[0].T.view(-1,1,self.window,self.window)
            mex_hat = (lat_interactions * res_tiles).sum([2,3]).view(self.current_response.shape) 
            
            lateral_delta = mex_hat * self.gains
            
            #l4_l3_exc = (((0.33*se_crops + 0.8*le_crops) - i_crops) * res_tiles).sum([2,3]).view(self.current_response.shape) #* self.balance_l3
            if layer_3:

                net_afferent_l3 = (l4_l3_int * res_tiles).sum([2,3]).view(self.current_response.shape) * self.aff_strength_l3
                
                #if microcolumnar:
                #    l4_l3_exc = ((se_crops - i_crops) * res_tiles).sum([2,3]).view(self.current_response.shape) * self.aff_strength_l3 
                #else:
                #    l4_l3_exc = ((se_crops - i_crops) * res_tiles).sum([2,3]).view(self.current_response.shape) * self.aff_strength_l3 #* self.balance_l3
                #global_interaction = self.global_exc_l3 + self.global_inh_l3
                #local_interaction = self.inh_l3 
                #interaction_l3 = global_interaction*(1-self.balance_l3) - local_interaction*self.balance_l3
    
                loc_b = self.p1
                global_int = self.global_exc_l3 - self.global_inh_l3
                interaction_l3 = global_int * (1-loc_b) + local_int * loc_b
                lateral_delta_l3 = (interaction_l3 * self.current_response_l3).sum([1,2,3]).view(self.current_response_l3.shape)
            
            update = lateral_delta + net_afferent  
            self.current_response = torch.tanh(torch.relu(update - self.thresholds + self.noise * noise_gamma)) 
            self.response_tracker[i] = self.current_response + 0

            if layer_3:

                delta_l3 = lateral_delta_l3 * self.gains_l3 
    
                #b = 0.95
                #update_l3 = l4_l3_exc * (1-b) + delta_l3 * b
                update_l3 = net_afferent_l3 + delta_l3
                
                self.current_response_l3 = torch.tanh(torch.relu(update_l3 - self.thresholds_l3 + self.noise_l3 * noise_gamma)) 
                self.response_tracker_l3[i] = self.current_response_l3 + 0

        
        if adaptation:

            beta = self.current_response

            self.mean_fr = self.mean_fr * (1 - beta) + self.current_response * beta
            gap = (self.mean_fr.mean() - self.lat_norm) / self.lat_norm
            self.gains -= gap * self.homeo_lr
            self.gains = self.gains.clip(0)
            
            self.mean_aff = self.mean_aff * (1 - beta) + net_afferent * beta 
            gap = (self.mean_aff.mean() - self.aff_norm) / self.aff_norm
            self.aff_strength -= gap * self.homeo_lr
            self.aff_strength = self.aff_strength.clip(0)

            new_hist = np.histogram(self.current_response[self.current_response>0].cpu(), bins=10, range=(0,1))[0]
            self.avg_hist = self.avg_hist*(1-self.homeo_lr) + new_hist*self.homeo_lr

            self.mean_activations = self.mean_activations*(1-self.homeo_lr) + self.current_response*self.homeo_lr
            thresh_update = (self.homeo_target - self.mean_activations) / self.homeo_target
            self.thresholds -= thresh_update * self.homeo_lr
            self.thresholds = self.thresholds.clip(-1,1)

            if layer_3:

                beta = self.current_response_l3

                self.delta_mag = (1-beta) * self.delta_mag + beta * lateral_delta_l3
    
                self.lat_norm_l3 = self.lat_norm
                self.mean_fr_l3 = self.mean_fr_l3 * (1 - beta) + self.current_response_l3 * beta
                gap = (self.mean_fr_l3.mean() - self.lat_norm_l3) / self.lat_norm_l3
                self.gains_l3 -= gap * self.homeo_lr
                self.gains_l3 = self.gains_l3.clip(0)
                #self.gains_l3 = self.gains_l3 *0 + 2.8
    
                self.aff_norm_l3 = self.p2
                self.mean_aff_l3 = self.mean_aff_l3 * (1 - beta) + net_afferent_l3 * beta 
                gap = (self.mean_aff_l3.mean() - self.aff_norm_l3) / self.aff_norm_l3
                self.aff_strength_l3 -= gap * self.homeo_lr
                self.aff_strength_l3 = self.aff_strength_l3.clip(0)
                #elf.aff_strength_l3 = self.aff_strength_l3 *0 + 2
    
                new_hist = np.histogram(self.current_response_l3[self.current_response_l3>0].cpu(), bins=10, range=(0,1))[0]
                self.avg_hist_l3 = self.avg_hist_l3*(1-self.homeo_lr) + new_hist*self.homeo_lr

                self.homeo_target_l3 = self.homeo_target
                self.mean_activations_l3 = self.mean_activations_l3*(1-self.homeo_lr) + self.current_response_l3*self.homeo_lr
                thresh_update = (self.homeo_target_l3 - self.mean_activations_l3) / self.homeo_target_l3
                self.thresholds_l3 -= thresh_update * self.homeo_lr 
                self.thresholds_l3 = self.thresholds_l3.clip(-1,1)
            
                        
    def hebbian_step(self, layer_3=None):

        if layer_3 is None:
            layer_3 = getattr(self, '_last_layer_3', False)

        self.step(self.afferent_weights, self.current_tiles, self.current_response.view(-1,1,1,1))
        self.step(self.lateral_correlations, self.current_response, self.current_response.view(-1,1,1,1))
        thresh = 0.04
        self.step(self.lateral_correlations_exc, self.current_response, (self.current_response.view(-1,1,1,1) - thresh), unlearning=0e-3)

        if not layer_3:
            return

        self.step(self.afferent_weights_l3, self.current_tiles_l3, self.current_response_l3.view(-1,1,1,1))
        self.step(self.lateral_correlations_l3, self.current_response_l3, 4*self.current_response_l3.view(-1,1,1,1))

        self.step(self.lateral_correlations_l4_l3, self.current_response, self.current_response_l3.view(-1,1,1,1))
        self.step(self.lateral_correlations_exc_l4_l3, self.current_response, 4*self.current_response_l3.view(-1,1,1,1) - thresh)

        thresh = 0.06
        
        #s = 4
        #w = torch.exp(- s * torch.linspace(0, 1, self.iterations, device=self.device)).view(-1,1,1,1)
        #w /= w.sum()
        #self.trace = self.response_tracker_l3 * w
        #self.trace = self.trace.sum(0, keepdim=True)
        
        self.step(self.lateral_correlations_exc_l3, self.current_response_l3, 4*(self.current_response_l3.view(-1,1,1,1) - thresh), unlearning=0e-3)
        #self.step(self.lateral_correlations_exc_l3, self.trace, 10*(self.trace.view(-1,1,1,1) - thresh), unlearning=0e-3)

    def step(self, weights, target, response, unlearning=0):

        delta = response * target - unlearning 
        weights += self.hebbian_lr * delta 
        weights *= weights > 0 
        weights /= weights.mean([1,2,3], keepdim=True) + 1e-11
        
        
    def get_aff_weights(self):

        aff_weights = self.afferent_weights * self.aff_cutoff
        aff_weights /= aff_weights.sum([1,2,3], keepdim=True) + 1e-11
        return aff_weights

    def get_aff_weights_l3(self):

        aff_weights = self.afferent_weights_l3 * self.aff_cutoff_l3
        aff_weights /= aff_weights.sum([1,2,3], keepdim=True) + 1e-11
        return aff_weights
        

    def update_interactions(self, layer_3=True):

        self.inh = self.i_cutoff * self.lateral_correlations
        self.inh /= self.inh.sum([2,3], keepdim=True)

        self.s_exc = self.se_cutoff #* self.lateral_correlations
        self.s_exc /= self.s_exc.sum([2,3], keepdim=True)

        if self.microcolumnar:

            self.l_exc = self.le_cutoff * self.lateral_correlations_exc + 1e-11 #* self.lat_sparsity
            self.l_exc /= (self.l_exc).sum([2,3], keepdim=True)

        if not layer_3:
            return

        self.inh_l3 = self.i_cutoff * self.lateral_correlations_l3
        self.inh_l3 /= self.inh_l3.sum([2,3], keepdim=True)

        self.inh_l4_l3 = self.i_cutoff * self.lateral_correlations_l4_l3
        self.inh_l4_l3 /= self.inh_l4_l3.sum([2,3], keepdim=True)

        self.s_exc_l3 = self.se_cutoff * self.loc_lat_sparsity + 1e-11 * self.eye
        self.s_exc_l3 /= self.s_exc_l3.sum([2,3], keepdim=True)

        if self.microcolumnar:

            self.l_exc_l4_l3 = self.le_cutoff * self.lateral_correlations_exc_l4_l3
            self.l_exc_l4_l3 /= self.l_exc_l4_l3.sum([2,3], keepdim=True) + 1e-11

            self.l_exc_l3 = self.le_cutoff * self.lateral_correlations_exc_l3 * self.lat_sparsity + 1e-11
            self.l_exc_l3 /= (self.l_exc_l3).sum([2,3], keepdim=True)
            

        # not using it because it breaks sparsity
        adjust = (self.lateral_correlations_exc_l3**2).sum([1,2,3], keepdim=True)
        adjust /= adjust.mean()

        self.global_exc_l3 = self.lateral_correlations_exc_l3 * self.lat_sparsity + 1e-11
        self.global_exc_l3 /= (self.global_exc_l3).sum([1,2,3], keepdim=True) #* adjust
        
        self.global_inh_l3 = self.lateral_correlations_l3 * self.lat_sparsity + 1e-11
        self.global_inh_l3 /= (self.global_inh_l3).sum([1,2,3], keepdim=True) #* adjust


    def init_indices(self):

        N = self.sheet_size
        
        self.crop_indeces = torch.arange(N**2).to(self.device)

        num_images = N**2
        
        batch_indices = torch.arange(num_images).view(num_images, 1, 1, 1)
        # Create a batch dimension for indices
        self.batch_indices = batch_indices.expand(num_images, 1, self.window, self.window)
        
        # Generate all possible row and column starts
        row_indices = torch.arange(0, N).repeat_interleave(N)
        col_indices = torch.arange(0, N).repeat(N)

        # Expand indices to use for gathering
        row_indices = row_indices.view(num_images, 1, 1).expand(num_images, self.window, self.window)
        col_indices = col_indices.view(num_images, 1, 1).expand(num_images, self.window, self.window)
        
        # Create range tensors for MxM crops
        range_rows = torch.arange(0, self.window).view(1, self.window, 1).expand(num_images, self.window, self.window)
        range_cols = torch.arange(0, self.window).view(1, 1, self.window).expand(num_images, self.window, self.window)

        # Add start indices and range indices
        self.final_row_indices = (row_indices + range_rows).view(num_images, 1, self.window, self.window).to(self.device)
        self.final_col_indices = (col_indices + range_cols).view(num_images, 1, self.window, self.window).to(self.device)
