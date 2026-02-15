import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

from wiring_efficiency_utils import *

class NeuralSheet(nn.Module):
    def __init__(
        self, 
        input_size,
        sheet_size, 
        R_rf,
        R_long,
        homeo_target=0.04,
        lat_norm=0.32,
        aff_norm=0.7,
        iterations=30,
        lr=1e-3,
        microcolumnar=True,
        device='cuda'
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
        self.aff_pad = self.rf_size
        self.R_long = R_long

        self.aff_cutoff = get_circle(self.rf_size, self.rf_size/2).float().to(device)

        if microcolumnar:
            a=1.8
            b=2
            self.se_cutoff = generate_circles(sheet_size, sheet_size, R_long/a/b/3.7).to(device)
            self.i_cutoff = generate_circles(sheet_size, sheet_size, R_long/a/b).to(device)
            self.le_cutoff = generate_circles(sheet_size, sheet_size, R_long/a).to(device) 

        else:
            self.se_cutoff = generate_gaussians(sheet_size, sheet_size, R_long/5).to(device)
            self.i_cutoff = generate_circles(sheet_size, sheet_size, R_long).to(device)

        
        afferent_weights = torch.rand((sheet_size**2, 2, self.rf_size, self.rf_size), device=device)
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)        
        self.afferent_weights = afferent_weights
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations + 0
        self.lateral_correlations_exc = lateral_correlations + 0

        self.aff_sparsity = torch.rand(afferent_weights.shape, device=self.device) < 1
        self.lat_sparsity = torch.rand(lateral_correlations.shape, device=self.device) < 1

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

        r = int(self.R_long//a+1 if self.microcolumnar else self.R_long)
        self.window = oddenise(r*2)

        self.init_indices()
        self.rf_grids = get_grids(input_size, input_size, self.rf_size, sheet_size, device=device)
        
        self.slicing_var = torch.zeros(
            (self.sheet_size**2, 3, sheet_size+r*2, sheet_size+r*2), 
            device=device
        )

        self.delta_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

    def forward(
        self, 
        input_crop,
        noise_gamma=0, 
        adaptation=True
    ):
        
        self.current_response *= 0
        self.response_tracker *= 0
        
        net_afferent = 0
        break_flag = False
        
        current_input = input_crop
        self.current_input = input_crop
        self.current_tiles = extract_patches(current_input, self.rf_grids)       
        afferent = self.current_tiles * self.get_aff_weights()
        afferent = afferent.sum([1,2,3])
        self.current_afferent = afferent.view(self.current_response.shape) 

        net_afferent = (self.current_afferent) * self.aff_strength

        self.update_interactions()
        
        pad_amount = self.window // 2

        se_padded = F.pad(self.s_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
        self.slicing_var[:, 0:1] = se_padded 
        
        i_padded = F.pad(self.inh, (pad_amount, pad_amount, pad_amount, pad_amount))
        self.slicing_var[:, 1:2] = i_padded

        if self.microcolumnar:
                    
            le_padded = F.pad(self.l_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 2:3] = le_padded

        sliced_interactions = self.slicing_var[self.batch_indices, :, self.final_row_indices, self.final_col_indices]

        se_crops = sliced_interactions[:,:,:,:,0]
        i_crops = sliced_interactions[:,:,:,:,1]
        le_crops = sliced_interactions[:,:,:,:,2]

        if self.microcolumnar:

            lat_interactions = (0.32*se_crops + 0.8*le_crops) - i_crops #.32 / .8

        else:

            lat_interactions = se_crops - i_crops

        for i in range(self.iterations):

            if noise_gamma:

                if i==0:
                    self.noise = torch.randn(self.current_response.shape, device=self.device) 

                else:
                    curr_noise = torch.randn(self.current_response.shape, device=self.device) 
                    beta = 0.8
                    self.noise = self.noise * beta + curr_noise * (1-beta)
                    self.noise /= (self.noise**2).mean()**0.5

            else:
                self.noise = 0
            
            padded_response = F.pad(self.current_response, (self.window//2, self.window//2, self.window//2, self.window//2))
            res_tiles = F.unfold(padded_response, self.window)[0].T.view(-1,1,self.window,self.window)

            lateral_delta = (lat_interactions * res_tiles).sum([2,3]).view(self.current_response.shape) * self.gains
            
            #se = (se_crops * res_tiles).sum([2,3]).view(self.current_response.shape)
            #le = (le_crops * res_tiles).sum([2,3]).view(self.current_response.shape)
            
            update = net_afferent + lateral_delta 
                        
            self.current_response = torch.tanh(torch.relu(update - self.thresholds + self.noise * noise_gamma)) 

            self.response_tracker[i] = self.current_response + 0


        beta = self.current_response
        self.delta_mag = (1-beta) * self.delta_mag + beta * lateral_delta

        
        if adaptation:

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
            
                        
    def hebbian_step(self):

        self.step(self.afferent_weights, self.current_tiles, self.current_response.view(-1,1,1,1))
        self.step(self.lateral_correlations, self.current_response, self.current_response.view(-1,1,1,1))
        thresh = 0.05
        self.step(self.lateral_correlations_exc, self.current_response, (self.current_response.view(-1,1,1,1) - thresh), unlearning=0e-3)
        

    def step(self, weights, target, response, unlearning=0):

        delta = response * target - unlearning 
        weights += self.hebbian_lr * delta 
        weights *= weights > 0 
        weights /= weights.mean([1,2,3], keepdim=True) + 1e-11
        
        
    def get_aff_weights(self):

        aff_weights = self.afferent_weights * self.aff_cutoff * self.aff_sparsity
        aff_weights /= aff_weights.sum([1,2,3], keepdim=True) + 1e-11
        return aff_weights
        

    def update_interactions(self):

        self.inh = self.i_cutoff * self.lateral_correlations #* (1 - self.se_cutoff)
        self.inh /= self.inh.sum([2,3], keepdim=True)

        self.s_exc = self.se_cutoff + 0 #* self.lateral_correlations
        self.s_exc /= self.s_exc.sum([2,3], keepdim=True)

        if self.microcolumnar:

            self.l_exc = self.le_cutoff * self.lateral_correlations_exc * self.lat_sparsity
            self.l_exc /= self.l_exc.sum([2,3], keepdim=True) + 1e-11


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
