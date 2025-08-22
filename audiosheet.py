import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from PIL import Image
import random

from wiring_efficiency_utils import *

class NeuralSheet(nn.Module):
    def __init__(
        self, 
        sheet_size, 
        channels, 
        time_window, 
        time_px,
        R_rf, 
        R_pat, 
        device,
        cutoff=5,
    ):
        super().__init__()

        #self.range_norm = 0.1
        self.homeo_lr = 1e-3
        self.homeo_target = 0.04
        self.aff_target = 0.025
        self.target_range = 0.3
        self.aff_unlearning = torch.tensor([0]).view(1,-1,1,1).to(device)
        self.lat_unlearning = torch.tensor([0]).to(device)
        self.iterations = 20
        self.window = R_pat*2 + 1
        
        self.sheet_size = sheet_size  # Size of the sheet
        self.channels = channels
        self.time_window = time_window
        self.device = device

        # Afferent (receptive field) weights for each neuron in the sheet
        self.rf_size = R_rf

        std_exc = R_pat / 5
        self.std_exc = std_exc
        self.R_pat = R_pat
        self.time_px = time_px

        self.grids, self.strides = self.get_grids(time_window, channels, R_rf, time_px, sheet_size)

        self.time_trace = torch.exp(-torch.linspace(0, 3, time_px)).view(1,1,-1).to(device).flip(-1)
        
        afferent_weights = torch.rand((sheet_size**2, 2, self.rf_size, self.time_px), device=device)
        afferent_weights[:,1] = 1
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_exc).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc

        self.mri_mask = 1 - self.lateral_weights_exc / self.lateral_weights_exc.view(sheet_size**2, -1).max(1)[0].view(-1,1,1,1)

        self.eye = torch.eye(sheet_size**2).view(sheet_size**2, 1, sheet_size, sheet_size).to(device)

        self.mid_cutoff = generate_circles(sheet_size, sheet_size, R_pat).to(device) 
        self.short_cutoff = generate_circles(sheet_size, sheet_size, std_exc*cutoff/2).to(device)
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations
                
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.prev_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.response_tracker = torch.zeros(100, 1, sheet_size, sheet_size, device=device)

        self.mean_afferent = torch.zeros(1, 1, sheet_size, sheet_size, device=device) #+ self.aff_target
        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.pos_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device) #+ self.target_range
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.trace_tracker = torch.zeros(1, 1, sheet_size, sheet_size, device=device) #+ self.aff_target

        self.gains = torch.ones(1, 1, sheet_size, sheet_size, device=device) * 2
        self.aff_strength = 0.5

        self.avg_hist = torch.zeros(10) 

        self.mid_range_inh = torch.tensor([0]).to(device)
        self.short_range_exc = torch.tensor([0]).to(device)

        self.init_indices()

    def forward(
        self, 
        input_crop
    ):
        
        #self.current_response *= 0
        
        net_afferent = 0
        break_flag = False
        
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        current_input = input_crop
        self.current_input = input_crop
        stride = (input_crop.shape[0] - self.rf_size) // self.sheet_size

        self.current_tiles = self.extract_patches(input_crop[None,None]).view(self.sheet_size**2, 1, self.rf_size, self.time_px)
        
        afferent = self.current_tiles * self.get_aff_weights()
        afferent = afferent.sum([2,3])
        self.current_afferent = (afferent[:,0] - afferent[:,1]).view(self.current_response.shape)

        self.current_response = torch.relu(self.current_afferent - self.thresholds)

        self.short_range_exc = self.lateral_weights_exc
        self.mid_range_inh = self.lateral_correlations * self.mri_mask * self.mid_cutoff
        self.mid_range_inh = self.mid_range_inh / self.mid_range_inh.sum([2,3], keepdim=True)
        interactions = self.short_range_exc - self.mid_range_inh

        interactions = F.pad(interactions, (self.window//2, self.window//2, self.window//2, self.window//2))
        crops = interactions[self.batch_indices, :, self.final_row_indices, self.final_col_indices]

        for i in range(self.iterations):

            padded_response = F.pad(self.current_response, (self.window//2, self.window//2, self.window//2, self.window//2))
            res_tiles = F.unfold(padded_response, self.window)[0].T
            lateral = (res_tiles * crops.view(res_tiles.shape)).sum(1)
                                                
            lateral = lateral.view(self.current_response.shape)
            net_afferent = (self.current_afferent - self.thresholds) * self.aff_strength    
            
            update = net_afferent + lateral 
                        
            self.current_response = torch.relu(update) * self.gains
            self.current_response = torch.tanh(self.current_response) 

            if i % 10 == 0:
                max_change = (self.current_response - self.prev_response).abs().max()
                if max_change < 3e-3:
                    break_flag = True
            
            self.prev_response = self.current_response + 0

            if break_flag:
                break_flag = False
                break

        self.response_tracker = self.response_tracker.roll(1, dims=0)
        self.response_tracker[-1] = self.current_response

        trace_beta = 1/20
        self.trace_tracker = self.trace_tracker * (1-trace_beta) + self.current_response * trace_beta

        slow_lr = self.homeo_lr / self.time_window * 10
        fast_lr = self.homeo_lr

        pos_aff_mask = net_afferent>0
        self.mean_afferent[pos_aff_mask] = self.mean_afferent[pos_aff_mask]*(1-fast_lr) + net_afferent[pos_aff_mask]*fast_lr
        self.mean_activations = self.mean_activations*(1-slow_lr) + self.current_response*slow_lr
        pos_act_mask = self.current_response>0
        
        if pos_act_mask.any():
            
            self.pos_activations[pos_act_mask] = self.pos_activations[pos_act_mask]*(1-fast_lr) + self.current_response[pos_act_mask]*fast_lr

            new_hist = np.histogram(self.current_response[pos_act_mask].cpu(), bins=10, range=(0,1))[0]           
            self.avg_hist = self.avg_hist*(1-fast_lr) + new_hist*fast_lr
            

        gap = (self.pos_activations - self.target_range) / self.target_range
        #print(gap)
        self.gains -= gap * self.homeo_lr * 1e-1
        self.gains = self.gains.clip(0)
   
        gap = (self.mean_afferent.mean() - self.aff_target) / self.aff_target
        self.aff_strength -= gap * self.homeo_lr * 1e-1
        self.aff_strength = self.aff_strength.clip(0)

        gap = (self.homeo_target - self.mean_activations) / self.homeo_target
        self.thresholds -= gap * self.homeo_lr * 1e-2
            
                        
    def hebbian_step(self):

        diff = (self.homeo_target - self.mean_activations) / self.homeo_target
        afferent_contributions = self.current_tiles.repeat(1,2,1,1) 
        
        response = (self.current_response - self.trace_tracker).view(-1,1,1,1).repeat(1,2,1,1)
        response[:,1] *= -1
        response = torch.relu(response)
        
        #print(diff.min(), diff.max(), response.shape, diff.shape)
        #response[:,1] = -response[:,1] + 0.1
        #print(response[:,0].max(),response[:,1].max())
        self.step(self.afferent_weights, afferent_contributions, response)
        
        contributions = self.current_response
        self.step(self.lateral_correlations, contributions, self.current_response)
        
        

    def step(self, weights, target, response, unlearning=0):

        delta = response.view(-1,response.shape[1],1,1) * target 
        weights += self.hebbian_lr * delta # add new changes
        weights -= unlearning
        weights *= weights > 0 # clear weak weights
        weights /= weights.sum([2,3], keepdim=True) + 1e-11 # normalise remaining weights
        
    def get_aff_weights(self):
        
        aff_weights = self.afferent_weights.clone()
        
        return aff_weights

    def init_indices(self):

        N = self.sheet_size
        
        self.crop_indeces = torch.arange(N**2).to(self.device)

        r = self.R_pat
        self.window = oddenise(r*2)
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
        

    def get_grids(self, time_window, channels, R_rf, time_px, N, device='cuda'):

        # Generate grid positions for each patch using broadcasting
        grid_strides_w = torch.logspace(0, np.log10((time_window-1) / time_px), N, device=device).view(-1, 1)
        #grid_strides_w = torch.linspace(1, (time_window-1) / time_px), N, device=device).view(-1, 1)
        print('strides: ', grid_strides_w.min(), grid_strides_w.max())
        grid_positions_h = torch.linspace(0, channels - R_rf, N, device=device).view(1, -1) / (channels - 1) * 2 - 1
        
        # Compute normalized coordinates for each patch
        x = -1 + torch.linspace(0, time_px, time_px, device=device).view(1, -1) / (time_window - 1) * 2 * grid_strides_w
        y = grid_positions_h + torch.linspace(0, R_rf-1, R_rf, device=device).view(-1, 1) / (channels - 1) * 2
        
        # Stack and reshape to create grid
        grids_x, grids_y = torch.meshgrid(x.flatten(), y.flatten())
        grids = torch.stack((grids_x, grids_y), dim=-1)
        grids = grids.view(N, time_px, R_rf, N,  2).permute(0,3,2,1,4).reshape(N*N, R_rf, time_px, 2)
    
        return grids, grid_strides_w

        
    def extract_patches(
        self,
        patch: torch.Tensor,         # (B, C, f, t)  already on CUDA
        align_corners: bool = True,  # whichever you prefer
    ) -> torch.Tensor:
        """
        Samples *all* S×S units from `patch` in one fused grid_sample call.
    
        Returns
        -------
        patches : (B, S, S, C, R_rf, time_px) tensor
        """

        grid = self.grids
        
        B, C, f, t   = patch.shape
        N_units, H, W, _ = grid.shape           # N_units = S*S
    
        # ------ replicate inputs & grids along a *flattened* batch dim ----
        patch_big = (
            patch                               # (B, C, f, t)
            .unsqueeze(1)                       # (B, 1, C, f, t)
            .expand(B, N_units, C, f, t)        # (B, N_units, C, f, t)
            .reshape(B * N_units, C, f, t)      # (B*N_units, C, f, t)
        )
        grid_big = (
            grid                                # (N_units, H, W, 2)
            .unsqueeze(0)                       # (1, N_units, H, W, 2)
            .expand(B, N_units, H, W, 2)        # (B, N_units, H, W, 2)
            .reshape(B * N_units, H, W, 2)      # (B*N_units, H, W, 2)
        )
    
        # ------ bilinear sampling (fully vectorised & GPU-friendly) -------
        out = F.grid_sample(
            patch_big, grid_big,
            mode="bilinear", align_corners=align_corners
        )                                       # (B*N_units, C, H, W)
    
        # ------ reshape back to (B, S, S, C, R_rf, time_px) --------------
        S = int(N_units ** 0.5)
        out = (
            out
            .view(B, N_units, C, H, W)          # (B, S*S, C, …)
            .view(B, S, S, C, H, W)             # (B, S, S, C, …)
        )
        return out
    
