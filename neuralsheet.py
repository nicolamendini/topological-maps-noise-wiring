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
        input_size, 
        R_rf,
        R_pat=2,
        R_long=0, 
        cutoff=4,
        device='cuda'
    ):
        super().__init__()

        self.noise_spatial_corr = 0
        #self.range_norm = 0.1
        self.homeo_lr = 1e-3
        self.homeo_target = 0.04
        self.aff_unlearning = 0
        self.lat_unlearning = 0
        self.iterations = 30
        self.strength = 2
        self.aff_strength = 0.5
        self.cutoff_speeding = True
        self.range_norm = 0.16
        self.gain = 1
        
        self.sheet_size = sheet_size  # Size of the sheet
        self.input_size = input_size  # Size of the input crop
        self.device = device
        self.phi_short = 1
        self.phi_long = 1
        self.phi_mid = 1

        # Afferent (receptive field) weights for each neuron in the sheet
        self.rf_size = oddenise(R_rf*2)

        std_exc = R_pat / 5
        self.std_exc = std_exc
        self.R_long = R_long
        self.R_pat = R_pat

        self.init_indices()

        self.retinotopic_bias = get_circle(self.rf_size, self.rf_size/2).float().to(device)
        self.retinotopic_bias /= self.retinotopic_bias.max()
        
        afferent_weights = torch.rand((sheet_size**2, 1, self.rf_size, self.rf_size), device=device)
        afferent_weights *= self.retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_exc).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc

        self.mri_mask = 1 - self.lateral_weights_exc / self.lateral_weights_exc.view(sheet_size**2, -1).max(1)[0].view(-1,1,1,1)

        self.eye = torch.eye(sheet_size**2).view(sheet_size**2, 1, sheet_size, sheet_size).to(device)

        #self.mid_cutoff = generate_gaussians(sheet_size, sheet_size, R_pat/2).to(device) 
        self.mid_cutoff = generate_circles(sheet_size, sheet_size, R_pat).to(device) 
        self.long_cutoff = generate_circles(sheet_size, sheet_size, R_long, offset=self.window//2).to(device)
        self.short_cutoff = generate_circles(sheet_size, sheet_size, std_exc*cutoff/2).to(device)
        #self.long_cutoff *= 1 - generate_circles(sheet_size, sheet_size, R_pat, offset=self.window//2).to(device) 
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations
                
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.prev_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)

        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.var_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.gains = torch.ones(1, 1, sheet_size, sheet_size, device=device)

        self.avg_hist = torch.zeros(10) 
        self.noise = 0

        self.long_range_inh = torch.tensor([0]).to(device)
        self.long_range_exc = torch.tensor([0]).to(device)
        self.mid_range_inh = torch.tensor([0]).to(device)
        self.short_range_exc = torch.tensor([0]).to(device)

        std_lre = R_long / 5
        self.lre_gauss = get_gaussian(sheet_size+self.window-1, std_lre).view(1,-1).to(device)
        self.lre_gauss = self.lre_gauss.sort(dim=1, descending=True)[0].expand(sheet_size**2, -1)

        self.rf_grids = get_grids(input_size, input_size, self.rf_size, sheet_size, device=device)

        # /4 for topo map
        self.env_std = R_long / 1
        self.envelope = generate_gaussians(sheet_size, sheet_size, self.env_std, offset=self.window//2).to(device)
        self.rolled_envelope = self.envelope.clone().detach()

        self.mean_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.std_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.spread = torch.tensor(0.)
        self.b = torch.zeros(sheet_size**2, 1,1,1, device=device)

        self.theta = get_angles(self.long_cutoff, offset=self.window//2)
        self.angular_biases = torch.ones(self.long_cutoff.shape, device=device)

        self.radial_b = 0

        euclid_distance = generate_euclidean_space(sheet_size, sheet_size, offset=self.window//2).to(device) * self.long_cutoff
        cos = torch.cos(self.theta) * euclid_distance
        sin = torch.sin(self.theta) * euclid_distance
        self.euclid = torch.cat(list(get_meshgrid(self.long_cutoff, self.window//2)), dim=1) * self.long_cutoff

        self.rolls = torch.tensor(0.)

        self.isdense = False
        self.needs_update = False

    def forward(
        self, 
        input_crop,
        noise_lvl=0, 
        noise_spatial_corr=0, 
        noise_temporal_corr=0,
        adaptation=True, 
        performance_mode=False,
        phi_short=1,
        phi_long=1,
        phi_mid=1
    ):

        #self.iterations = 50
        #self.response_tracker = torch.zeros(self.iterations, 1, self.sheet_size, self.sheet_size, device=self.device)

        if noise_spatial_corr != self.noise_spatial_corr:
            self.noise_spatial_corr = noise_spatial_corr
            self.update_noise_kernel()

        if phi_long != self.phi_long:
            self.update_long_sparsity(phi_long)

        if phi_mid != self.phi_mid:
            self.update_mid_sparsity(phi_mid)
        
        if phi_short != self.phi_short:
            self.update_short_sparsity(phi_short)
            
        if phi_short>1 or phi_mid>1 or phi_long>1:
            if self.isdense or self.needs_update:
                self.update_interactions(phi_short, phi_mid, phi_long)
                self.isdense=False
                self.needs_update=False
        else:
            if not self.isdense:
                self.update_interactions(phi_short, phi_mid, phi_long)
                self.isdense=True

        if self.noise_spatial_corr != noise_spatial_corr:
            self.noise_spatial_corr = noise_spatial_corr
            self.update_spatial_corr_kernel()
        
        self.current_response *= 0
        
        net_afferent = 0
        break_flag = False
        
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        current_input = input_crop
        self.current_input = input_crop
        self.current_tiles = extract_patches(current_input, self.rf_grids)
        afferent = self.current_tiles * self.get_aff_weights()
        afferent = afferent.sum([2,3])
        self.current_afferent = afferent.sum(1).view(self.current_response.shape)          

        #self.update_interactions(phi_short, phi_mid, phi_long)

        interactions = (self.short_range_exc - self.mid_range_inh) * (1-self.b) + (self.long_range_exc - self.long_range_inh) * self.b
        
        if self.cutoff_speeding:
            # Unfold the tensor to extract sliding windows
            interactions = F.pad(interactions, (self.window//2, self.window//2, self.window//2, self.window//2))
            crops = interactions[self.batch_indices, :, self.final_row_indices, self.final_col_indices]

        for i in range(self.iterations):    
            
            if noise_lvl:

                curr_noise = torch.randn(self.current_response.shape, device=self.current_response.device)
                if i==0:
                    self.noise = curr_noise
                else:
                    self.noise = self.noise * noise_temporal_corr + curr_noise * (1-noise_temporal_corr)   

                if self.noise_spatial_corr:
                    self.noise = F.conv2d(self.noise, self.spatial_corr_kernel, padding=self.spatial_corr_kernel.shape[-1]//2)

                self.noise /= self.noise.abs().max() + 1e-11 
        
            if not self.cutoff_speeding:
                # interactions come with some border that needs to be removed if this line enters
                lateral = F.conv2d(self.current_response, interactions, padding='valid')
                
            else:
                padded_response = F.pad(self.current_response, (self.window//2, self.window//2, self.window//2, self.window//2))
                res_tiles = F.unfold(padded_response, self.window)[0].T
                lateral = (res_tiles * crops.view(res_tiles.shape)).sum(1)
                                                
            lateral = lateral.view(self.current_response.shape) * (1 - self.aff_strength)
            net_afferent = (self.current_afferent - self.thresholds) * self.aff_strength
            
            update = net_afferent + lateral 
                        
            self.current_response = torch.relu(update*self.gains.mean() + self.noise*noise_lvl) 
            self.current_response = torch.tanh(self.current_response) 

            if i % 10 == 0:
                max_change = (self.current_response - self.prev_response).abs().max()
                if max_change < 3e-3:
                    break_flag = True
            
            self.prev_response = self.current_response + 0

            if not performance_mode:
                self.response_tracker[i] = self.current_response + 0

            if break_flag:
                break_flag = False
                break
                                      
        if adaptation:
            fast_lr = self.homeo_lr * 10
            res_pos = self.current_response[self.current_response>0]
            res_pow = res_pos**2
            if self.current_response.max():
                
                target = self.range_norm
                self.mean_fr[self.current_response>0] = self.mean_fr[self.current_response>0]*(1-fast_lr) + res_pow*fast_lr
                gap = (self.mean_fr[self.current_response>0] - target) / target
                self.gains[self.current_response>0] -= gap * fast_lr * 10
                
                #gap = res_max - 0.65
                #self.gains -= gap * fast_lr
                
                self.gains = self.gains.clip(0, 5)

                curr_stds = (res_pos - self.mean_fr[self.current_response>0])**2
                self.std_fr[self.current_response>0] = self.std_fr[self.current_response>0]*(1-fast_lr) + curr_stds*fast_lr
            
            aff_pos = net_afferent[net_afferent>0]
            aff_max = net_afferent.max()
            if aff_pos.shape[0]:
                # can be seen as controlling the average rate of change, the higher the afferent, the slower the change
                #diff = cosim(torch.relu(net_afferent), self.current_response)
                target = 0.017
                #gap = (diff - target) / target
                gap = (aff_pos.mean() - target) / target
                self.aff_strength -= gap * self.homeo_lr * 1e-1
                #print(aff_pos.max(), aff_pos.mean())
                #self.ap = aff_pos.clone()
                
            self.aff_strength = self.aff_strength.clip(0, 1)
            
            if not performance_mode:
                new_hist = np.histogram(res_pos.cpu(), bins=10, range=(0,1))[0]
                self.avg_hist = self.avg_hist*(1-self.homeo_lr) + new_hist*self.homeo_lr

            self.mean_activations = self.mean_activations*(1-self.homeo_lr) + self.current_response*self.homeo_lr
            thresh_update = self.homeo_target - self.mean_activations
            self.thresholds -= (thresh_update/self.homeo_target) * self.homeo_lr * 1e-1

            if self.R_long:
                self.std_target = 0.215
                target_cf = self.long_range_inh
                masses = (np.pi * self.R_long**2).round()
                self.spread = get_masses_and_spreads(target_cf, norm_flag=True, masses=masses)[1].view(-1,1,1,1)
                gap = (self.std_target - self.spread.mean()) / self.std_target
                self.b += gap * self.homeo_lr 
                self.b = self.b.clip(min=0, max=0.9)

                #if gap.abs().mean() < 0.1:

                #target_cf = self.long_range_exc
                #exc_spread = get_masses_and_spreads(target_cf, norm_flag=True)[1].view(-1,1,1,1)
                #gap = (exc_spread - 0.25) / 0.25
                #self.p_exc = self.p_exc - gap * self.homeo_lr * 2
                #self.p_exc = self.p_exc.clip(1/5, 5)

            else:
                target_cf = self.lateral_correlations * self.mid_cutoff
                self.spread = get_masses_and_spreads(target_cf, norm_flag=True)[1].view(-1,1,1,1)
            
                        
    def hebbian_step(self):

        afferent_contributions = self.current_tiles - self.aff_unlearning
        self.step(self.afferent_weights, afferent_contributions, self.current_response)
        
        contributions = self.current_response - self.lat_unlearning 
        self.step(self.lateral_correlations, contributions, self.current_response)

    def step(self, weights, target, response):

        delta = response.view(-1,1,1,1) * target 
        weights += self.hebbian_lr * delta # add new changes
        weights *= weights > 0 # clear weak weights
        weights /= weights.sum([2,3], keepdim=True) + 1e-11 # normalise remaining weights
        
    def get_aff_weights(self):
        
        aff_weights = self.afferent_weights * self.retinotopic_bias
        aff_weights /= aff_weights.sum([2,3], keepdim=True) + 1e-11
        
        return aff_weights

    def update_interactions(self, phi_short, phi_mid, phi_long):

        sre = self.lateral_weights_exc

        mri = self.lateral_correlations * self.mid_cutoff * self.mri_mask
        mri /= mri.sum([2,3], keepdim=True) + 1e-11

        if self.R_long:

            lateral_padded = F.pad(self.lateral_correlations, (self.window//2, self.window//2, self.window//2, self.window//2), mode='reflect')
            sampling_cf = lateral_padded * self.long_cutoff 
            lri = sampling_cf[:,:,self.window//2:-self.window//2+1,self.window//2:-self.window//2+1]
            lri = lri / (lri.sum([2,3], keepdim=True) + 1e-11)

            sampling_cf = sampling_cf * self.rolled_envelope

            #lre_masks = get_sparsity_masks(sampling_cf,self.long_cutoff, 1/3)
            #lre = sampling_cf * lre_masks

            lre_sorted = sampling_cf.view(sampling_cf.shape[0], -1).sort(dim=1, descending=True)[1]
            lre = self.lre_gauss * 0
            lre[torch.arange(sampling_cf.shape[0], device=self.device)[:, None], lre_sorted] = self.lre_gauss          

            lre = lre.view(sampling_cf.shape) 

            asym = (lre * self.euclid).sum([2,3])
            
            self.rolls = self.rolls + asym * self.homeo_lr * 32 * 1e2
            self.rolls = self.rolls.clip(-self.env_std, self.env_std)
            #print(self.rolls)
            
            self.rolled_envelope = batch_roll_2d(self.envelope, -self.rolls.round().int())
            
            asym = asym.abs().mean()            
            self.radial_b = self.radial_b * (1-1e-1) + asym * 1e-1

            
            lre = lre[:,:,self.window//2:-self.window//2+1,self.window//2:-self.window//2+1]
            lre /= lre.sum([2,3], keepdim=True) + 1e-11

            if phi_long>1:

                lre = lre * self.long_sparsity_masks
                #lre -= (lre+self.long_sparsity_masks+(1-self.long_cutoff)).view(lre.shape[0], -1).min(1)[0].view(-1,1,1,1)
                lre = lre ** self.lre_norm
                lre = lre / (lre.sum([2,3], keepdim=True) + 1e-11)
    
                lri = lri * self.long_sparsity_masks
                #lri -= (lri+self.long_sparsity_masks+(1-self.long_cutoff)).view(lri.shape[0], -1).min(1)[0].view(-1,1,1,1)
                lri = lri ** self.lri_norm
                lri = lri / (lri.sum([2,3], keepdim=True) + 1e-11)

            self.long_range_exc = lre
            self.long_range_inh = lri

        if phi_mid>1:
            mri = mri * self.mid_sparsity_masks
            mri = mri ** self.mri_norm
            mri = mri / (mri.sum([2,3], keepdim=True) + 1e-11)

        if phi_short>1:
            sre = sre * self.short_sparsity_masks
            sre = sre ** self.sre_norm
            sre = sre / (sre.sum([2,3], keepdim=True) + 1e-11)

        self.mid_range_inh = mri
        self.short_range_exc = sre

    def init_indices(self):

        N = self.sheet_size
        
        self.crop_indeces = torch.arange(N**2).to(self.device)

        r = max(self.R_long, self.R_pat)
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

    def update_noise_kernel(self):
        
        sck_size = oddenise(self.noise_spatial_corr*self.cutoff)
        self.spatial_correlation_kernel = get_gaussian(sck_size, self.noise_spatial_corr).to(self.device)

    def update_long_sparsity(self, phi_long):

        if phi_long > 1:
            self.phi_long = phi_long
            self.long_sparsity_masks = get_sparsity_masks(
                                    self.lateral_correlations,
                                    self.long_cutoff[:,:,self.window//2:-self.window//2+1,self.window//2:-self.window//2+1], 
                                    1/self.phi_long
                                )
            self.lre_norm = np.exp(-self.phi_long + 1) * 0.5 + 0.5
            self.lri_norm = np.exp(-self.phi_long + 1) * 0.8 + 0.2

            self.needs_update=True

    def update_mid_sparsity(self, phi_mid):

        if phi_mid > 1:
            self.phi_mid = phi_mid
            self.mid_sparsity_masks = get_sparsity_masks(
                                    self.lateral_correlations,
                                    self.mid_cutoff * self.mri_mask, 
                                    1/self.phi_mid
                                )
            mri = self.lateral_correlations * self.mid_cutoff * self.mri_mask
            self.mri_norm = np.exp(-self.phi_mid + 1) * 0.8 + 0.2

            self.needs_update=True


    def update_short_sparsity(self, phi_short):

        if phi_short > 1:
            self.phi_short = phi_short
            self.short_sparsity_masks = get_sparsity_masks(
                                    self.lateral_weights_exc,
                                    self.short_cutoff, 
                                    1/self.phi_short
                                )
            short_interactions = self.lateral_weights_exc * self.short_cutoff
            self.sre_norm = np.exp(-self.phi_mid + 1)

            self.needs_update=True

    def update_spatial_corr_kernel(self):

        size = oddenise(self.noise_spatial_corr*self.cutoff)
        self.spatial_corr_kernel = get_gaussian(size, self.noise_spatial_corr).to(self.device)

        

        
