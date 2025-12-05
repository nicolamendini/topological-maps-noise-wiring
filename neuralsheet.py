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
from scipy.stats import rv_histogram

from wiring_efficiency_utils import *

class NeuralSheet(nn.Module):
    def __init__(
        self, 
        sheet_size, 
        input_size, 
        R_rf=5,
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
        self.cutoff_speeding = True
        self.range_norm = 0.16
        self.gain = 0.1
        
        self.sheet_size = sheet_size  # Size of the sheet
        self.input_size = input_size  # Size of the input crop
        self.device = device
        self.phi_short = 1
        self.phi_long = 1
        self.phi_mid = 1

        # Afferent (receptive field) weights for each neuron in the sheet
        self.rf_size = oddenise(R_rf*2)
        self.aff_pad = self.rf_size

        std_exc = R_long / 3 #* np.sqrt(2) #if not R_long else 0.1
        self.std_exc = std_exc
        self.R_long = R_long
        self.R_pat = R_pat

        self.R_ret = 8
        R_ret_exc = self.R_ret #/ np.sqrt(2)

        self.init_indices()
        
        self.aff_cutoff = get_circle(self.rf_size, self.rf_size/2).float().to(device)
        
        #self.retinotopic_bias = get_gaussian(self.rf_size + self.aff_pad*2, R_rf).float().to(device).repeat(sheet_size**2,1,1,1)
        #self.retinotopic_bias /= self.retinotopic_bias.max()
        self.aff_cartesian = get_cartesian(self.rf_size).to(device).permute(0,1,3,2)
        
        afferent_weights = torch.rand((sheet_size**2, 1, self.rf_size, self.rf_size), device=device)
        #afferent_weights *= self.retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights

        self.aff_mask = torch.rand(afferent_weights.shape, device=self.device) < 1

        self.retinotopic_bias = get_gaussian(self.rf_size, R_rf).float().to(device)
        self.retinotopic_bias = 1

        self.aff_euclid = get_euclid(self.rf_size).to(device)

        lateral_weights_exc = generate_circles(sheet_size, sheet_size, std_exc).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc

        self.eye = torch.eye(sheet_size**2).view(sheet_size**2, 1, sheet_size, sheet_size).to(device)

        self.rand = 1 + torch.randn(self.eye.shape, device=self.device) * 0.2
        self.rand = torch.relu(self.rand)
        #self.rand *= self.ret_cutoff_exc
        #self.rand /= self.rand.sum([2,3], keepdim=True)

        self.ret_cutoff = generate_circles(sheet_size, sheet_size, self.R_ret).to(device) 
        self.ret_cutoff_exc = generate_circles(sheet_size, sheet_size, R_ret_exc).to(device)

        self.sp_rand = lattice_connectivity_exact_p(self.sheet_size, R_ret_exc, 0.5).to(device).view(self.eye.shape)
        #self.sp_rand = reciprocal_connectivity(self.sheet_size, 0.05).to(device).view(self.eye.shape)
        #self.sp_rand = reciprocal_gaussian(sheet_size, round(0.5*np.pi*R_long**2), R_long).to(device).view(self.eye.shape)
        #self.sp_rand = torch.rand(self.sheet_size**2, self.sheet_size**2).to(device) < 0.12
        #self.sp_rand = lattice_connectivity_exact_p(self.sheet_size, self.R_ret, 0.5).to(device)
        #probs = generate_gaussians(self.sheet_size, self.sheet_size, self.R_ret / 5, offset=self.R_ret).view(self.sheet_size**2,-1)
        #ns = round(np.pi*(self.R_ret/5)**2) 
        #samples = torch.multinomial(probs, ns, replacement=False)
        #mask = torch.zeros(self.sheet_size**2, (self.sheet_size+self.R_ret*2)**2, dtype=torch.int64)
        #mask.scatter_(1, samples, 1)
        #mask = mask.view(self.sheet_size**2, self.sheet_size+self.R_ret*2, self.sheet_size+self.R_ret*2)
        #mask = mask[:,self.R_ret:-self.R_ret,self.R_ret:-self.R_ret]
        #self.sp_rand = mask.to(device)
        self.sp_rand = self.sp_rand.view(self.eye.shape) * (1-self.eye) + self.eye
        self.sp_rand = self.sp_rand * self.ret_cutoff_exc
        self.sp_rand = self.sp_rand > 0
        self.sp_rand = self.sp_rand.float()

        #self.sp_rand2 = lattice_connectivity_exact_p(self.sheet_size, R_ret_exc, 0.5).to(device).view(self.eye.shape)
        #self.sp_rand2 = self.sp_rand2.view(self.eye.shape) * (1-self.eye) + self.eye

        #ns = round(p * self.R_ret**2 * np.pi)
        #self.sp_rand = reciprocal_gaussian(self.sheet_size, ns, self.R_ret/2).float().to(device).view(self.eye.shape)
        
        #self.mri_mask = 1 - self.lateral_weights_exc / self.lateral_weights_exc.view(sheet_size**2, -1).max(1)[0].view(-1,1,1,1)
        #self.mri_mask = 1

        self.mid_cutoff = generate_circles(sheet_size, sheet_size, R_pat).to(device) 
        #self.mid_cutoff = generate_gaussians(sheet_size, sheet_size, 2).to(device) 
        self.long_cutoff = generate_circles(sheet_size, sheet_size, R_long).to(device) 
        self.decor_cutoff = generate_circles(sheet_size, sheet_size, 5).to(device)
        self.decor_gauss = generate_gaussians(sheet_size, sheet_size, 5/5).to(device)
        #self.decor_cutoff += self.decor_cutoff + torch.rand(self.decor_cutoff.shape, device=self.device) 
        #self.decor_cutoff /= self.decor_cutoff.sum([2,3], keepdim=True)
        #self.short_cutoff = generate_circles(sheet_size, sheet_size, std_exc*cutoff/2).to(device)
        self.euclid_cutoff = generate_circles(sheet_size, sheet_size, R_long, offset=self.window//2).to(device)
        #self.long_cutoff *= 1 - generate_circles(sheet_size, sheet_size, R_pat, offset=self.window//2).to(device) 

        #self.ii_cutoff = generate_circles(sheet_size, sheet_size, R_long/2).to(device)
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations

        lateral_correlations_ei = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations_ei /= lateral_correlations_ei.sum([2,3], keepdim=True)
        self.lateral_correlations_ei = lateral_correlations_ei

        #lateral_correlations_ii = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        #lateral_correlations_ii /= lateral_correlations_ii.sum([2,3], keepdim=True)
        #self.lateral_correlations_ii = lateral_correlations_ii

        lateral_correlations_exc = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations_exc /= lateral_correlations_exc.sum([2,3], keepdim=True)
        self.lateral_correlations_exc = lateral_correlations_exc

        self.current_response_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.prev_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)

        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.var_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.gains = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_tracker = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_strength = torch.ones(1, 1, sheet_size, sheet_size, device=device) * 0.5
        self.lat_tracker = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.lat_strength = torch.ones(1, 1, sheet_size, sheet_size, device=device) * 0.5
        self.mean_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.std_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.mean_activations_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.thresholds_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.var_activations_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.gains_l3 = torch.ones(1, 1, sheet_size, sheet_size, device=device) 
        self.aff_tracker_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_strength_l3 = torch.ones(1, 1, sheet_size, sheet_size, device=device) * 0.5
        self.mean_fr_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.avg_hist = torch.zeros(10) 
        self.noise = 0

        self.long_range_inh = torch.tensor([0]).to(device)
        self.long_range_exc = torch.tensor([0]).to(device)
        self.mid_range_inh = torch.tensor([0]).to(device)
        self.short_range_exc = torch.tensor([0]).to(device)

        std_lre = R_long / 5
        self.lre_gauss = get_gaussian(sheet_size+self.window-1, std_lre).view(1,-1).to(device)
        self.lre_gauss = self.lre_gauss.sort(dim=1, descending=True)[0].expand(sheet_size**2, -1)

        self.jitter = torch.randn(self.sheet_size**2,1,1,1, device=self.device) / (input_size - self.rf_size) * 0

        self.rf_grids = get_grids(input_size, input_size, self.rf_size, sheet_size, jitter=self.jitter, device=device)

        # /4 for topo map
        self.env_std = R_long / 2
        self.env_pad = round(R_long * 2)
        #offset = self.env_pad
        self.envelope = generate_gaussians(sheet_size, sheet_size, self.env_std).to(device)
        self.envelope /= self.envelope.view(self.sheet_size**2,-1).max(1)[0].view(-1,1,1,1)
        self.rolled_envelope = self.envelope.clone().detach()
        
        self.spread = torch.tensor(0.)
        self.b = torch.zeros(sheet_size**2, 1,1,1, device=device)

        self.radial_b = torch.zeros(sheet_size**2,1,1,1).to(device)

        self.euclid_distance = generate_euclidean_space(sheet_size, sheet_size).to(device)
        self.euclid = torch.cat(list(get_meshgrid(self.euclid_cutoff, self.window//2)), dim=1) * self.euclid_cutoff

        self.rolls = torch.tensor(0.)
        self.aff_rolls = torch.zeros(sheet_size**2,2,1,1).to(device)

        self.isdense = False
        self.needs_update = False

        self.exc_spread = torch.tensor([0.])

        pad_amount = self.window // 2
        padded_dim = self.sheet_size + 2 * pad_amount 
        
        self.slicing_var = torch.zeros(
            (self.sheet_size**2, 6, padded_dim, padded_dim), 
            device=device
        )

        self.blend_gauss = generate_circles(self.sheet_size, self.sheet_size, self.R_ret * np.sqrt(2)).to(self.device)
        self.blend_gauss /= self.blend_gauss.view(self.sheet_size**2,-1).max(1)[0].view(-1,1,1,1)

        self.sre_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mri_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + 0.16
        self.lre_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.lri_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.delta_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.mri_p = torch.ones((sheet_size**2, 1, 1, 1), device=device)
        self.sre_p = torch.ones((sheet_size**2, 1, 1, 1), device=device)
        self.lri_p = torch.ones((sheet_size**2, 1, 1, 1), device=device) * 0
        self.lre_p = torch.ones((sheet_size**2, 1, 1, 1), device=device) * 0
        self.aff_p = torch.ones((sheet_size**2, 1, 1, 1), device=device)

        self.short_sparsity_masks = 1
        self.mid_sparsity_masks = 1
        self.long_sparsity_masks = 1

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
        self.response_tracker *= 0
        
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

        net_afferent = (self.current_afferent - self.thresholds) * self.aff_strength

        #self.update_mid_sparsity(phi_mid)
        #self.update_interactions(phi_short, phi_mid, phi_long)

        #plt.imshow(self.mid_range_inh[100,0].cpu())
        #plt.show()

        #print((self.mid_range_inh * self.lateral_correlations).sum([2,3]).mean())
        
        if self.cutoff_speeding:
            pad_amount = self.window // 2
            
            # The weights are [M, 1, N, N]. F.pad preserves the 1 channel dim.
            
            # Use unsqueeze to ensure short_range_exc has a channel dimension before padding
            sre_padded = F.pad(self.short_range_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 0:1] = sre_padded # Assign to the channel slice [:, 0:1] which is [M, 1, P, P]
            
            mri_padded = F.pad(self.mid_range_inh, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 1:2] = mri_padded
            
            if self.R_long:
                
                lre_padded = F.pad(self.long_range_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
                self.slicing_var[:, 2:3] = lre_padded

                #self.r = self.current_afferent.mean() * 0.00
                #lri = torch.relu(self.lateral_correlations - self.r) + 1e-11
                #lri = lri * self.long_cutoff_inh
                #lri = lri / lri.sum([2,3], keepdim=True)
                lri_padded = F.pad(self.long_range_inh, (pad_amount, pad_amount, pad_amount, pad_amount))
                self.slicing_var[:, 3:4] = lri_padded

                #ei = self.lateral_correlations_ei * self.long_cutoff
                #ei /= ei.sum([2,3], keepdim=True) + 1e-11
                #ei_padded = F.pad(ei, (pad_amount, pad_amount, pad_amount, pad_amount))
                #self.slicing_var[:, 4:5] = ei_padded

                #ii = self.lateral_correlations_ii * self.mid_cutoff #* self.mri_mask
                #ii /= ii.sum([2,3], keepdim=True) + 1e-11
                #ii_padded = F.pad(ii, (pad_amount, pad_amount, pad_amount, pad_amount))
                #self.slicing_var[:, 5:6] = ii_padded

        sliced_interactions = self.slicing_var[self.batch_indices, :, self.final_row_indices, self.final_col_indices]

        sre_crops = sliced_interactions[:,:,:,:,0]
        mri_crops = sliced_interactions[:,:,:,:,1]
        lre_crops = sliced_interactions[:,:,:,:,2]
        
        lri_crops = sliced_interactions[:,:,:,:,3]

        ei_crops = sliced_interactions[:,:,:,:,4]
        #ii_crops = sliced_interactions[:,:,:,:,5]

        
        #plt.imshow(mri_crops[0,0].cpu())
        #plt.show()

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

                #padded_response_i = F.pad(self.current_response_i, (self.window//2, self.window//2, self.window//2, self.window//2))
                #res_tiles_i = F.unfold(padded_response_i, self.window)[0].T

                pr = 1
                fails = 1#(torch.rand(res_tiles.shape, device=res_tiles.device)<pr)/pr
                lateral_sre = (res_tiles * sre_crops.view(res_tiles.shape) * fails).sum(1).view(self.current_response.shape)
                lateral_mri = (res_tiles * mri_crops.view(res_tiles.shape) * fails).sum(1).view(self.current_response.shape)

                if self.R_long:

                    #padded_response = F.pad(self.current_response, (self.window//2, self.window//2, self.window//2, self.window//2))
                    #res_tiles = F.unfold(padded_response, self.window)[0].T

                    if True:
                        
                        pr = 1
                        fails = 1#(torch.rand(res_tiles.shape, device=res_tiles.device)<pr)/pr
                        lateral_lre = (res_tiles * lre_crops.view(res_tiles.shape) * fails).sum(1).view(self.current_response.shape)
                        lateral_lri = (res_tiles * lri_crops.view(res_tiles.shape) * fails).sum(1).view(self.current_response.shape)
                        #self.current_afferent_l3 = (res_tiles * ei_crops.view(res_tiles.shape)).sum(1).view(self.current_response.shape)
    
                        #ei = (res_tiles * ei_crops.view(res_tiles.shape)).sum(1).view(self.current_response.shape)
                        #ii = (res_tiles_i * ii_crops.view(res_tiles.shape)).sum(1).view(self.current_response.shape)

                    else:

                        #lre = self.long_range_exc
                        #lateral_lre = (self.long_range_exc * self.current_response).sum([2,3]).view(self.current_response.shape)

                        if i==0:
                            pass
                            #r = random.random() / self.R_long**2 * 0.05

                            
                            #r = 1 / self.R_long**2 * 0.05

                        #lri = torch.relu(self.lateral_correlations - self.r) + 1e-11
                        #lri = lri * self.long_cutoff_inh
                        ##lri = self.long_range_inh
                        #lri = lri / lri.sum([2,3], keepdim=True)

                        #plt.imshow(torch.relu(lre - lri).cpu()[120,0])
                        #plt.show()
                        
                        #lateral_lri = (lri * self.current_response).sum([2,3]).view(self.current_response.shape)

                        if False:
                            local_ret = self.sp_rand * self.ret_cutoff_exc * self.lateral_correlations_exc
                            local_ret /= local_ret.sum([2,3], keepdim=True)
                    
                            local_inh = self.ret_cutoff * self.lateral_correlations
                            local_inh /= local_inh.sum([2,3], keepdim=True)
    
                            lateral_sre = (local_ret * self.current_response).sum([2,3]).view(self.current_response.shape)
                            lateral_mri = (local_inh * self.current_response).sum([2,3]).view(self.current_response.shape)

                    
                else:
                    
                    lateral_lre = torch.tensor([0], device=self.device)
                    lateral_lri = torch.tensor([0], device=self.device)


            ########## inh
            
            #update_i = self.current_response*0.5 + ei - ii * 0.
            #self.current_response_i = torch.relu(update_i)

            #padded_response_i = F.pad(self.current_response_i, (self.window//2, self.window//2, self.window//2, self.window//2))
            #res_tiles_i = F.unfold(padded_response_i, self.window)[0].T

            #lateral_lri = (res_tiles_i * lri_crops.view(res_tiles.shape)).sum(1).view(self.current_response.shape)
            
            ##########

            #b = self.b.view(self.current_response.shape)
            #b = 0.65

            #inv_t = 10
            #short_f = max(1, 2.5*(1 - i / inv_t))
            short_pot = (lateral_sre - lateral_mri) * 1  #self.b.mean() #* short_f

            #long_f = min(1, i / inv_t)
            long_pot = (lateral_lre - lateral_lri) * 1 #* self.gains.mean()

            #print(short_f, long_f)
            
            update = net_afferent + (short_pot + long_pot) * self.gains
                        
            self.current_response = torch.tanh(torch.relu(update + self.noise*noise_lvl)) 

            #net_afferent_l3 = (self.current_afferent_l3 - self.thresholds_l3)  * self.aff_strength_l3

            #long_pot = (lateral_lre - lateral_lri)

            #update = net_afferent

            #self.current_response_l3 = torch.tanh(torch.relu(update*self.gains_l3))


            #if i % 10 == 0:
            #    max_change = (self.current_response - self.prev_response).abs().max()
            #    if max_change < 3e-3:
            #        break_flag = True
            
            self.prev_response = self.current_response + 0

            if not performance_mode:
                self.response_tracker[i] = self.current_response + 0

            #if break_flag:
            #    break_flag = False
            #    break


        beta = 1e-3

        res_pos = self.current_response > 0

        beta = self.homeo_lr * 1e2 * self.current_response / 2
        #self.aff_mag = (((self.afferent_weights - self.thresholds.view(-1,1,1,1)) * self.aff_strength.view(-1,1,1,1))**2).sum([2,3], keepdim=True) 
        #self.aff_mag = (self.afferent_weights**2).sum([2,3]) * self.rf_size**2 * np.pi / 4
        self.lri_mag = (1-beta) * self.lri_mag + beta * long_pot
        self.delta_mag = (1-beta) * self.delta_mag + beta * lateral_lre #* self.b.view(self.delta_mag.shape)

        
        if adaptation:

            beta_fr = self.homeo_lr * 1e2 * self.current_response / 2
            self.range_norm = 0.3
            target = self.range_norm
            self.mean_fr = self.mean_fr * (1 - beta_fr) + beta_fr * self.current_response
            gap = (self.mean_fr - target) / target
            self.gains -= gap * self.homeo_lr
            self.gains = self.gains.clip(0)
            
            #gap = res_max - 0.65
            #self.gains -= gap * fast_lr
            
            #self.gains = self.gains.clip(0, 5)

            self.std_fr = self.std_fr * (1 - beta_fr) + beta_fr * self.current_response**2
            

            self.aff_tracker = self.aff_tracker * (1 - beta) + net_afferent * beta 
            target = 0.15
            gap = (self.aff_tracker - target) / target
            #gap = gap.view(-1,1,1,1)
            self.aff_strength = self.aff_strength - gap * self.homeo_lr
            self.aff_strength = self.aff_strength.clip(0)

            self.lat_tracker = self.lat_tracker * (1 - beta) + short_pot * beta 
            target = 0.14
            gap = (self.lat_tracker - target) / target
            #gap = gap.view(-1,1,1,1)
            self.lat_strength = self.lat_strength - gap * self.homeo_lr
            self.lat_strength = self.lat_strength.clip(0)
            
            if not performance_mode:
                new_hist = np.histogram(self.current_response[self.current_response>0].cpu(), bins=10, range=(0,1))[0]
                self.avg_hist = self.avg_hist*(1-self.homeo_lr) + new_hist*self.homeo_lr

            self.mean_activations = self.mean_activations*(1-self.homeo_lr) + self.current_response*self.homeo_lr
            thresh_update = (self.homeo_target - self.mean_activations) / self.homeo_target
            self.thresholds -= thresh_update * self.homeo_lr 
            self.thresholds = self.thresholds.clip(-1,1)

            #if self.R_long and False:

            #    beta_fr = self.homeo_lr * 1e2 * self.current_response_l3 / 2
            #    target = self.range_norm
            #    self.mean_fr_l3 = self.mean_fr_l3 * (1 - beta_fr) + beta_fr * self.current_response_l3
            #    gap = (self.mean_fr_l3 - target) / target
            #    self.gains_l3 -= gap * self.homeo_lr / 2 
            #    
            #    self.aff_tracker_l3 = self.aff_tracker_l3 * (1 - beta) + net_afferent_l3 * beta 
            #    target = 0.08
            #    gap = (self.aff_tracker_l3 - target) / target
            #    #gap = gap.view(-1,1,1,1)
            #    self.aff_strength_l3 = self.aff_strength_l3 - gap * 1e-4
            #    self.aff_strength_l3 = self.aff_strength_l3.clip(0)
    #
            #    self.mean_activations_l3 = self.mean_activations_l3*(1-self.homeo_lr) + self.current_response_l3*self.homeo_lr
            #    thresh_update = (self.homeo_target - self.mean_activations_l3) / self.homeo_target
            #    self.thresholds_l3 -= thresh_update * self.homeo_lr 
            #    self.thresholds_l3 = self.thresholds_l3.clip(-1,1)

            #print(thresh_update.mean(), self.homeo_target)

            target_cf = self.mid_range_inh
            max_mass = np.pi * self.R_pat**2
            masses = target_cf / target_cf.view(self.sheet_size**2, 1, 1, -1).max(3, keepdim=True)[0]
            masses = masses.sum([2,3], keepdim=True) + 1e-11
            #masses = masses / masses.max() 
            ms = get_masses_and_spreads(target_cf, norm_flag=True)
            self.spread = ms[1].view(-1,1,1,1)


            #target = 0.125
            target = self.R_pat / 1.73
            gap = (target - self.mri_mag) / target
            self.mri_p -= gap.view(self.mri_p.shape) * 1e-3 *0#* (self.homeo_lr < 5e-4)
            self.mri_p = self.mri_p.clip(0, 3)

            target = self.R_pat / 4
            gap = (target - self.sre_mag) / target
            self.sre_p -= gap.view(self.sre_p.shape) * 1e-3 *0
            self.sre_p = self.sre_p.clip(0, 3)

            self.aff_dist = (self.aff_euclid * self.get_aff_weights()).sum([2,3]).view(-1,1,1,1)
            target = 5.8
            gap = (target - self.aff_dist) / target
            #self.aff_p -= gap.view(self.aff_p.shape) * 1e-3
            #self.aff_p = self.aff_p.clip(0.1, 3)

            if self.R_long:
                self.std_target = 0.24
                target_cf = self.long_range_inh
                max_mass = np.pi * self.R_long**2
                masses = target_cf / target_cf.view(self.sheet_size**2, 1, 1, -1).max(3, keepdim=True)[0]
                masses = masses.sum([2,3], keepdim=True) + 1e-1
                masses = masses / masses.max() 
                self.spread = get_masses_and_spreads(target_cf, norm_flag=True, masses=max_mass)[1].view(-1,1,1,1)
                gap = (self.std_target - self.spread.mean()) / self.std_target

                target_cf = self.long_range_exc
                max_mass = self.R_long**2
                self.exc_spread = get_masses_and_spreads(target_cf, norm_flag=True, masses=max_mass)[1].view(-1,1,1,1)
                #gap = (self.std_target - self.exc_spread.mean()) / self.std_target    

                if True:
                    # x has shape (M, 1, N, N), with M = N*N
                    #M, _, N, _ = self.lateral_correlations.shape
                    
                    # compute row and col indices for each m in 0...(M-1)
                    #rows = torch.arange(M) // N
                    #cols = torch.arange(M) % N
                    
                    # reshape to (M,1,1,1)
                    #out = self.long_range_exc[torch.arange(M), 0, rows, cols].view(M, 1, 1, 1) 
                    
                    #max_masses = self.long_cutoff[:,:,self.window//2:-self.window//2+1,self.window//2:-self.window//2+1].sum([2,3], keepdim=True)
                    
                    #self.self_corr = out * masses
                        
                    #self.corr_target = 10
    
                    #gap = (self.self_corr - self.corr_target) / self.corr_target

                    #self.exc_masses = get_masses_and_spreads(self.long_range_exc, norm_flag=True)[0].view(self.b.shape) / max_masses

                    #target = 1/3
                    #gap = (self.exc_masses - target) / target
                    #gap = - (1 - self.b)
                    #target = 0.13
                    #gap = (self.std_fr - target).view(self.b.shape) / target

                    #target = 0.88 # 0.465
                    #gap = - (self.aff_mag - target) / target
                    #gap = gap.view(self.b.shape)

                    #gap = self.b - 12
                    #target = 0.65
                    #gap = (self.delta_mag.view(self.b.shape) - target) / target

                    #target = 0.72 * self.long_cutoff.sum([2,3]).view(self.lri_mag.shape) * 3.5 / np.pi
                    #target = 0.13
                    target = 0.1
                    gap = - (target - self.lri_mag.view(self.b.shape)) / target
                    
                    self.b -= gap * 2e-4
                    self.b = self.b.clip(0)

        
                #target = self.R_long / 1.9
                #target = 0.1
                #gap = -(target - self.lri_mag) / target
                #target = 0.8 * self.long_cutoff.sum([2,3]).view(self.lre_mag.shape) / np.pi
                target = 0.1
                gap = - (target - self.lri_mag) / target
                self.lre_p -= gap.view(self.lre_p.shape) * 2e-5
                self.lre_p = self.lre_p.clip(0, 0.2)

                #self.lre_p *= 0
                #self.lre_p += 4


                #target = 0.11
                #gap = (target - self.lri_mag) / target

                target = 0.12
                gap = (target - self.lri_mag) / target
                self.lri_p -= gap.view(self.lri_p.shape) * 1e-5 #* (self.homeo_lr < 3e-4)
                self.lri_p = self.lri_p.clip(0.00, 0.01)

                #self.lre_p *= 0
                #self.lre_p += 3


            
                        
    def hebbian_step(self):

        s = 0
        w = torch.exp(- s * torch.linspace(0, 1, self.iterations, device=self.device)).view(-1,1,1,1)
        w /= w.sum()
        self.trace = self.response_tracker * w
        self.trace = self.trace.sum(0, keepdim=True)

        #contributions = (self.current_response - self.lat_unlearning)
        #contributions = mean_res.view(self.current_response.shape)
        #contributions = torch.relu(self.current_afferent - self.thresholds) 

        afferent_contributions = self.current_tiles - self.aff_unlearning
        self.step(self.afferent_weights, afferent_contributions, self.current_response.view(-1,1,1,1))

        #inv_mask = 1 - self.blending_gauss
        #add_rand = self.blending_gauss * self.rand
        #print(inv_mask.shape, self.trace.shape)
        contributions = self.trace
        self.step(self.lateral_correlations_exc, contributions, contributions.view(-1,1,1,1) - self.homeo_target)

        #a = contributions + add_rand
        #plt.imshow(a[130,0].cpu())
        #plt.show()
        #print(a.max(), a.min())

        #contributions = (self.current_response - self.lat_unlearning)
        #self.step(self.lateral_correlations_exc, contributions, -contributions.view(-1,1,1,1)*0.25 - 0.0)

        contributions = self.current_response - self.lat_unlearning 
        self.step(self.lateral_correlations, contributions, contributions.view(-1,1,1,1))

        contributions = self.current_response
        self.step(self.lateral_correlations_ei, contributions, self.homeo_target - contributions.view(-1,1,1,1))

        #contributions = (self.current_response_i - self.lat_unlearning)
        #self.step(self.lateral_correlations_ii, contributions, self.current_response_i.view(-1,1,1,1)*10)

    def step(self, weights, target, response, unlearning=0):

        #weights **= power_amp
        delta = response * target - unlearning 
        weights += self.hebbian_lr * delta #* (10 if unlearning>0 else 1) # add new changes

        #if fixed_alpha:
        #    weights += self.blending_gauss * fixed_alpha * self.homeo_target
            
        weights *= weights > 0 # clear weak weights
        weights += 1e-11
        weights /= weights.sum([2,3], keepdim=True) # normalise remaining weights
        
    def get_aff_weights(self):

        #retinotopy = self.aff_crop()
        aff_weights = self.afferent_weights * self.aff_cutoff * self.aff_mask * self.retinotopic_bias
        #aff_weights = self.afferent_weights * self.aff_cutoff
        aff_weights /= aff_weights.sum([2,3], keepdim=True) + 1e-11
        
        return aff_weights

    def update_interactions(self, phi_short, phi_mid, phi_long):

        #if self.R_long:
            
        #sre = self.eye + self.rand * 2
        #sre *= self.decor_cutoff
        #sre /= sre.sum([2,3], keepdim=True) + 1e-11
            
        #else:

        #deco_strength = 0.4

        sre = self.lateral_weights_exc
        sre = sre / sre.sum([2,3], keepdim=True)
        
        mri = self.lateral_correlations * self.ret_cutoff
        mri /= mri.sum([2,3], keepdim=True) + 1e-11

        local_ret = self.decor_gauss
        local_ret /= local_ret.sum([2,3], keepdim=True)

        local_inh = self.decor_cutoff * self.lateral_correlations
        local_inh /= local_inh.sum([2,3], keepdim=True)

        deco_strength = 0
        ret = 1

        sre = sre * ret + local_ret * deco_strength #self.b.mean()
        mri = mri * ret + local_inh * deco_strength #* (0.25 + self.b.mean())

        #sre /= 1 + ret_strength
        #mri /= 1 + ret_strength

        #sre = self.lateral_weights_exc if self.R_pat else self.local_ret
        #sre /= sre.sum([2,3], keepdim=True) + 1e-11

        #mri = self.lateral_correlations * self.decor_cutoff
        #mri /= mri.sum([2,3], keepdim=True) + 1e-11

        #sre = decor_exc * deco_strength + sre * (1-deco_strength)
        #mri = decor_inh * deco_strength + mri * (1-deco_strength)

        #if self.R_long:

        #    b_sp = 0.67

        #    sre = (1-b_sp) * sre + b_sp * self.long_cutoff / self.long_cutoff.sum([2,3], keepdim=True)
        #    sre /= sre.sum([2,3], keepdim=True) + 1e-11

        #    inh = self.lateral_correlations_exc * self.mri_mask * self.long_cutoff 
        #    inh /= inh.sum([2,3], keepdim=True) + 1e-11
            
        #    mri = (1-b_sp) * mri + b_sp * inh
        #    mri /= mri.sum([2,3], keepdim=True) + 1e-11

        if self.R_long:

            lri = (self.lateral_correlations + 1e-11) * self.long_cutoff  #* (1-self.decor_cutoff)
            lri = lri / lri.sum([2,3], keepdim=True)

            #lre_masks = get_sparsity_masks(sampling_cf,self.long_cutoff, 1/3)
            #lre = sampling_cf * lre_masks

            #lre_sorted = sampling_cf.view(sampling_cf.shape[0], -1).sort(dim=1, descending=True)[1]
            #lre = self.lre_gauss * 0
            #lre[torch.arange(sampling_cf.shape[0], device=self.device)[:, None], lre_sorted] = self.lre_gauss

            #lre = (self.lateral_correlations_exc + self.lateral_weights_exc*1e-11) * self.long_cutoff
            lre =  self.long_cutoff * self.lateral_correlations_exc #* (1-self.decor_cutoff)
            #lre = self.envelope[:,:, self.env_pad:-self.env_pad, self.env_pad:-self.env_pad]  * self.long_cutoff
            #lre = lre * self.rolled_envelope[:,:, self.env_pad:-self.env_pad, self.env_pad:-self.env_pad] 
            lre = lre / lre.sum([2,3], keepdim=True)

            #lre = self.lateral_weights_exc * 4 + lre

            #lre = lre / 2.5

            #lre = lre * (1-self.blending_gauss) + self.blending_gauss / (np.pi * self.R_long**2) * 5

            #lre = lre / lre.sum([2,3], keepdim=True)

            #plast_strength = self.b.mean()

            #ret_strength = 0
            
            #lre = lre * plast_strength + self.lateral_weights_exc * ret_strength

            #lri = lri * (plast_strength + ret_strength)

            #lre = lre + self.rand * 0.5

            #lre = lre / lre.sum([2,3], keepdim=True)

            #cf = lre
            #sorted_vec = cf.view(cf.shape[0], -1).sort(dim=1, descending=True)
            #locs = (sorted_vec[0].cumsum(dim=1) < 0.95) * (sorted_vec[0].cumsum(dim=1) > 0)

            #locs = sorted_vec[1] * locs

            #vec = (lre * 0).view(self.sheet_size**2, -1)
            
            #vec[torch.arange(1600)[:,None], locs] = 1
            
            #vec = vec.view(lre.shape)

            #vec = vec / (vec.sum([2,3], keepdim=True) + 1e-11)

            #lre = lre.view(sampling_cf.shape)

            #print(lre.shape, self.afferent_weights.shape)

            if False:
                
                lre_padded = F.pad(lre, (self.window//2,self.window//2,self.window//2,self.window//2), mode='reflect')
                lre_padded = lre_padded / lre_padded.sum([2,3], keepdim=True) + 1e-11
                asym = (lre_padded * self.euclid).sum([2,3])
                
                self.rolls = self.rolls + asym * 32 * 1e-1 / 2
                self.rolls = self.rolls.clip(-self.env_pad, self.env_pad)
                #print(self.rolls)
    
                self.rolled_envelope = batch_roll_2d(self.envelope, -self.rolls.int())
    
                asym = asym.abs().mean()            
                #self.radial_b = self.radial_b * (1-1e-1) + asym * 1e-1

            if phi_long>1:

                lre = lre * self.long_sparsity_masks
                #lre -= (lre+self.long_sparsity_masks+(1-self.long_cutoff)).view(lre.shape[0], -1).min(1)[0].view(-1,1,1,1)
                #lre = lre ** self.lre_p
                lre = lre / (lre.sum([2,3], keepdim=True) + 1e-11)
    
                lri = lri * self.long_sparsity_masks
                #lri -= (lri+self.long_sparsity_masks+(1-self.long_cutoff)).view(lri.shape[0], -1).min(1)[0].view(-1,1,1,1)
                lri = lri ** self.lri_p
                lri = lri / (lri.sum([2,3], keepdim=True) + 1e-11)

            self.long_range_exc = lre
            self.long_range_inh = lri

            #self.lre_mag = (lre * self.euclid_distance).sum([2,3]).view(self.lre_mag.shape)
            #self.lri_mag = get_masses_and_spreads(lre, masses=self.long_cutoff.sum([1,2,3]) * 0.7, norm_flag=True)[1].view(self.lre_mag.shape)
            #self.lri_mag = get_masses_and_spreads(lri, masses=self.long_cutoff.sum([1,2,3]), norm_flag=True)[1].view(self.lre_mag.shape)
            self.lre_mag = get_masses_and_spreads(self.long_range_exc)[0].view(self.lre_mag.shape) / self.long_cutoff.sum([1,2,3]).view(self.lre_mag.shape)
            #self.lre_mag = (self.long_range_exc / self.long_range_exc.view(self.sheet_size**2, -1).max(1)[0].view(-1,1,1,1)).sum([2,3]).view(self.lre_mag.shape) / self.sheet_size**2
            self.sre_mag = get_masses_and_spreads(lre, masses=self.long_cutoff.sum([1,2,3])/3.5, norm_flag=True)[1].view(self.lre_mag.shape)
            #self.lre_mag = (lre * self.euclid_distance).sum([2,3]).view(self.sre_mag.shape)

        if False:
            
            aff_asym = (self.get_aff_weights() * self.aff_cartesian).sum([2,3]).view(self.aff_rolls.shape)

            self.aff_rolls = self.aff_rolls + aff_asym * 32 * 1e-2 
            self.aff_rolls = self.aff_rolls.clip(-self.aff_pad, self.aff_pad)

            aff_asym = torch.sqrt((aff_asym**2).sum(1, keepdim=True))
            self.radial_b = self.radial_b * (1-1e-1) + aff_asym * 1e-1

        if phi_mid>1 and False:
            #mri = mri * self.mid_sparsity_masks
            mri = mri ** self.mri_p
            mri = mri / (mri.sum([2,3], keepdim=True) + 1e-11)

        if phi_short>1 and False:
            #sre = sre * self.short_sparsity_masks
            sre = sre ** self.sre_p
            sre = sre / (sre.sum([2,3], keepdim=True) + 1e-11)

        self.mid_range_inh = mri
        self.short_range_exc = sre

        #self.sre_mag = (sre * self.euclid_distance).sum([2,3]).view(self.sre_mag.shape)
        #self.mri_mag = (mri * self.euclid_distance).sum([2,3]).view(self.mri_mag.shape)

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
            #self.mid_sparsity_masks = get_sparsity_masks(
            #                        self.lateral_correlations,
            #                        self.mid_cutoff * self.mri_mask, 
            #                        1/self.phi_mid
            #                    )
            mri = self.lateral_correlations * self.mid_cutoff * self.mri_mask
            mri = mri / mri.sum([2,3], keepdim=True)
            mri *= np.pi * self.R_pat**2
            #self.mri_norm = np.exp(-self.phi_mid + 1) * 0.8 + 0.2
            self.mid_sparsity_masks = (torch.rand(self.mid_range_inh.shape, device=self.device) * mri) > phi_mid / mri.max()
            #self.needs_update=True


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


    def aff_crop(self):

        device = self.retinotopic_bias.device
        B = self.sheet_size ** 2
        rf_size = self.rf_size
    
        # Offsets (dy, dx)
        offsets = self.aff_rolls.squeeze(-1).squeeze(-1).long()  # shape [B, 2]
    
        # Compute top-left corner for each crop
        y0 = self.aff_pad + offsets[:, 0]
        x0 = self.aff_pad + offsets[:, 1]
    
        # Coordinate grids for the receptive field
        y_idx = torch.arange(rf_size, device=device)
        x_idx = torch.arange(rf_size, device=device)
        yy, xx = torch.meshgrid(y_idx, x_idx, indexing='ij')
    
        # Shift grids per batch element
        yy = yy[None, :, :] + y0[:, None, None]
        xx = xx[None, :, :] + x0[:, None, None]
        batch_idx = torch.arange(B, device=device)[:, None, None]
    
        # Gather crops
        rb = self.retinotopic_bias[:,0]

        crops = rb[batch_idx, yy, xx]  # [B, 1, rf_size, rf_size]

        return crops[:,None]

        

        
