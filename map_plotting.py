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
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import seaborn as sns
from adjustText import adjust_text
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter

from wiring_efficiency_utils import *

def sample_and_plot(distribution, num_samples, sample_idx, ori_map=None, full=False):

    M = distribution.shape[-1]

    # Convert distribution to PyTorch tensor and flatten for sampling
    dist_tensor = torch.tensor(distribution.flatten(), dtype=torch.float)
    
    # Sample S locations from the distribution
    indices = torch.multinomial(dist_tensor, num_samples, replacement=True)

    if full:
        indices = torch.where(dist_tensor>0)[0]
        num_samples = indices.shape[0]
    
    # Convert flat indices back to 2D indices
    y, x = np.unravel_index(indices.numpy(), (M, M))
    
    # Get the center coordinates
    center_x = sample_idx % M
    center_y = sample_idx // M

    x = x / M
    y = y / M
    center_x = center_x / M
    center_y = center_y / M

    
    # Display the original image with HSV colormap and save it
    plt.imshow(ori_map.cpu(), cmap='hsv')
    plt.axis('off')
    plt.savefig('original_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Load the saved image using PIL
    image = Image.open('original_image.png')
    image_np = np.array(image)
    
    # Pad the image before blurring to avoid losing corners
    pad_size = 25  # The same size as the Gaussian kernel
    padded_image_np = cv2.copyMakeBorder(image_np, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    # Apply Gaussian blur to the padded image
    blurred_padded_image = cv2.GaussianBlur(padded_image_np, (5,5), 0)
    
    # Remove the padding after blurring
    blurred_image = blurred_padded_image[pad_size:-pad_size, pad_size:-pad_size]
    
    # Convert the blurred image to a tensor
    blurred_image_tensor = TF.to_tensor(Image.fromarray(blurred_image))
    
    # Add batch dimension and convert to float
    blurred_image_tensor = blurred_image_tensor.unsqueeze(0).float()
    
    # Remove the batch dimension
    blurred_image_tensor = blurred_image_tensor.squeeze(0)
    
    # Convert tensor to numpy array for plotting
    blurred_image_tensor = blurred_image_tensor.permute(1, 2, 0).numpy()
    
    # Display the upsampled blurred image
    plt.figure(figsize=(5, 5))
    #plt.imshow(blurred_image_tensor, alpha=0.15)
    plt.axis('off')

    k = blurred_image_tensor.shape[0]
    # Add scatter to the sampled points with random scatter
    x_scatter = x + np.random.randn(num_samples) * 7e-3  # Add random scatter to x coordinates
    x_scatter = np.clip(x_scatter, 0, 1) * k
    y_scatter = y + np.random.randn(num_samples) * 7e-3  # Add random scatter to y coordinates
    y_scatter = np.clip(y_scatter, 0, 1) * k

    colors = [blurred_image_tensor[int(y), int(x)] for x, y in zip(np.round(x_scatter), np.round(y_scatter))]
    
    # Add scatter to the sampled points and draw lines from center to each point
    for i in range(len(x)):
        plt.plot([center_x*k, x_scatter[i]], [center_y*k, y_scatter[i]], color='black', linestyle='-', linewidth=1, alpha=0.2, zorder=1)  # More transparent lines

    #plt.xlim(130,200)
    #plt.ylim(165 ,230)
    plt.scatter(x_scatter, y_scatter, color=colors, s=200, alpha=0.8, zorder=2, edgecolors=None)  # Add transparency to the sampled points
    
    plt.scatter(center_x*k, center_y*k, color='white', s=400, zorder=3, edgecolors=None)  # Plot the center
    plt.scatter(center_x*k, center_y*k, color='black', s=200, zorder=4)  # Plot the center
    plt.axis('off')
    plt.savefig('samples.svg', bbox_inches='tight', pad_inches=0)
    plt.close()

    resized_ori_map = F.interpolate(ori_map[None,None], blurred_image_tensor.shape[0])[0,0]
    sampled_oris = [resized_ori_map[int(y), int(x)] for x, y in zip(np.round(x_scatter), np.round(y_scatter))]

    plt.hist(sampled_oris, bins=13)
    plt.axis('off')
    plt.savefig('ori_hist.svg', bbox_inches='tight', pad_inches=0)
    plt.close()


# Function to animate an array as a useful visualisation
def animate(array, n_frames, cmap=None, interval=300):
    
    fig = plt.figure(figsize=(6,6))
    global i
    i = -2
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    im = plt.imshow(array[0], animated=True, cmap=cmap)

    def updatefig(*args):
        global i
        if (i < n_frames - 1):  # ensure that we don't go out of bounds
            i += 1
        im.set_array(array[i])
        return im,

    anim = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=interval, repeat=True)
    plt.close(fig)  # Prevents the static plot from showing in the notebook
    return HTML(anim.to_jshtml())  # Directly display the animation


def show_map(model, network, random_sample=None):

    plt.figure(figsize=(12, 14))
    titles = [
        "Current Input", "Afferent Weights", "Current Aff Response", "Inhibitory weights",
        "Lateral correlations", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "LRE", "L4 Afferent", "L4 Histogram",
        "Reconstruction", "Thresholds", "STD FRs", "Mean Frs"
    ]

    # Displaying the model's current input
    img = model.current_input[0, 0].detach().cpu()
    #c = model.rf_size // 2
    #img = img[c:-c,c:-c]
    plt.subplot(4, 4, 1)
    plt.imshow(img, cmap=cm.Greys)
    plt.title(titles[0])

    # Afferent weights of a random sample
    aff_weights = model.get_aff_weights()[random_sample, 0] #- model.afferent_weights[random_sample, 1]
    aff_weights[0,0] = 0
    plt.subplot(4, 4, 13)
    plt.imshow(aff_weights.detach().cpu())
    plt.title(titles[1])

    # Afferent weights of a random sample
    net_afferent = model.current_afferent[0,0].detach().cpu() - model.thresholds[0,0].detach().cpu()
    net_afferent_bar = net_afferent + 0
    net_afferent_bar[0,0] = 0
    plt.subplot(4, 4, 3)
    plt.imshow(net_afferent_bar)
    plt.title(titles[2])

    # Lateral correlations of the random sample
    plt.subplot(4, 4, 4)
    #plotvar = model.long_interactions[random_sample, 0]#* model.eye[random_sample, 0]
    inh = model.long_range_inh if model.long_range_inh.any() else model.mid_range_inh
    plotvar = inh[random_sample, 0]
    plotvar[0,0] = 0
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[3])

    # Lateral weights excitation of the random sample
    plt.subplot(4, 4, 5)
    plotvar = model.lateral_correlations[random_sample, 0]
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[4])

    # Model's current response
    plt.subplot(4, 4, 6)
    plt.imshow(model.current_response[0, 0].detach().cpu())
    plt.title(titles[5])

    # Histogram of the current response
    plt.subplot(4, 4, 7)
    hist = model.current_response.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0], range=(0,1))
    plt.title(titles[6])

    # Generate and display orientation and phase maps
    weights = model.get_aff_weights().clone()
    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping
    ori_map, phase_map, mean_tc = get_orientations(weights, gabor_size=model.rf_size)
    ori_map = ori_map.reshape(M, M).cpu()
    phase_map = phase_map.reshape(M, M).cpu()
    
    # Orientation map
    plt.subplot(4, 4, 8)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[7])

    # Orientation histogram
    plt.subplot(4, 4, 9)
    hist_map = ori_map.flatten()
    plt.hist(hist_map, bins=15)
    plt.title(titles[8])

    # LRE
    plt.subplot(4, 4, 10)
    if model.long_range_exc.any():
        plt.imshow(model.long_range_exc[random_sample,0].cpu())
    plt.title(titles[9])

    # Retinotopic Bias
    plt.subplot(4, 4, 11)
    detectors = get_detectors(model.rf_size, 1)
    plt.imshow(detectors[0,0].cpu())
    plt.title(titles[10])

    plt.subplot(4, 4, 12)
    plt.stairs(model.avg_hist.int(), torch.linspace(0,1,11), fill=True)
    plt.title(titles[11])

    reco_input = network['activ'](network['model'](model.current_response))[0,0].detach().cpu()
    # nn reconstruction
    plt.subplot(4, 4, 2)
    plt.imshow(reco_input)
    plt.title(titles[12])

    # afferent with thresholds
    plt.subplot(4, 4, 14)
    plt.imshow(model.thresholds[0,0].cpu())
    plt.title(titles[13])

    # thresholds
    #thresholds[0,0] = 0
    plt.subplot(4, 4, 15)
    plt.imshow(model.short_range_exc[random_sample,0].cpu())
    plt.title(titles[14])

    # thresholds
    plt.subplot(4, 4, 16)
    mean_fr = model.mean_fr[0,0].cpu()
    plt.imshow(mean_fr)
    plt.title(titles[15])

    print('Net Afferent Max: {:.3f}, Net Afferent Min: {:.3f}'. format(net_afferent.max(), net_afferent.min()))
    print('L4 Thresholds Max: {:.3f}, L4 Thresholds Min: {:.3f}'. format(model.thresholds.max(), model.thresholds.min()))
    print('Mean current response: {:.3f}'.format(model.current_response.mean()))
    print('L4 Strength: {:.3f} aff strength: {:.3f}'.format(model.strength, model.aff_strength))
    loss = torch.mean((reco_input - img)**2)
    print('Reco loss: {:.3f}%'.format(loss))


    plt.show()


def plot_absolute_phases(model,target_channel=0):

    # exctracting useful params
    rfs = model.afferent_weights.clone().cpu()
    aff_units = rfs.shape[-1]
    sheet_units = model.sheet_size
    channels = 1
    
    # making a meshgrid to localise any points within the aff cf
    rng = torch.arange(aff_units) - aff_units//2
    coordinates = torch.meshgrid(rng,rng)
    coordinates = torch.stack(coordinates)[None]
    
    # averaging over all locations to detect the greatest intensity
    rfs = rfs.view(-1,channels,aff_units,aff_units)[:,target_channel][:,None]
    rfs = rfs.repeat(1,2,1,1)
    c = (coordinates * rfs)
    c = c.sum([2,3]) * 2

    # organising everything into a grid and plotting the centre of mass of each point
    rng = torch.arange(sheet_units)
    topography = torch.meshgrid(rng,rng)
    topography = torch.stack(topography)
    topography = topography.reshape(2,-1)
    topography = (topography.T.float() + c).T

    plt.figure(figsize=(9,9))
    plt.scatter(topography[0],topography[1])

    # plotting the lines of the grid
    topography = topography.T.view(sheet_units,sheet_units,2)
    segs1 = topography
    segs2 = segs1.permute(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
    plt.tight_layout()
    plt.show()
    
