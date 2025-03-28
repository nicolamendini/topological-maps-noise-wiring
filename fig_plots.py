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
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from scipy.optimize import least_squares

from wiring_efficiency_utils import *

UNIT = 13
tickscale = 0.8
sizesvar = [40, 56, 69]
fs = 0.8

def make_fig1_plots(data_sp, data_topo):

    trialvar = np.array(data_sp['trialvar'])
    fontsize = 25
        
    # ----------------------- lambda
    
    peaks_sp = data_sp['peak_tracker'][-1][None]
    peaks_topo = data_topo['peak_tracker'][-1][None]
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    plt.xlabel('excitatory pool size       ', fontsize=fontsize, labelpad=10)
    plt.ylabel('norm. map scale       ', fontsize=fontsize, labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.xticks(fontsize=fontsize*fs)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.yticks(fontsize=fontsize*fs)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    def linear_func(x, m, b):
        return m*x + b
    
    x = np.round(trialvar**2 * np.pi) 
    y_sp = 1 / peaks_sp.flatten() **2 
    y_topo = 1 / peaks_topo.flatten() **2 

    
    line_range = np.array([x.min(), x.max()+10])

    lambda_popt, _ = curve_fit(linear_func, x, y_sp)
    plt.plot(line_range, linear_func(line_range, lambda_popt[0], lambda_popt[1]), color='#DD8452', linewidth=2, zorder=-1)
    
    lambda_popt, _ = curve_fit(linear_func, x, y_topo)
    plt.plot(line_range, linear_func(line_range, lambda_popt[0], lambda_popt[1]), color='#4C72B0', linewidth=2, zorder=-1)

    for s in range(len(sizesvar)):
        ax.scatter(x, y_topo, label='N='+str(sizesvar[s]**2), s=150, color='#4C72B0', marker='o', alpha=0.5)
        ax.scatter(x, y_sp, label='N='+str(sizesvar[s]**2), s=150, color='#DD8452', marker='s', alpha=0.5)

    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #plt.ylim(2e0, 9e2)
    #plt.xlim(2e1, 1e3)

    plt.savefig('./figures/fig1/lambda.svg')
    plt.close()

    # ----------------------- animals lateral connectivity

    labels = ['gh', 'rat', 'squ', 'rab', 'ts', 'cat', 'fer', 'sm', 'sm(V2)', 'om', ' mac', ' mac(V2)']
    connectivity_length = [153000,  153000, 539000, 539000, 1487000, 2389000, 2389000, 1182000, 2224000, 1182000, 5970000, 5998000]
    v1_lambda = np.array([4, 4, 4, 4, 29, 42, 38, 29, 34, 36, 42, 49])

    # Create figure and axis
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    
    plt.ylabel("norm. map scale      ", fontsize=fontsize, labelpad=9)
    plt.xlabel('plastic pool size           ', fontsize=fontsize)
    plt.xticks(fontsize=fontsize*fs)
    plt.yticks(fontsize=fontsize*fs)
    ax.yaxis.get_offset_text().set_fontsize(fontsize) 

    # Scatter plot
    scatter = ax.scatter(torch.tensor(connectivity_length[:4]) + torch.randn(4,)*2e4, torch.tensor(v1_lambda[:4]**2) + torch.randn(4,)*0.5, s=200, linewidth=2, color='#DD8452', marker='s', alpha=0.9)
    scatter = ax.scatter(connectivity_length[4:], v1_lambda[4:]**2, s=200, linewidth=2, color='#4C72B0', marker='o', alpha=0.9)

    def linear_func(x, m, b):
        return m*x + b
    
    popt, pcov = curve_fit(linear_func, connectivity_length[4:], v1_lambda[4:]**2)
    x = np.array([1e6, 1e7])
    y = linear_func(x, popt[0], popt[1]) - 4

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.ylim(1.1e1, 5e3)
    plt.xlim(1.1e5, 1e7)

    # Save the figure
    plt.savefig('./figures/fig1/connectivity.svg')
    plt.close()

    # ----------------------- fits

    # Create figure and axis
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    
    plt.ylabel("norm. map scale      ", fontsize=fontsize, labelpad=9)
    plt.xlabel('plastic pool size           ', fontsize=fontsize)
    plt.xticks(fontsize=fontsize*fs)
    plt.yticks(fontsize=fontsize*fs)
    ax.yaxis.get_offset_text().set_fontsize(fontsize) 

    # Scatter plot
    scatter = ax.scatter(torch.tensor(connectivity_length[:4]) + torch.randn(4,)*2e4, torch.tensor(v1_lambda[:4]**2) + torch.randn(4,)*0.5, s=200, linewidth=2, color='#DD8452', marker='s')
    scatter = ax.scatter(connectivity_length[4:], v1_lambda[4:]**2, s=200, linewidth=2, color='#4C72B0', marker='o')
    
    popt, pcov = curve_fit(linear_func, connectivity_length[4:], v1_lambda[4:]**2)
    x = np.array([1e6, 1e7])
    y = linear_func(x, popt[0], popt[1]) #- 4
    ax.plot(x, y, linewidth=2, color='#4C72B0')
    ax.plot([0, 1e6], [v1_lambda[0]**2, v1_lambda[0]**2], linewidth=2, color='#DD8452')
    ax.plot([1e6, 1e6], [v1_lambda[0]**2, y.min()], linewidth=2, color='black', linestyle='--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.ylim(1.1e1, 5e3)
    plt.xlim(1.1e5, 1e7)

    # Save the figure
    plt.savefig('./figures/fig1/fits.svg')
    plt.close()

    # ----------------------- animals lateral connectivity normalised

    labels = ['gh', 'rat', 'squ', 'rab', 'ts', 'cat', 'fer', 'sm', 'sm(V2)', 'om', ' mac', ' mac(V2)']
    species_indices = torch.arange(len(v1_lambda))+1

    # Create figure and axis
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(bottom=0.2, left=0.25)
    
    plt.ylabel('norm. map scale          ', fontsize=fontsize, labelpad=10)
    plt.xlabel('sorted animal species', fontsize=fontsize, labelpad=20)
    plt.xticks(ticks=[], fontsize=fontsize*fs)
    plt.yticks(fontsize=fontsize*fs)

    sorted_lambda = torch.tensor(v1_lambda).sort()
    v1_lambda = sorted_lambda[0]
    sorted_indices = sorted_lambda[1]
    labels = [labels[s] for s in sorted_indices]

    ax.plot([1, 4.5], [v1_lambda[0]**2, v1_lambda[0]**2], color='#DD8452', linewidth=2)

    popt, _ = curve_fit(linear_func, np.linspace(5, 12, 8), v1_lambda[4:]**2)
    
    x = np.array([4.5, 13])
    y = linear_func(x, popt[0], popt[1])
    ax.plot(x, y*1.1, color='#4C72B0', linewidth=2)

    # Scatter plot
    scatter = ax.scatter(species_indices[:4], v1_lambda[:4]**2, color='#DD8452', s=200, marker='s', alpha=0.9)
    scatter = ax.scatter(species_indices[4:], v1_lambda[4:]**2, color='#4C72B0', s=200, marker='o', alpha=0.9)

    ax.set_yscale('log')
    plt.ylim(11, 8e3)

    # Save the figure
    plt.savefig('./figures/fig1/species_norm.svg')
    plt.close()

def make_fig2_plots(data_sp, data_topo):

    SCALE = 1
    trialvar = np.array(data_sp['trialvar'])**2 * np.pi
    colors = ['#a6b8e1', '#f3c6aa', '#a8d4b0']
    fontsize = 22
    ratios = ['1', '2', '3']

    def dec_exp_func(x, a, b, c):
        return a * np.exp(-x * b) + c

    # ----------------------- accuracy curves
    
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.16, left=0.18)
    accuracy_sp = data_sp['reco_tracker'][:,:,-10000:].mean(2)
    accuracy_topo = data_topo['reco_tracker'][:,:,-10000:].mean(2)

    acc_cat = torch.cat([accuracy_sp, accuracy_topo], dim=1)
    x_fit = trialvar.repeat(2)
    x_plot = torch.linspace(0, trialvar.max(), 100)
    
    plt.xlabel('excitatory pool size      ', fontsize=fontsize)
    plt.ylabel('accuracy', fontsize=fontsize)
    plt.xticks(fontsize=fontsize*tickscale)
    plt.yticks(fontsize=fontsize*tickscale)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    palette1 = sns.color_palette("magma", len(sizesvar))

    opt = np.array([
        [0.110, 0.0030, 0.735],
        [0.095, 0.0023, 0.750],
        [0.095, 0.0017, 0.750]
    ])
    
    plt.ylim(0.72, 0.86)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    
    for s in range(len(sizesvar)):

        y_acc_fit = dec_exp_func(x_plot, opt[s,0], opt[s,1], opt[s,2])
        
        plt.scatter(trialvar, accuracy_sp[s], s=60, marker='s', linewidth=1.5, label='V1:LGN='+ratios[s], alpha=0.7, color=palette1[s])
        plt.scatter(trialvar, accuracy_topo[s], s=60, marker='o', linewidth=1.5, label='V1:LGN='+ratios[s], alpha=0.7, color=palette1[s])
        plt.plot(x_plot, y_acc_fit, linewidth=1.5, zorder=-1, color=palette1[s], alpha=0.7)

    #plt.legend(fontsize=12, frameon=False)
    plt.savefig('./figures/fig2/accuracy.svg')
    plt.close() 
    
    # ----------------------- complexity curves
            
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    complexity_sp = data_sp['se_pca_tracker'].float()
    complexity_topo = data_topo['se_pca_tracker'].float()
    label = 'PCA'
    baseline = 112 

    complexity_sp /= baseline
    complexity_topo /= baseline

    com_cat = torch.cat([complexity_sp, complexity_topo], dim=1)

    plt.subplots_adjust(bottom=0.16, left=0.2)
    plt.xlabel('excitatory pool size      ', fontsize=fontsize)
    plt.ylabel('dimensionality      ', fontsize=fontsize, labelpad=10)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.xticks(fontsize=fontsize*tickscale)
    plt.yticks(fontsize=fontsize*tickscale)

    opt = np.array([
        [3.5, 0.008, 0.43],
        [3.5, 0.0055, 0.53],
        [3.5, 0.004, 0.6],
    ])

    for s in range(len(sizesvar)):

        y_com_fit = dec_exp_func(x_plot, opt[s,0], opt[s,1], opt[s,2])
        
        plt.scatter((trialvar), complexity_sp[s], s=60, marker='s', linewidth=1.5, label='V1:LGN='+ratios[s], alpha=0.7, color=palette1[s])
        plt.scatter((trialvar), complexity_topo[s], s=60, marker='o', linewidth=1.5, label='V1:LGN='+ratios[s], alpha=0.7, color=palette1[s])
        plt.plot(x_plot, y_com_fit, linewidth=1.5, zorder=-1, color=palette1[s], alpha=0.7)
        

    plt.plot([0, trialvar[-1]], [1, 1], linewidth=1.5, linestyle='--', color='black', zorder=-1)
    ax.set_yscale('log')

    plt.ylim(1e-1, 1e1)


    #plt.legend(fontsize=fontsize*tickscale, frameon=False)
    plt.savefig(f'./figures/fig2/complexity_{label}.svg')
    plt.close() 
        
    # ----------------------- accuracy comxplexity tradeoff curves
    
    fig = plt.figure(figsize=(6,5.5))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.16, left=0.2)
    plt.gca().yaxis.get_offset_text().set_fontsize(16) 
    plt.ylabel('dimensionality      ', fontsize=fontsize, labelpad=10)
    plt.xlabel('accuracy', fontsize=fontsize)

    plt.xticks(fontsize=fontsize*tickscale)
    plt.yticks(fontsize=fontsize*tickscale)

    plt.xlim(0.72, 0.86)
    plt.ylim(1e-1, 1e1)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))    
    
    tradeoff_popt = []
    
    for s in range(len(sizesvar)):
        
        plt.scatter(accuracy_sp[s], complexity_sp[s], s=60, marker='s', linewidth=1.5, label='V1:LGN='+ratios[s], alpha=0.7, color=palette1[s])
        plt.scatter(accuracy_topo[s], complexity_topo[s], s=60, marker='o', linewidth=1.5, label='V1:LGN='+ratios[s], alpha=0.7, color=palette1[s])

    plt.plot([0, 1], [1, 1], linewidth=1.5, linestyle='--', color='black', zorder=-1)
    ax.set_yscale('log')

    #plt.legend(fontsize=fontsize*tickscale, frameon=False)
    plt.savefig('./figures/fig2/tradeoff.svg')
    plt.close() 


def make_fig3_plots(data_sp, data_topo):

    # ----------------------- noise trajectories

    noise_rob_topo = data_topo['noise_rob']
    noise_rob_sp = data_sp['noise_rob']
    noise = torch.linspace(0, 0.5, noise_rob_topo.shape[-1])
    trialvar = np.array(data_topo['trialvar'])**2
    trialvar = np.round(trialvar * np.pi) 

    custom_palette = sns.color_palette("viridis", trialvar.shape[0])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_palette)

    fontsize = 23

    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.subplots_adjust(bottom=0.20, left=.20)
    
    plt.xlabel('noise intensity γ', fontsize=fontsize)
    plt.ylabel('stability', fontsize=fontsize)

    plt.xticks(fontsize=fontsize*fs)
    plt.yticks(ticks=[0.9, 0.95, 1], fontsize=fontsize*fs)

    plt.ylim(0.9, 1)
        
    for t in range(noise_rob_topo.shape[1]):

        plt.plot(noise, noise_rob_topo[-1, t], linewidth=2)
        #plt.scatter(noise, noise_rob[1, t, 0], s=20, zorder=t)

        #fit_noise = curve_fit(bumped_decay, np.array(noise), np.array(noise_rob[0, t, 0]), sigma=np.linspace(1.1, 1, 10))[0]
        #x = np.linspace(0, noise.max(), 10)
        #plt.plot(x, bumped_decay(x, fit_noise[0], fit_noise[1], fit_noise[2],  fit_noise[3], fit_noise[4], fit_noise[5]), color='black', linewidth=1)

    #plt.legend(fontsize=ticksize, frameon=False)
    plt.plot([0, noise.max()], [.95, .95], linewidth=1.5, linestyle='--', color='black', zorder=-1)
    plt.savefig('./figures/fig3/noise_trajetories.svg')
    plt.close() 

    # ----------------------- cumulative curves

    def log_fit(x, a, b):
            return a*np.log(x/b + 1)

    fig = plt.figure()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.2, left=0.2)
    
    plt.xlabel('excitatory pool size        ', fontsize=fontsize)
    plt.ylabel('noise robustness Γ', fontsize=fontsize)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    plt.xticks(fontsize=fontsize*tickscale)
    plt.yticks(fontsize=fontsize*tickscale)

    #plt.ylim(0, 1)

    val = 0.9
    max_noise_topo = torch.tensor([find_val_loc(item, val) for item in noise_rob_topo[-1, :]])
    max_noise_topo = max_noise_topo * noise.max() / noise.shape[-1] 

    max_noise_sp = torch.tensor([find_val_loc(item, val) for item in noise_rob_sp[-1, :]])
    max_noise_sp = max_noise_sp * noise.max() / noise.shape[-1] 

    #fit_rob = curve_fit(log_fit, np.array(trialvar).repeat(2), torch.cat([max_noise_topo, max_noise_sp]))[0]
    x = np.linspace(trialvar.min() - 0.1, trialvar.max() + 0.1, 100)
    #plt.plot(x, log_fit(x, fit_rob[0], fit_rob[1]), color='black', linewidth=1, zorder=1)

    #a, b = exact_log_fit(trialvar.repeat(2).flatten(), torch.cat([max_noise_topo[None], max_noise_sp[None]], dim=0).flatten()

    a,b = (0.07, 6.5)

    plt.scatter(trialvar, max_noise_topo, s=150, marker='o', linewidth=2.5, zorder=2, color='#4C72B0', label='topological')
    plt.scatter(trialvar, max_noise_sp, s=150, marker='s', linewidth=2.5, zorder=2, color='#DD8452', label='salt and pepper')

    #a, b = curve_fit(log_fit, trialvar, max_noise_topo)[0]
    plt.plot(x, log_fit(x, a, b), zorder=-1, linewidth=1.5, color='black', alpha=0.7)

    #plt.ylim(0.25, 0.55)
    plt.legend(fontsize=18, frameon=False, loc='lower right')
    plt.savefig('./figures/fig3/cumulative_robustness.svg')
    plt.close() 



def make_fig4_plots(data_sp, data_topo):

    rob_sp = data_sp['sparsity_rob']
    rob_topo = data_topo['sparsity_rob']

    # ------------ many curves

    sizesvar = [data_sp['map_tracker'].shape[-1]]
    trialvar = np.round(np.array(data_sp['trialvar'])**2 * np.pi)
    fontsize=22
    tickscale=0.8
    trials = trialvar.shape[0]
    sparsity = np.arange(rob_sp.shape[-1])*2+1
    
    custom_palette = sns.color_palette("viridis", trialvar.shape[0])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_palette)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylabel('stability', fontsize=fontsize)
    plt.xlabel('sparsity    ', fontsize=fontsize)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.subplots_adjust(bottom=0.15, left=0.16)
    plt.xticks(fontsize=fontsize*tickscale)
    plt.yticks(ticks=[0.9, 0.95, 1], fontsize=fontsize*tickscale)
    plt.ylim(0.9, 1)
    
    colors = ['#55a868', '#c44e52']
    
    for i in range(trials):
        plt.plot(sparsity, data_topo['sparsity_rob'][-1, i, 0], linewidth=2)
    
    plt.plot([1, sparsity.max()], [.95, .95], linewidth=1.5, linestyle='--', color='black', zorder=-1)
    plt.savefig('./figures/fig4/curves.svg')
    plt.close() 


    # ---------- trends

    def linear_func(x, m, b):
        return m*x + b

    fontsize=22
    tickscale=0.8
    
    colors = ['#4c72b0', '#dd8452']
    custom_palette = sns.color_palette("viridis", 20)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_palette)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylabel('sparsity robustness    ', fontsize=fontsize)
    plt.xlabel('excitatory pool size      ', fontsize=fontsize)
    
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.yticks(fontsize=fontsize*tickscale)
    plt.xticks(fontsize=fontsize*tickscale)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    x = np.linspace(trialvar.min(), trialvar.max(), 100)
    val = 0.95
        
    res = [find_val_loc(item, val)*2 + 1 for item in rob_topo[-1, :, 0]]
    
    plt.scatter(trialvar, res, s=150, marker='o', linewidth=2.5, label='topological', zorder=3, color=colors[0])
    m, b = curve_fit(linear_func, trialvar, res)[0]
    plt.plot(x, linear_func(x, m, b), zorder=-1, linewidth=1.5, color='#4C72B0', alpha=0.7)
        
    
    res = [find_val_loc(item, val)*2 + 1 for item in rob_sp[-1, :, 0]]
    
    plt.scatter(trialvar, np.array(res), s=150, marker='s', linewidth=2.5, label='salt and pepper', zorder=3, color=colors[1])
    m, b = curve_fit(linear_func, trialvar, res)[0]
    plt.plot(x, linear_func(x, m, b), zorder=-1, linewidth=1.5, color='#DD8452', alpha=0.7)
    
    plt.legend(fontsize=18, frameon=False)
    plt.savefig('./figures/fig4/trends.svg')
    plt.close() 

    # ---------- gap plot of costs

    fontsize=22
    tickscale=0.8
    
    map_sizes = np.sqrt(np.linspace(0.45**2, 4**2))
    custom_palette = sns.color_palette("viridis", 20)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_palette)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.ylabel('', fontsize=fontsize)
    plt.xlabel('', fontsize=fontsize)
    
    plt.subplots_adjust(bottom=0.2, left=0.2)
    yticks = np.linspace(0, 100, 4)
    xticks = np.linspace(0, 2700, 4)
    plt.yticks(fontsize=fontsize*tickscale)
    plt.xticks(fontsize=fontsize*tickscale)
    
    colors = ['#55a868', '#c44e52']
    x = np.linspace(0, 8, 8)
    
    radii = np.linspace(3.5e4, 1.5e6, trials)
    trialvar = np.sqrt(np.linspace(4**2, 30**2, trials))
    
    costs = collect_wiring_pool(radii, trialvar, r_0=0)

    s = 5
    print(costs[:, s], trialvar[s])

    return
    
    map_sizes = costs.argmin(0) / trialvar.shape[0] * trialvar.max()
    map_sizes = np.round(map_sizes**2 * 25 * np.pi) + 1

    nglobal = np.round(radii**2 * np.pi)
    c = map_sizes / map_sizes.max()

    plt.scatter(nglobal, map_sizes, s=200, marker='o', linewidth=1, edgecolor='black', c=c, cmap=cm.viridis)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylim(0.5, 4000)
    plt.xlim(1e3, 5e4)
    
    plt.legend(fontsize=fontsize*tickscale, frameon=False)
    plt.savefig('./figures/fig5/final.svg')
    plt.close() 


    # ---------- map cost curves

    plt.ylabel('connection cost   ', fontsize=fontsize)
    plt.xlabel('', fontsize=fontsize)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.2, left=0.2)
    yticks = np.linspace(0, 1e5, 4)
    xticks = np.linspace(0, 5e4, 4)
    plt.yticks(ticks=yticks, fontsize=fontsize*tickscale)
    plt.xticks(ticks=xticks, fontsize=fontsize*tickscale)

    plt.ylim(3e2, 5e3)
    plt.xlim(1e3, 5e4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    for i in range(trials):
    
        plt.plot(nglobal[::3], costs[i][::3])
    
    plt.savefig('./figures/fig5/costs.svg')




def plot_long_sparsification(connection_field, masks, jitter_strength=0.3, S=1815, marker_size=200, fontsize=60):

    if S is None:
        S = random.randint(0, connection_field.shape[-1]**2)
        print('sample: ', S)

    #ori_map, _, _ = get_orientations(model.afferent_weights, gabor_size=model.rf_size)
    #ori_map = ori_map.view(connection_field.shape[-1], connection_field.shape[-1])

    shape = lr_comp.shape[-1]
    x = np.arange(shape)
    y = x.repeat(shape).reshape(shape, shape)
    x = np.tile(x, shape).reshape(shape, shape)
    coords = np.concatenate([x[None], y[None]], axis=0).astype(float)

    coords_sparse = coords[:, lr_comp[S,0].cpu().bool()]
    jitter_sparse = torch.randn(coords_sparse.shape).numpy() * jitter_strength
    coords_sparse = np.concatenate([coords_sparse, lr_comp[S][lr_comp[S]!=0][None].cpu()], axis=0)

    coords_unselected = coords[:, (~lr_comp[S,0].cpu().bool() * cutoffs[S,0].cpu())]
    jitter_unselected = torch.randn(coords_unselected.shape).numpy() * jitter_strength

    alphas = lr_comp[S][lr_comp[S]!=0].cpu() 
    #alphas = alphas**p.cpu()

    plt.figure(figsize=(16,7))
    plt.subplot(1,2,1)
    plt.scatter(coords_unselected[0]+jitter_unselected[0], coords_unselected[1]+jitter_unselected[1], color='white', s=marker_size, linewidths=1, edgecolor='grey')
    plt.scatter(coords_sparse[0]+jitter_sparse[0], coords_sparse[1]+jitter_sparse[1], c=alphas, cmap=cm.Reds, s=marker_size, linewidths=1, edgecolor='grey')
    plt.axis('off')

    if False:
        ax = plt.subplot(1,2,2)
        plt.xlabel('weight (a.u.)', fontsize=fontsize, labelpad=10)
        plt.ylabel('No.connections', fontsize=fontsize, labelpad=10)
        plt.xticks(ticks=[0, 1e-2], labels=[0, 1], fontsize=fontsize*0.8)
        plt.yticks(ticks=[0, 1.5e6], labels=[0, 1.5], fontsize=fontsize*0.8)
        conn_hist = lr_comp[plastic_sparsity_masks.bool()].flatten().cpu()
        plt.hist(conn_hist, bins=15, color='black')
        plt.xlim(0, 1e-2)
        plt.ylim(0, 1.5e6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)    
        ax.spines['bottom'].set_linewidth(8)
        ax.spines['left'].set_linewidth(8)
        ax.tick_params(axis='both', which='major', width=5, length=10)
        ax.tick_params(axis='both', which='minor', width=5, length=10)

    plt.show()


def plot_exc_inh_samples(model, batch, jitter_strength=0.2, S=1830, batch_idx=6):

    if batch_idx==-1:
        batch_idx = random.randint(0, batch.shape[0] - 1)
        print(batch_idx)
        
    image = batch[batch_idx, 0:1][None].flip(1)

    fixed_sparsity_masks = get_sparsity_masks(
                    model.lateral_weights_exc.clone(), 
                    model.exc_cutoff.clone(), 
                    1, 
                    keep_centre=False
                )

    plastic_sparsity_masks = get_sparsity_masks(
                    model.mid_range_inhibition.clone(), 
                    model.untuned_inh.clone(), 
                    1/20
                )

    p = find_norm_p(model.mid_range_inhibition.clone(), plastic_sparsity_masks)
    e = find_norm_p(model.lateral_weights_exc.clone(), fixed_sparsity_masks)   

    model(image, rf_grids,adaptation=False)
    curr_res = model.current_response[0,0].cpu()

    model(image, rf_grids,adaptation=False, pla_sparsity_masks=plastic_sparsity_masks, fix_sparsity_masks=fixed_sparsity_masks, e_norm=e, p_norm=p)
    curr_sparse_res = model.current_response[0,0].cpu()

    inh_comp =  model.mid_range_inhibition.clone() * plastic_sparsity_masks
    inh_comp = inh_comp**p
    inh_comp /= inh_comp.sum([-1,-2], keepdim=True) + 1e-11
    exc_comp = model.lateral_weights_exc.clone() * fixed_sparsity_masks
    exc_comp = exc_comp**e
    exc_comp /= exc_comp.sum([-1,-2], keepdim=True) + 1e-11

    dense_connections =  model.lateral_weights_exc.clone() - model.mid_range_inhibition.clone()
    sparse_connections = exc_comp - inh_comp

    
    shape = curr_res.shape[-1]
    x = np.arange(shape)
    y = x.repeat(shape).reshape(shape, shape)
    x = np.tile(x, shape).reshape(shape, shape)
    coords = np.concatenate([x[None], y[None]], axis=0).astype(float)

    coords_dense = coords[:, dense_connections[S,0].cpu().bool()]
    jitter_dense = torch.randn(coords_dense.shape).numpy() * jitter_strength
    coords_dense = np.concatenate([coords_dense, dense_connections[S][dense_connections[S]!=0][None].cpu()], axis=0)
    
    coords_sparse = coords[:, sparse_connections[S,0].cpu().bool()]
    jitter_sparse = torch.randn(coords_sparse.shape).numpy() * jitter_strength
    coords_sparse = np.concatenate([coords_sparse, sparse_connections[S][sparse_connections[S]!=0][None].cpu()], axis=0)

    # Define the colormap and normalization
    # We'll create a colormap with Reds for negative values and Blues for positive values.
    cmap = mcolors.ListedColormap(["red", "blue"])
    norm = mcolors.TwoSlopeNorm(vmin=-1e-3, vcenter=0, vmax=5e-3)
    
    # Create a custom colormap that blends 'Reds' and 'Blues' based on values
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        ["blue", "white", "red"],
        N=256
    )

    alphas_sparse = np.abs(coords_sparse[2]) > 2e-3

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(curr_res, cmap=cm.Greys)
    plt.scatter(coords_dense[0]+jitter_dense[0], coords_dense[1]+jitter_dense[1], c=coords_dense[2], cmap=cmap, norm=norm, s=150, linewidths=1)

    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(curr_sparse_res, cmap=cm.Greys)
    plt.scatter(coords_sparse[0]+jitter_sparse[0], coords_sparse[1]+jitter_sparse[1], c=coords_sparse[2], cmap=cmap, norm=norm, s=150, linewidths=1, alpha=alphas_sparse)
    plt.show()

    similarity = cosim(curr_res[None,None], curr_sparse_res[None,None])
    print('cosine similarity: ', similarity)

