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
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from matplotlib import colors as mcolors
import matplotlib as mpl
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

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


def animate(array, n_frames=None, cmap="viridis", interval=300):
    if n_frames is None:
        n_frames = array.shape[0]

    # Convert torch.Tensor → numpy if needed
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()

    vmin, vmax = array.min(), array.max()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # fill entire figure, no margins

    # Transparent figure background (optional)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("black")  # optional: set to match your data background

    im = ax.imshow(array[0], animated=True, cmap=cmap, vmin=vmin, vmax=vmax)

    def update(frame):
        im.set_array(array[frame])
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True, repeat=True
    )

    plt.close(fig)
    return HTML(anim.to_jshtml())


def show_map(model, network, random_sample=None):

    plt.figure(figsize=(12, 14))
    titles = [
        "Current Input", "Afferent Weights", "Current Aff Response", "Inhibitory weights",
        "Lateral correlations", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "LRE", "Fourier domain", "Mean Histogram",
        "Reconstruction", "Thresholds", "Excitatory weights", "Mean Frs"
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
    plt.subplot(4, 4, 12)
    plt.imshow(aff_weights.detach().cpu())
    plt.title(titles[1])

    # Afferent weights of a random sample
    net_afferent = model.current_afferent[0,0].detach().cpu() - model.thresholds[0,0].detach().cpu()
    net_afferent_bar = net_afferent + 0
    net_afferent_bar[0,0] = 0
    plt.subplot(4, 4, 2)
    plt.imshow(net_afferent_bar)
    plt.title(titles[2])

    # Lateral correlations of the random sample
    plt.subplot(4, 4, 4)
    #plotvar = model.long_interactions[random_sample, 0]#* model.eye[random_sample, 0]
    inh = model.inh
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
    plt.imshow(model.current_response[0, 0].detach().cpu(), cmap=cm.Greys)
    plt.title(titles[5])

    # Histogram of the current response
    plt.subplot(4, 4, 7)
    hist = model.current_response.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0], range=(0,1))
    plt.title(titles[6])

    # Generate and display orientation and phase maps
    weights = model.get_aff_weights().clone()
    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping
    ori_map = detect_orientation_map_from_aff_weights(model.get_aff_weights())['pref'] / 180 * np.pi
    ori_map = ori_map.reshape(M, M).cpu()
    
    # Orientation map
    plt.subplot(4, 4, 9)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[7])

    # Orientation histogram
    plt.subplot(4, 4, 10)
    hist_map = ori_map.flatten()
    plt.hist(hist_map, bins=10)
    plt.title(titles[8])

    # Retinotopic Bias
    plt.subplot(4, 4, 11)
    _,ring,_ = get_typical_dist_fourier(ori_map, 0)
    plt.imshow(ring.cpu(), cmap=cm.Greys)
    plt.title(titles[10])

    plt.subplot(4, 4, 8)
    plt.stairs(model.avg_hist.int(), torch.linspace(0,1,11), fill=True)
    plt.title(titles[11])

    reco_input = network['activ'](network['model'](model.current_response))[0,0].detach().cpu()
    # nn reconstruction
    plt.subplot(4, 4, 14)
    plt.imshow(reco_input)
    plt.title('reco')

    # thresholds
    #thresholds[0,0] = 0
    plt.subplot(4, 4, 3)
    plt.imshow(model.thresholds.view(M,M).cpu())
    plt.title('thresh')

    exc = model.s_exc if not model.microcolumnar else model.l_exc
    plt.subplot(4, 4, 13)
    plt.imshow(exc[random_sample,0].view(M,M).cpu())
    plt.title(titles[-7])

    plt.subplot(4,4,15)
    plt.plot(model.response_tracker[:model.iterations].sum([1,2,3]).cpu(), color='black')
    plt.title('mean act')

    plt.subplot(4,4,16)
    plt.imshow(model.lateral_correlations[random_sample,0].cpu())
    plt.title('exc_corr')


    print('Net Afferent Max: {:.3f}, Net Afferent Min: {:.3f}'. format(net_afferent.max(), net_afferent.min()))
    print('L4 Thresholds Max: {:.3f}, L4 Thresholds Min: {:.3f}'. format(model.thresholds.max(), model.thresholds.min()))
    print('Mean current response: {:.3f}'.format(model.current_response.mean()))
    loss = torch.mean((reco_input - img)**2)
    print('Reco loss: {:.3f}%'.format(loss))


    plt.show()


def plot_absolute_phases(rfs,target_channel=0,figpath=None,ori_map=None):

    # exctracting useful params
    rfs = rfs.clone().cpu()
    aff_units = rfs.shape[-1]
    sheet_units = int(np.sqrt(rfs.shape[0]))
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

    plt.figure(figsize=(10,10))
    if ori_map is not None:
        plt.scatter(topography[0],topography[1], s=50, c=ori_map.flatten() / np.pi, cmap='hsv', edgecolor='black')
    else:
        plt.scatter(topography[0],topography[1], s=10, color='black')

    # plotting the lines of the grid
    topography = topography.T.view(sheet_units,sheet_units,2)
    segs1 = topography
    segs2 = segs1.permute(1,0,2)
    plt.gca().add_collection(LineCollection(segs1, linewidth=1, color='black', zorder=-1))
    plt.gca().add_collection(LineCollection(segs2, linewidth=1, color='black', zorder=-1))

    #plt.ylim(0, 40)
    #plt.xlim(0, 40)

    if figpath:
        plt.savefig(figpath)

    plt.show()

    return topography.view(-1,2)


def plot_mexican_hat(Z, r=None):
    # Get dimensions
    h, w = Z.shape
    
    # Generate X and Y grid centered at 0 (so circle mask is symmetric)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    y = np.linspace(-(h-1)/2, (h-1)/2, h)
    X, Y = np.meshgrid(x, y)
    
    # Create a mask for outside the circle
    if r is not None:
        mask = np.sqrt(X**2 + Y**2) > r
        Z = Z.clone()
        Z[mask] = np.nan  # masked region

    # Set up figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Colormap (blue for negative, red for positive)
    cmap = cm.coolwarm
    norm = plt.Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))

    # Plot surface (NaNs are simply skipped -> background shows through)
    surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Z)),
                           rstride=1, cstride=1, antialiased=True)

    # Remove axes for a clean look
    ax.set_axis_off()
    
    # Set camera view
    ax.view_init(elev=20, azim=80)

    plt.tight_layout()
    plt.show()


def plot_tensor_grid(tensor, s, figpath=None):
    """
    Plots the first s^2 slices from the last dimension of a tensor of shape (N, N, N^2)
    in an s x s grid of subplots with minimal spacing and no axes/labels.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, N, N^2)
        s (int): Number of rows and columns in the subplot grid
    """
    N = tensor.shape[0]
    assert tensor.shape[1] == N, "Tensor must be cubic in first two dims"
    assert tensor.shape[2] >= s**2, "Tensor's last dimension must have at least s^2 elements"
    
    # Select first s^2 slices along the last dimension
    slices = tensor[:, :, :s**2]  # shape: (N, N, s^2)
        
    # Create figure with tight layout
    fig, axes = plt.subplots(s, s, figsize=(10,10))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    for i in range(s**2):
        axes[i].imshow(slices[:, :, i], cmap='Greys')
        axes[i].axis('off')  # remove axes
    
    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    if figpath:
        plt.savefig(figpath, dpi=300)
    
    plt.show()

    
def plot_umap_with_angles_3d(
    code_tracker,
    variable,
    n_components=3,
    n_neighbors=15,
    min_dist=0.5,
    metric="euclidean",
    random_state=42,
    fs=22,
    figpath=None
):
    """
    Plot UMAP embedding of codes with HSV color based on `variable`.
    Axes, grids, ticks, panes, and labels are completely removed.
    """

    assert n_components in (2, 3), "n_components must be 2 or 3"

    # Convert codes to 2D array [N, D]
    codes = torch.cat([c.view(1, -1) for c in code_tracker], dim=0)
    codes = codes.detach().cpu().numpy()

    variable = np.asarray(variable, dtype=float)
    variable = variable - variable.min()
    variable = variable / (variable.max() + 1e-11)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(codes)

    # HSV → RGB
    hsv_colors = np.zeros((len(variable), 3))
    hsv_colors[:, 0] = variable
    hsv_colors[:, 1] = 1.0
    hsv_colors[:, 2] = 1.0
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=rgb_colors, s=20)

        # Remove everything
        ax.set_axis_off()
        ax.set_aspect("equal")
        plt.tight_layout()
        return

    # -------- 3D --------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=rgb_colors,
        s=100
    )

    # Remove absolutely everything
    ax.set_axis_off()
    ax.grid(False)

    # Extra safety: remove panes if backend still draws them
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_alpha(0)

    ax.view_init(elev=90, azim=-270)
    plt.tight_layout()

    if figpath:
        plt.savefig(figpath, dpi=300)

    plt.show()
    return embedding


    
def analyze_orientation_periodicity(
    ori_map: torch.Tensor,
    max_scale_mm: float,
    n_bins: int = 50,
    fs: int = 36,
    figpath=None
):
    """
    Analyze hidden spatial periodicity in an orientation map.

    Parameters
    ----------
    ori_map : torch.Tensor
        NxN tensor with values in [0, pi)
    max_scale_mm : float
        Physical size of the map (e.g., 1.0 for 1 mm)
    n_bins : int
        Number of radial bins for averaging
    fs : int
        Font size for plotting

    Returns
    -------
    radii_mm : np.ndarray
        Radial distances (mm)
    autocorr_radial : np.ndarray
        Radially averaged autocorrelation
    freq_centers : np.ndarray
        Radially averaged spatial frequencies (cycles/mm)
    power_radial : np.ndarray
        Radially averaged power spectrum
    """

    # ---------- Safety checks ----------
    assert ori_map.ndim == 2 and ori_map.shape[0] == ori_map.shape[1], \
        "ori_map must be square NxN"
    assert ori_map.min() >= 0 and ori_map.max() <= np.pi + 1e-6, \
        "Orientation values must be in [0, pi)"

    N = ori_map.shape[0]
    dx = max_scale_mm / N  # mm per pixel

    # ---------- Handle circular orientation ----------
    ori_complex = torch.exp(2j * ori_map)
    ori_complex = ori_complex - ori_complex.mean()  # remove mean

    # ---------- 2D autocorrelation ----------
    fft_map = torch.fft.fft2(ori_complex)
    power = fft_map * torch.conj(fft_map)
    autocorr = torch.fft.ifft2(power).real
    autocorr = torch.fft.fftshift(autocorr)
    autocorr /= autocorr.max()

    # ---------- 2D power spectrum ----------
    power_spectrum = torch.abs(fft_map) ** 2
    power_spectrum = torch.fft.fftshift(power_spectrum)

    # ---------- Radial coordinates ----------
    coords = torch.arange(-N//2, N//2)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    rr_mm = rr.numpy() * dx

    # ---------- Radial bins ----------
    r_max = rr_mm.max()
    bins = np.linspace(0, r_max, n_bins + 1)

    def radial_average(field, rr_array):
        field = field.numpy()
        radial_mean = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (rr_array >= bins[i]) & (rr_array < bins[i + 1])
            if np.any(mask):
                radial_mean[i] = field[mask].mean()
        return radial_mean

    # ---------- Compute radially averaged autocorr ----------
    autocorr_radial = radial_average(autocorr, rr_mm)

    # ---------- Compute radially averaged power spectrum using frequency bins ----------
    freqs = np.fft.fftfreq(N, d=dx)  # cycles/mm
    fx, fy = np.meshgrid(freqs, freqs, indexing="ij")
    f_r = np.sqrt(fx**2 + fy**2)

    f_max = f_r.max()
    bins_f = np.linspace(0, f_max, n_bins + 1)
    power_spectrum_np = power_spectrum.numpy()
    power_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (f_r >= bins_f[i]) & (f_r < bins_f[i+1])
        if np.any(mask):
            power_radial[i] = power_spectrum_np[mask].mean()
    freq_centers = 0.5 * (bins_f[:-1] + bins_f[1:])

    # ---------- Axes for autocorr ----------
    radii_mm = 0.5 * (bins[:-1] + bins[1:])

    # ---------- Plot ----------
    plt.figure(figsize=(6.5,5))
    ax = plt.gca()

    # Autocorrelation
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(radii_mm[:N//9], autocorr_radial[:N//9], lw=3, color='black')
    ax.set_xlabel("distance (mm)", fontsize=fs)
    ax.set_ylabel("mean autocorr.", fontsize=fs)
    ax.tick_params(labelsize=fs*0.8)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylim(-0.4, 1.1)

    plt.tight_layout()

    if figpath:
        plt.savefig(figpath)
    
    plt.show()

    return radii_mm, autocorr_radial, freq_centers, power_radial


    
def plot_sparse_tensor(tensor: torch.Tensor, highlight_idx: int, figpath=None):
    """
    Scatter plot of a sparse NxN tensor.
    
    Parameters
    ----------
    tensor : torch.Tensor
        NxN tensor containing only 0s and 1s
    highlight_idx : int
        Linear index in [0, N^2 - 1] to highlight
    """
    assert tensor.dim() == 2 and tensor.shape[0] == tensor.shape[1], "Tensor must be NxN"
    
    N = tensor.shape[0]
    assert 0 <= highlight_idx < N * N, "highlight_idx out of range"

    # Get coordinates of ones
    rows, cols = torch.nonzero(tensor, as_tuple=True)

    # Convert linear index to (row, col)
    hi_row = highlight_idx % N
    hi_col = highlight_idx // N

    fig, ax = plt.subplots(figsize=(10,10))

    # Plot ones as black filled points
    ax.scatter(cols, rows, facecolors=(0, 0, 0, 0.), s=300, edgecolor='black', linewidth=2)

    # Highlight selected unit with larger grey circle
    #ax.scatter(hi_col, hi_row, c='grey', s=400, edgecolors='black')

    # Formatting
    ax.set_aspect('equal')
    ax.invert_yaxis()  # matrix-style orientation
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove top/right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if figpath:
        plt.savefig(figpath, dpi=300)

    plt.show()


def analyze_patchiness(connection_fields: torch.Tensor,
                       pixel_size_mm: float = 0.022,
                       mask_radius_mm: float = 0.5,
                       figpath=None,
                       max_freq_cyc_per_mm: float = 10.0,
                       band: str = "sem",          # "std" or "sem"
                       alpha: float = 0.25,
                       linewidth: float = 4.0):
    """
    Plot mean normalized Fourier amplitude vs spatial frequency (cycles/mm)
    with a soft grey uncertainty band (mean ± std or mean ± SEM).

    Notes:
    - Uses signed fft frequencies but plots only the non-negative half (freq >= 0),
      which avoids the common plotting artifact from abs(freq).
    - Uncertainty convention in many papers: mean ± SEM (default), or mean ± SD.

    Parameters
    ----------
    connection_fields : torch.Tensor
        Shape (B, N, N) or (B, 1, N, N).
    pixel_size_mm : float
        Physical size of one pixel in mm.
    mask_radius_mm : float
        (Currently unused in your code; kept for API compatibility.)
    figpath : str or None
        If provided, save figure here.
    max_freq_cyc_per_mm : float
        Max spatial frequency to show.
    band : {"sem","std"}
        Uncertainty band type.
    alpha : float
        Transparency of uncertainty band.
    linewidth : float
        Width of mean curve.
    """
    # Handle input shape
    if connection_fields.dim() == 4:
        connection_fields = connection_fields.squeeze(1)  # (B, N, N)
    if connection_fields.dim() != 3:
        raise ValueError("Expected (B, N, N) or (B, 1, N, N).")

    B, N, _ = connection_fields.shape
    device = connection_fields.device
    center_idx = N // 2

    # --- Frequency axis: signed, shifted; then plot only freq >= 0 ---
    freqs = torch.fft.fftshift(torch.fft.fftfreq(N, d=pixel_size_mm)).cpu().numpy()  # (N,)
    keep = (freqs >= 0) & (freqs <= max_freq_cyc_per_mm)

    # --- Your Gaussian mask (as in your code) ---
    # Assumes get_gaussian returns tensor shaped like (1,1,N,N) or similar; you used [0,0]
    gmask = get_gaussian(N, N/2)[0, 0].to(device)
    gmask = gmask / (gmask.max() + 1e-12)

    profiles = []

    for i in range(B):
        field = connection_fields[i].float()  # (N, N)
        field_masked = field * gmask

        if field_masked.sum().item() < 10:
            continue

        # Principal axis via PCA
        coords = torch.nonzero(field_masked, as_tuple=False).float()  # (K, 2): row, col
        if coords.shape[0] < 2:
            angle_deg = 0.0
        else:
            coords_centered = coords - torch.tensor([center_idx, center_idx], device=device).float()
            cov = torch.cov(coords_centered.T)
            eigvecs = torch.linalg.eigh(cov).eigenvectors
            principal_dir = eigvecs[:, -1]

            # coords are (row=y, col=x). atan2(y, x) => atan2(principal_dir[0], principal_dir[1])
            angle_rad = torch.atan2(principal_dir[0], principal_dir[1])
            angle_deg = (-angle_rad * 180.0 / np.pi).item()

        # Rotate to align principal axis horizontally
        field_rot = rotate(
            field_masked.unsqueeze(0).unsqueeze(0),  # (1,1,N,N)
            angle=angle_deg,
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            fill=0
        ).squeeze(0).squeeze(0)  # (N, N)

        # 2D FFT
        fft = torch.fft.fftshift(torch.fft.fft2(field_rot))
        amplitude = torch.abs(fft)

        dc = amplitude[center_idx, center_idx]
        if dc == 0:
            continue

        amplitude_normalized = amplitude / dc
        horiz_profile = amplitude_normalized[center_idx, :].detach().cpu().numpy()  # (N,)
        profiles.append(horiz_profile)

    if len(profiles) == 0:
        print("No valid connection fields after processing.")
        return

    profiles = np.stack(profiles, axis=0)  # (M, N)
    mean_profile = profiles.mean(axis=0)

    # Uncertainty band
    std_profile = profiles.std(axis=0, ddof=1) if profiles.shape[0] > 1 else np.zeros_like(mean_profile)
    if band.lower() == "sem":
        denom = np.sqrt(profiles.shape[0]) if profiles.shape[0] > 0 else 1.0
        spread = std_profile / denom
        band_label = "Mean ± SEM"
    elif band.lower() == "std":
        spread = std_profile
        band_label = "Mean ± SD"
    else:
        raise ValueError("band must be 'sem' or 'std'.")

    # Plot
    plt.figure(figsize=(8, 6))

    x = freqs[keep]
    y = mean_profile[keep]
    s = spread[keep]

    # Grey uncertainty band (convention)
    plt.fill_between(x, y - s, y + s, alpha=alpha, label=band_label)

    # Mean curve
    plt.plot(x, y, color="black", linewidth=linewidth, label="Mean")

    fs = 36
    plt.xlabel("spatial freq. (cycles/mm)", fontsize=fs)
    plt.ylabel("mean Fourier coef.", fontsize=fs)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)
    plt.tight_layout()
    plt.gca().spines[["top", "right"]].set_visible(False)

    # Optional: legend (often kept small / off in figure panels)
    # plt.legend(frameon=False, fontsize=fs * 0.6)

    if figpath:
        plt.savefig(figpath, dpi=300, bbox_inches="tight")

    plt.show()



def fit_and_plot(
    x, y,
    num_fit_points=100,
    font_size=25,
    axis_labels=("X-axis", "Y-axis"),
    legend_labels=None,
    color=None,
    integer_axes=(False, False),
    x_ticks=None,
    y_ticks=None,
    global_fit=False,
    fit_func="exp",            # NEW: "exp", "one_minus_exp", "log"
    b_grid=None,               # only used for one_minus_exp
    mask_eps=1e-8,             # mask threshold for exp-log fit
    xlim=None,
    ylim=None,
    fit_flag=True,
    scatter=True,
    log_x0=None,               # optional: fixed x0 for log fit; else chosen automatically
    log_delta=1e-3,            # how far below min(x) to place x0 when auto
    marker_mode=False,
    s=100,
    cmap='inferno',
    figpath=None
):
    """
    Fits one of:
      - fit_func="exp":          y = a * exp(-b x)
      - fit_func="one_minus_exp":y = c + a*(1 - exp(-b x))
      - fit_func="log":          y = c + a*log(x - x0)  (x0 chosen so x-x0>0)

    global_fit:
      - False: fit per curve
      - True : fit once across all points, broadcast params to all curves

    Returns:
      - exp: (a, b)
      - one_minus_exp: (c0, a, b)
      - log: (c0, a, x0)
    """

    x = x.float()
    y = y.float()
    c_curves, p = y.shape

    fit_func = str(fit_func).lower().strip()
    if fit_func not in {"exp", "one_minus_exp", "log"}:
        raise ValueError(f"fit_func must be one of: 'exp', 'one_minus_exp', 'log'. Got: {fit_func}")

    # params to return (some unused depending on fit_func)
    a = torch.zeros(c_curves, dtype=torch.float32)
    b = torch.zeros(c_curves, dtype=torch.float32)
    c0 = torch.zeros(c_curves, dtype=torch.float32)
    x0 = torch.zeros(c_curves, dtype=torch.float32)  # used for log fit

    colors = sns.color_palette(cmap, n_colors=c_curves)
    marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]

    plt.figure(figsize=(7, 6))

    if legend_labels is None:
        legend_labels = [''] * c_curves

    def _get_xi(i):
        return x if x.dim() == 1 else x[i]

    def _fit_exp(xi, yi):
        """
        Fit y = a * exp(-b x), allowing sign flips via sign(a) and growth via sign(b).
        Uses log(|y|) with a near-zero mask.
        """
        m = torch.abs(yi) > mask_eps
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            a_i = yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device)
            b_i = torch.tensor(0.0, device=yi.device)
            return a_i, b_i

        yi_abs = torch.abs(yi_m).clamp_min(1e-12)

        x_min = xi_m.min()
        x_max = xi_m.max()
        delta_x = (x_max - x_min).clamp_min(1e-12)
        x_norm = (xi_m - x_min) / delta_x

        Y = torch.log(yi_abs).unsqueeze(1)
        X = torch.cat([torch.ones_like(x_norm).unsqueeze(1), -x_norm.unsqueeze(1)], dim=1)

        sol = torch.linalg.lstsq(X, Y).solution.squeeze()
        log_a_norm, b_norm = sol[0], sol[1]
        a_norm = torch.exp(log_a_norm)

        b_i = b_norm / delta_x
        a_pos = a_norm * torch.exp(b_norm * x_min / delta_x)

        s = torch.sign(yi_m.mean())
        if s == 0:
            s = torch.tensor(1.0, device=yi.device)
        a_i = s * a_pos
        return a_i, b_i

    def _fit_one_minus_exp(xi, yi):
        """
        Fit y = c + a*(1 - exp(-b x)) by grid-searching b and solving (c,a) via LS.
        """
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            return (yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device))

        if b_grid is None:
            xr = (xi_m.max() - xi_m.min()).clamp_min(1e-12)
            b_vals = torch.logspace(-3, 3, steps=121, device=yi.device) / xr
        else:
            b_vals = torch.as_tensor(b_grid, dtype=torch.float32, device=yi.device)

        ones = torch.ones_like(xi_m)
        best_sse = None
        best_c, best_a, best_b = None, None, None

        for bi in b_vals:
            phi = 1.0 - torch.exp(-bi * xi_m)
            A = torch.stack([ones, phi], dim=1)

            sol = torch.linalg.lstsq(A, yi_m.unsqueeze(1)).solution.squeeze()
            ci, ai = sol[0], sol[1]

            y_hat = ci + ai * phi
            sse = torch.sum((y_hat - yi_m) ** 2)

            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_c, best_a, best_b = ci, ai, bi

        return best_c, best_a, best_b

    def _fit_log(xi, yi):
        """
        Fit y = c + a*log(x - x0), with x0 chosen so x-x0 > 0 for all used points.
        This is a simple linear LS in [1, log(x-x0)].
        """
        m = torch.isfinite(xi) & torch.isfinite(yi)
        xi_m = xi[m]
        yi_m = yi[m]
        if xi_m.numel() < 2:
            return (yi_m.mean() if yi_m.numel() > 0 else torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device),
                    torch.tensor(0.0, device=yi.device))

        # choose x0
        if log_x0 is None:
            # place x0 a bit below min(x) so (x-x0) is safely positive
            xmin = xi_m.min()
            span = (xi_m.max() - xmin).clamp_min(1e-12)
            x0_i = xmin - float(log_delta) * span - 1e-12
        else:
            x0_i = torch.tensor(float(log_x0), device=yi.device)

        z = (xi_m - x0_i).clamp_min(1e-12)
        phi = torch.log(z)

        A = torch.stack([torch.ones_like(phi), phi], dim=1)  # columns: c, a
        sol = torch.linalg.lstsq(A, yi_m.unsqueeze(1)).solution.squeeze()
        ci, ai = sol[0], sol[1]
        return ci, ai, x0_i

    # ---- fitting: per-curve or global ----
    if fit_flag:
        if global_fit:
            xs, ys = [], []
            for i in range(c_curves):
                xs.append(_get_xi(i).reshape(-1))
                ys.append(y[i].reshape(-1))
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)

            if fit_func == "exp":
                a_s, b_s = _fit_exp(x_all, y_all)
                a[:] = a_s
                b[:] = b_s
            elif fit_func == "one_minus_exp":
                c_s, a_s, b_s = _fit_one_minus_exp(x_all, y_all)
                c0[:] = c_s
                a[:] = a_s
                b[:] = b_s
            elif fit_func == "log":
                c_s, a_s, x0_s = _fit_log(x_all, y_all)
                c0[:] = c_s
                a[:] = a_s
                x0[:] = x0_s

        else:
            for i in range(c_curves):
                xi = _get_xi(i)
                yi = y[i]

                if fit_func == "exp":
                    ai, bi = _fit_exp(xi, yi)
                    a[i], b[i] = ai, bi
                elif fit_func == "one_minus_exp":
                    ci, ai, bi = _fit_one_minus_exp(xi, yi)
                    c0[i], a[i], b[i] = ci, ai, bi
                elif fit_func == "log":
                    ci, ai, x0i = _fit_log(xi, yi)
                    c0[i], a[i], x0[i] = ci, ai, x0i

    # ---- plotting ----
    for i in range(c_curves):
        xi = _get_xi(i)
        yi = y[i]

        marker_i = 'o'
        if marker_mode:
            marker_i = marker_list[i % len(marker_list)]

        color_i = colors[i]
        
        if scatter:
            plt.scatter(
                xi.detach().cpu().numpy(),
                yi.detach().cpu().numpy(),
                color=color_i if not color else color,
                marker=marker_i,
                s=s,
                linewidths=2 if marker_mode else 0,
                label=legend_labels[i] if global_fit else None,
                edgecolor='black'
            )
            
        else:
            plt.plot(
                xi.detach().cpu().numpy(),
                yi.detach().cpu().numpy(),
                color=color_i if not color else color,
                marker=marker_i if marker_mode else None,
                linewidth=2
            )

        if global_fit:
            x_fit = torch.linspace(x.min(), x.max(), num_fit_points, device=xi.device)
        else:
            x_fit = torch.linspace(xi.min(), xi.max(), num_fit_points, device=xi.device)

        if fit_func == "exp":
            y_fit = a[i] * torch.exp(-b[i] * x_fit)
        elif fit_func == "one_minus_exp":
            y_fit = c0[i] + a[i] * (1.0 - torch.exp(-b[i] * x_fit))
        elif fit_func == "log":
            z = (x_fit - x0[i]).clamp_min(1e-12)
            y_fit = c0[i] + a[i] * torch.log(z)

        if fit_flag and (not global_fit or i==0):

            plt.plot(
                x_fit.detach().cpu().numpy(),
                y_fit.detach().cpu().numpy(),
                color=colors[i] if not global_fit else 'black',
                linewidth=2,
                label=legend_labels[i] if not global_fit else None
            )

    # ---- axes/ticks (your original) ----
    plt.xlabel(axis_labels[0], fontsize=font_size)
    plt.ylabel(axis_labels[1], fontsize=font_size)
    plt.xticks(fontsize=font_size * 0.8)
    plt.yticks(fontsize=font_size * 0.8)

    if x_ticks is None:
        x_ticks_vals = torch.linspace(x.min(), x.max(), 4)
        if integer_axes[0]:
            x_ticks_vals = torch.round(x_ticks_vals)
        x_labels = [str(int(val)) if integer_axes[0] else f"{val:.2g}"
                    for val in x_ticks_vals.detach().cpu().numpy()]
    else:
        x_ticks_vals = torch.tensor(x_ticks, dtype=torch.float32)
        x_labels = [str(int(val)) if float(val).is_integer() else f"{float(val):.2g}" for val in x_ticks_vals]

    if y_ticks is None:
        y_ticks_vals = torch.linspace(y.min(), y.max(), 4)
        if integer_axes[1]:
            y_ticks_vals = torch.round(y_ticks_vals)
        y_labels = [str(int(val)) if integer_axes[1] else f"{val:.2g}"
                    for val in y_ticks_vals.detach().cpu().numpy()]
    else:
        y_ticks_vals = torch.tensor(y_ticks, dtype=torch.float32)
        y_labels = [str(int(val)) if float(val).is_integer() else f"{float(val):.2g}" for val in y_ticks_vals]

    plt.xticks(x_ticks_vals.detach().cpu().numpy(), x_labels)
    plt.yticks(y_ticks_vals.detach().cpu().numpy(), y_labels)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend_labels[0] != '':
        plt.legend(frameon=False, fontsize=font_size * 0.8)

    plt.tight_layout()

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if figpath:
        plt.savefig(figpath)

    plt.show()

    # ---- return params ----
    if fit_func == "exp":
        return a, b
    if fit_func == "one_minus_exp":
        return c0, a, b
    if fit_func == "log":
        return c0, a, x0


def _make_grating_bank_onoff(
    W: int,
    thetas_rad: torch.Tensor,      # [K]
    freqs: torch.Tensor,           # [F] cycles per image (or per W pixels, see notes)
    phases_rad: torch.Tensor,      # [P]
    device=None,
    dtype=torch.float32,
    centered: bool = True,
) -> torch.Tensor:
    """
    Returns stimuli shaped [K, F, P, 2, W, W] for ON/OFF channels:
      L(x,y) = cos(2π f (x cosθ + y sinθ) + phase)
      ON = relu(L), OFF = relu(-L)
    """
    device = device or thetas_rad.device
    thetas_rad = thetas_rad.to(device=device, dtype=dtype)
    freqs = freqs.to(device=device, dtype=dtype)
    phases_rad = phases_rad.to(device=device, dtype=dtype)

    # grid in [-0.5, 0.5] (roughly) or [0,1)
    if centered:
        coords = torch.linspace(-(W - 1) / 2, (W - 1) / 2, W, device=device, dtype=dtype)
        coords = coords / max(W, 1)  # normalize to ~[-0.5,0.5]
    else:
        coords = torch.linspace(0, 1, W, device=device, dtype=dtype)

    yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # [W,W]

    # Expand for broadcasting
    # theta: [K,1,1,1,1]
    ct = torch.cos(thetas_rad)[:, None, None, None, None]
    st = torch.sin(thetas_rad)[:, None, None, None, None]

    # freqs: [1,F,1,1,1]
    fr = freqs[None, :, None, None, None]

    # phases: [1,1,P,1,1]
    ph = phases_rad[None, None, :, None, None]

    # project coordinate along orientation
    # proj: [K,1,1,W,W]
    proj = (xx[None, None, None, :, :] * ct) + (yy[None, None, None, :, :] * st)

    # L: [K,F,P,W,W]
    L = torch.cos(2 * math.pi * fr * proj + ph)

    on = F.relu(L)
    off = F.relu(-L)

    stim = torch.stack([on, off], dim=3)  # [K,F,P,2,W,W]
    return stim


def detect_orientation_map_from_aff_weights(
    aff: torch.Tensor,                 # [N, 2, W, W]  (channel 0=ON, 1=OFF)
    *,
    num_orientations: int = 18,         # e.g. 18 => 0..170° in 10° steps
    freqs: Optional[torch.Tensor] = None,
    num_phases: int = 8,
    rectify_output: bool = False,       # optional cortical half-wave rectification
    batch_size: int = 2048,
    return_degrees: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Approximates the paper's map readout (vector-average from grating responses),
    using the afferent ON/OFF->V1 weights as the "receptive field".

    Steps:
      1) Build ON/OFF grating bank with K orientations, F spatial freqs, P phases.
      2) For each neuron, compute linear response = sum(w * stim).
         Optionally rectify.
      3) For each orientation, take max over phase and frequency -> r(theta).
      4) Vector average with 2*theta: v = Σ r(theta) * exp(i*2*theta)
         pref = 0.5 * arg(v), selectivity = |v|, osi = |v| / (Σ r + eps)

    Returns dict with:
      - pref: [N] preferred orientation (deg in [0,180) unless return_degrees=False)
      - selectivity: [N] magnitude of vector sum (raw)
      - osi: [N] normalized selectivity in [0,1] (common)
      - r_theta: [N,K] orientation response curve after max over phase/freq
    """
    if aff.ndim != 4 or aff.shape[1] != 2 or aff.shape[2] != aff.shape[3]:
        raise ValueError(f"Expected aff shape [N,2,W,W], got {tuple(aff.shape)}")

    N, _, W, _ = aff.shape
    device = aff.device
    dtype = aff.dtype

    # orientations in [0, pi)
    K = int(num_orientations)
    thetas = torch.linspace(0, math.pi, K + 1, device=device, dtype=dtype)[:-1]  # [K]

    # spatial freqs (cycles per normalized image width ~W). Tune as needed.
    if freqs is None:
        # A small set tends to work well. You can pass your own.
        freqs = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)  # [F]
    else:
        freqs = freqs.to(device=device, dtype=dtype)

    P = int(num_phases)
    phases = torch.linspace(0, 2 * math.pi, P + 1, device=device, dtype=dtype)[:-1]  # [P]

    stim = _make_grating_bank_onoff(
        W=W,
        thetas_rad=thetas,
        freqs=freqs,
        phases_rad=phases,
        device=device,
        dtype=dtype,
        centered=True,
    )  # [K,F,P,2,W,W]

    # Flatten stim spatial+channel for fast matmul
    stim_flat = stim.reshape(K * freqs.numel() * P, 2 * W * W)  # [K*F*P, 2WW]
    stim_flat_t = stim_flat.t().contiguous()  # [2WW, K*F*P]

    r_theta_all = torch.empty((N, K), device=device, dtype=dtype)

    # Batch neurons to control memory
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        w = aff[start:end].reshape(end - start, 2 * W * W)  # [B,2WW]

        # Linear responses to all gratings: [B, K*F*P]
        resp = w @ stim_flat_t
        if rectify_output:
            resp = F.relu(resp)

        # Reshape to [B,K,F,P]
        resp = resp.view(end - start, K, freqs.numel(), P)

        # Max over phase then frequency -> [B,K]
        r_theta = resp.max(dim=3).values.max(dim=2).values
        r_theta_all[start:end] = r_theta

    # Vector average using 2*theta
    # v = Σ r(theta) * exp(i*2θ)
    exp_i2t = torch.exp(1j * (2.0 * thetas).to(torch.complex64))  # [K] complex
    r_complex = r_theta_all.to(torch.complex64)  # [N,K]
    v = (r_complex * exp_i2t[None, :]).sum(dim=1)  # [N] complex

    pref_rad = 0.5 * torch.atan2(v.imag, v.real)  # [-pi/2, pi/2]
    # map to [0, pi)
    pref_rad = torch.remainder(pref_rad, math.pi)

    selectivity = torch.abs(v).to(dtype)  # [N]
    denom = r_theta_all.sum(dim=1).clamp_min(1e-8)
    osi = (selectivity / denom).clamp(0, 1)

    if return_degrees:
        pref = pref_rad * (180.0 / math.pi)
    else:
        pref = pref_rad

    return {
        "pref": pref,                 # [N]
        "selectivity": selectivity,   # [N] raw
        "osi": osi,                   # [N] normalized
        "r_theta": r_theta_all,       # [N,K]
        "thetas": (thetas * (180.0 / math.pi) if return_degrees else thetas),  # [K]
    }