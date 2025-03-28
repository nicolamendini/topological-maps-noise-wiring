import os
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from skimage.io import imread

def sample_and_plot_histogram_and_images(target_folder):
    # Find all image paths
    image_paths = glob.glob(os.path.join(target_folder, '*.*'))
    
    # Ensure there are enough images
    if len(image_paths) < 100:
        print("Not enough images in the folder. Found only", len(image_paths), "images.")
        return
    
    # Randomly sample 100 image paths for histograms
    sampled_paths = random.sample(image_paths, 100)
    
    # Initialize arrays to accumulate histograms for ON and OFF channels
    combined_histogram_on = np.zeros(256)
    combined_histogram_off = np.zeros(256)
    
    # Compute and accumulate histograms for ON and OFF channels
    for path in sampled_paths:
        img = imread(path)
        if img.shape[2] > 2:  # Check if the image has multiple channels
            on_channel = img[:, :, 0]  # ON channel
            off_channel = img[:, :, 1]  # OFF channel
		    
            # Calculate histogram for nonzero values only for ON channel
            histogram_on, _ = np.histogram(on_channel[on_channel > 0], bins=np.arange(257))
            combined_histogram_on += histogram_on
            
            # Calculate histogram for nonzero values only for OFF channel
            histogram_off, _ = np.histogram(off_channel[off_channel > 0], bins=np.arange(257))
            combined_histogram_off += histogram_off
    
    # Randomly pick one image for displaying ON and OFF channel images
    random_image_path = random.choice(image_paths)
    random_img = imread(random_image_path)
    on_image = random_img[:, :, 0]
    off_image = random_img[:, :, 1]

    # Plotting histograms and images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # ON channel histogram
    axes[0, 0].bar(np.arange(256), combined_histogram_on, color='blue')
    axes[0, 0].set_title('Histogram of Nonzero Pixel Values (ON Channel)')
    axes[0, 0].set_xlabel('Pixel Intensity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_xlim([0, 255])
    
    # OFF channel histogram
    axes[0, 1].bar(np.arange(256), combined_histogram_off, color='red')
    axes[0, 1].set_title('Histogram of Nonzero Pixel Values (OFF Channel)')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim([0, 255])

    # Display random ON channel image
    axes[1, 0].imshow(on_image, cmap='gray')
    axes[1, 0].set_title('Random Sample Image (ON Channel)')
    axes[1, 0].axis('off')  # Turn off axis labels

    # Display random OFF channel image
    axes[1, 1].imshow(off_image, cmap='gray')
    axes[1, 1].set_title('Random Sample Image (OFF Channel)')
    axes[1, 1].axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

# Example usage
target_folder = './input_stimuli'  # Change this to your target folder path
sample_and_plot_histogram_and_images(target_folder)

