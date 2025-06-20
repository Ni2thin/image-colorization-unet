import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(tensor, path):
    """Save a tensor as an image"""
    # Convert tensor to numpy array
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Handle different tensor shapes
    if len(tensor.shape) == 4:  # Batch of images
        tensor = tensor[0]
    if len(tensor.shape) == 3 and tensor.shape[0] == 1:  # Single channel
        tensor = tensor.squeeze(0)
    elif len(tensor.shape) == 3 and tensor.shape[0] == 3:  # RGB
        tensor = tensor.transpose(1, 2, 0)
    
    # Normalize to [0, 1] if needed
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    # Convert to PIL Image and save
    image = Image.fromarray((tensor * 255).astype(np.uint8))
    image.save(path)

def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize_batch(gray_images, color_images, predicted_images, save_path):
    """Visualize a batch of images"""
    # Convert tensors to numpy arrays
    gray_np = gray_images.cpu().numpy()
    color_np = color_images.cpu().numpy()
    pred_np = predicted_images.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(len(gray_images), 3, figsize=(15, 5 * len(gray_images)))
    
    for i in range(len(gray_images)):
        # Plot grayscale image
        axes[i, 0].imshow(gray_np[i, 0], cmap='gray')
        axes[i, 0].set_title('Input (Grayscale)')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(color_np[i].transpose(1, 2, 0))
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot predicted image
        axes[i, 2].imshow(pred_np[i].transpose(1, 2, 0))
        axes[i, 2].set_title('Predicted')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM)"""
    return ssim(img1, img2, data_range=1.0, multichannel=True)

def preprocess_image(image_path, size=256):
    """Preprocess an image for model input"""
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (size, size))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    gray = gray.astype(np.float32) / 255.0
    
    # Convert to tensors
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    gray_tensor = torch.from_numpy(gray).float().unsqueeze(0)
    
    return gray_tensor, image_tensor

def postprocess_image(tensor):
    """Convert model output tensor to image"""
    # Convert to numpy array
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    # Handle different tensor shapes
    if len(tensor.shape) == 4:  # Batch of images
        tensor = tensor[0]
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:  # RGB
        tensor = tensor.transpose(1, 2, 0)
    
    # Normalize to [0, 1] if needed
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    # Convert to uint8
    image = (tensor * 255).astype(np.uint8)
    
    return image
