import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from model import UNet
from dataset import ColorizationDataset

def load_model(model_path, device):
    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint dictionaries and direct state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def calculate_metrics(pred, target):
    # Convert tensors to numpy arrays
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Calculate MSE
    mse = mean_squared_error(pred_np, target_np)
    
    # Calculate SSIM for each channel
    ssim_values = []
    for i in range(3):  # RGB channels
        ssim_val = ssim(pred_np[i], target_np[i], data_range=1.0)
        ssim_values.append(ssim_val)
    
    # Average SSIM across channels
    avg_ssim = np.mean(ssim_values)
    
    return mse, avg_ssim

def visualize_results(gray_image, color_image, predicted_image, save_path):
    # Convert tensors to numpy arrays
    gray_np = gray_image.squeeze().cpu().numpy()
    color_np = color_image.permute(1, 2, 0).cpu().numpy()
    pred_np = predicted_image.permute(1, 2, 0).cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot grayscale image
    plt.subplot(131)
    plt.imshow(gray_np, cmap='gray')
    plt.title('Input (Grayscale)')
    plt.axis('off')
    
    # Plot ground truth
    plt.subplot(132)
    plt.imshow(color_np)
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Plot predicted image
    plt.subplot(133)
    plt.imshow(pred_np)
    plt.title('Predicted')
    plt.axis('off')
    
    # Save figure
    plt.savefig(save_path)
    plt.close()

def evaluate(model, test_loader, device, output_dir):
    model.eval()
    total_mse = 0
    total_ssim = 0
    num_samples = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (gray_images, color_images) in enumerate(tqdm(test_loader, desc="Evaluating")):
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            
            # Generate predictions
            predicted_images = model(gray_images)
            
            # Calculate metrics
            for pred, target in zip(predicted_images, color_images):
                mse, ssim_val = calculate_metrics(pred, target)
                total_mse += mse
                total_ssim += ssim_val
                num_samples += 1
            
            # Visualize results for first batch
            if i == 0:
                for j in range(min(4, len(gray_images))):
                    save_path = os.path.join(output_dir, f'result_{i}_{j}.png')
                    visualize_results(
                        gray_images[j],
                        color_images[j],
                        predicted_images[j],
                        save_path
                    )
    
    # Calculate average metrics
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    
    return avg_mse, avg_ssim

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create test dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    test_dataset = ColorizationDataset(args.test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate model
    mse, ssim = evaluate(model, test_loader, device, args.output_dir)
    
    print(f"\nEvaluation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"SSIM: {ssim:.6f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"SSIM: {ssim:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the colorization model')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=256, help='Size of input images')
    
    args = parser.parse_args()
    
    main(args)
