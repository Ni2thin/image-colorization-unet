import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def download_cifar10(output_dir):
    """Download and prepare CIFAR-10 dataset"""
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])
    
    # Download CIFAR-10
    print("Downloading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Split training set into train and validation
    train_data, val_data = train_test_split(trainset, test_size=0.15, random_state=42)
    
    # Save images
    print("Saving training images...")
    for i, (img, _) in enumerate(tqdm(train_data)):
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(train_dir, f'image_{i:05d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("Saving validation images...")
    for i, (img, _) in enumerate(tqdm(val_data)):
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(val_dir, f'image_{i:05d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("Saving test images...")
    for i, (img, _) in enumerate(tqdm(testset)):
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(test_dir, f'image_{i:05d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("Dataset preparation completed!")

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download and prepare dataset
    download_cifar10(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the colorization dataset')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save processed dataset')
    args = parser.parse_args()
    main(args)
