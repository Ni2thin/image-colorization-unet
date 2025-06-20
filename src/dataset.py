import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=128):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Convert to grayscale for input
        gray_image = image.convert('L')
        
        # Apply transforms
        if self.transform:
            color_image = self.transform(image)
            gray_image = self.transform(gray_image)
        
        return gray_image, color_image

def create_dataloaders(train_dir, val_dir, batch_size=16, img_size=128):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = ColorizationDataset(train_dir, transform=transform, img_size=img_size)
    val_dataset = ColorizationDataset(val_dir, transform=transform, img_size=img_size)
    
    # Create dataloaders with fewer workers
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def convert_to_grayscale(image):
    """Convert a color image to grayscale using OpenCV"""
    if isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif isinstance(image, Image.Image):
        return image.convert('L')
    else:
        raise TypeError("Input must be either numpy array or PIL Image")

def normalize_image(image):
    """Normalize image values to [0, 1] range"""
    if isinstance(image, np.ndarray):
        return image.astype(np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        return image.float() / 255.0
    else:
        raise TypeError("Input must be either numpy array or torch Tensor")
