import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse
from model import UNet, ColorizationLoss
from dataset import create_dataloaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for gray_images, color_images in tqdm(train_loader, desc="Training"):
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)
        
        optimizer.zero_grad()
        outputs = model(gray_images)
        loss = criterion(outputs, color_images)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for gray_images, color_images in tqdm(val_loader, desc="Validation"):
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            
            outputs = model(gray_images)
            loss = criterion(outputs, color_images)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main(args):
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(in_channels=1, out_channels=3)
    model = model.to(device)
    
    # Create dataloaders with smaller batch size
    train_loader, val_loader = create_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=16,  # Reduced batch size
        img_size=128    # Reduced image size
    )
    
    # Loss and optimizer
    criterion = ColorizationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save training logs
        log_data = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        with open(os.path.join(args.log_dir, 'training_log.json'), 'w') as f:
            json.dump(log_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the colorization model')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory containing validation images')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_interval', type=int, default=2, help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)
