import os
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_training(log_dir):
    """Monitor training progress"""
    log_file = os.path.join(log_dir, 'training_log.json')
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        train_losses = data.get('train_loss', [])
        val_losses = data.get('val_loss', [])
        
        if train_losses and val_losses:
            print(f"Training Progress:")
            print(f"Epochs completed: {len(train_losses)}")
            print(f"Latest Train Loss: {train_losses[-1]:.4f}")
            print(f"Latest Val Loss: {val_losses[-1]:.4f}")
            
            if len(train_losses) > 1:
                print(f"Best Val Loss: {min(val_losses):.4f}")
                print(f"Improvement: {val_losses[0] - val_losses[-1]:.4f}")
            
            # Plot training curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, 'training_progress.png'))
            plt.close()
            
            return True
    return False

def main():
    log_dir = 'logs'
    
    print("Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            if monitor_training(log_dir):
                print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("No training log found yet...")
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == '__main__':
    main() 