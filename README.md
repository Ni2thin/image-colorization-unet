# Image Colorization using U-Net Autoencoder

A deep learning project that colorizes black and white images using a U-Net autoencoder architecture. This project achieves excellent performance with **MSE: 0.006378** and **SSIM: 0.917006**.

##  Project Overview

This project implements an end-to-end solution for automatic image colorization using a U-Net autoencoder. The model takes grayscale images as input and generates realistic colorized versions while preserving structural details.

##  Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **MSE** | 0.006378 | Excellent |
| **SSIM** | 0.917006 | Excellent |

##  Architecture

The model uses a U-Net architecture with:
- **Encoder**: 4 levels of downsampling with convolutional layers
- **Decoder**: 4 levels of upsampling with transposed convolutions
- **Skip Connections**: Between corresponding encoder and decoder levels
- **Custom Loss**: Combines MSE and SSIM for optimal results

### Loss Function
```python
Loss = α × MSE + (1-α) × SSIM
```
Where α = 0.5 balances pixel-wise accuracy and structural similarity.

## Results

The model achieves excellent performance:
- **Low MSE (0.006378)**: Indicates high pixel-wise accuracy
- **High SSIM (0.917006)**: Shows excellent structural preservation
- **Fast Inference**: Real-time colorization capability

---
