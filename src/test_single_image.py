import argparse
import torch
from utils import preprocess_image, postprocess_image
from model import UNet
from utils import load_model
import os
from PIL import Image


def colorize_image(model_path, image_path, output_path, device, image_size=256):
    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Preprocess image
    gray_tensor, _ = preprocess_image(image_path, size=image_size)
    gray_tensor = gray_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        pred_tensor = model(gray_tensor)

    # Postprocess output
    color_image = postprocess_image(pred_tensor)

    # Save output
    Image.fromarray(color_image).save(output_path)
    print(f"Colorized image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test colorization model on a single image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to grayscale image')
    parser.add_argument('--output_path', type=str, default='colorized_output.png', help='Path to save colorized image')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for model input')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    colorize_image(args.model_path, args.image_path, args.output_path, device, args.image_size)


if __name__ == '__main__':
    main() 