import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

# Ensure the project root is in sys.path to allow imports
# This allows running the script from anywhere, e.g., 'python src/rgb/inference.py'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try imports
try:
    from src.models.rgb.train import RGBClassifier
except ImportError:
    # Fallback for when running as a script where src is not a package
    # This might happen if project structure is slightly different or sys.path isn't working as expected
    print("Error: Could not import RGBClassifier. Ensure you are running from the project root using 'python -m src.models.rgb.inference' or that PYTHONPATH is set.")
    sys.exit(1)

def get_inference_transforms():
    """
    Returns the validation/inference transform pipeline using Albumentations.
    Matches the normalization and resizing used in training data.
    """
    return A.Compose([
        # Resize to standard efficientnet size (224x224)
        A.Resize(224, 224),
        
        # Normalize with ImageNet stats (Same as in src/rgb/data.py)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Convert to PyTorch Tensor
        ToTensorV2(),
    ])

def preprocess_image(image_path, transform):
    """
    Loads and preprocesses an image from disk.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not decode image at {image_path}")
        
    # Convert BGR to RGB (OpenCV loads as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    augmented = transform(image=image)
    image_tensor = augmented['image']
    
    # Add batch dimension (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict(model, image_tensor, device):
    """
    Runs inference on the image tensor.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        logits = model(image_tensor)
        
        # Apply sigmoid to get probability
        prob = torch.sigmoid(logits).item()
        
    return prob

def main():
    parser = argparse.ArgumentParser(description="Inference script for RGB Deepfake Classifier")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("--model_path", type=str, default="best_rgb_model.pt", help="Path to the trained model checkpoint")
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    try:
        model = RGBClassifier()
        model.to(device)
        
        # Load weights
        if not os.path.exists(args.model_path):
             raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
             
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Handle state dict structure (saved as 'model_state_dict' key in train.py)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare logic
    transform = get_inference_transforms()
    
    try:
        # Preprocess
        input_tensor = preprocess_image(args.image_path, transform)
        
        # Predict
        probability = predict(model, input_tensor, device)
        
        # Interpret result
        # Assuming label 0 = 'real', 1 = 'fake' (as per train.py mapping)
        predicted_class = "Deepfake" if probability > 0.5 else "Real"
        confidence = probability if probability > 0.5 else 1 - probability
        
        print("-" * 30)
        print(f"Image: {args.image_path}")
        print(f"Prediction: {predicted_class}")
        print(f"Probability (Fake): {probability:.4f}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
