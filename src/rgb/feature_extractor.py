import torch
import timm
import numpy as np
import os

def create_feature_extractor():
    """
    Creates an EfficientNet-B0 model for feature extraction.
    Removes the classification head by setting num_classes=0.
    """
    # Load pretrained efficientnet_b0 model
    # num_classes=0 removes the final classification layer, returning the feature vector
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    
    # Set model to evaluation mode
    model.eval()
    return model

def extract_and_save_features(model, input_tensor, output_path):
    """
    Extracts features from an input tensor and saves them as a .npy file.
    
    Args:
        model: The PyTorch model.
        input_tensor: A torch.Tensor of shape (1, 3, 224, 224).
        output_path: Path to save the .npy file.
    """
    # Ensure no gradients are computed
    with torch.no_grad():
        features = model(input_tensor)
        
    # The output features will have shape (1, 1280) for efficientnet_b0 with num_classes=0
    # Flatten strictly to 1D array of 1280 numbers
    features_np = features.cpu().numpy().flatten()
    
    # Save to disk
    np.save(output_path, features_np)
    print(f"Features saved to {output_path} with shape {features_np.shape}")

if __name__ == '__main__':
    # Initialize the model
    print("Initializing model...")
    model = create_feature_extractor()
    
    # Create a dummy random tensor (Batch Size=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Define output path
    output_filename = "dummy_features.npy"
    
    # Extract and save features
    print("Extracting features...")
    extract_and_save_features(model, dummy_input, output_filename)
    
    # Verify the file was created
    if os.path.exists(output_filename):
        print("Success! File generated.")
        # Load and check shape
        loaded_features = np.load(output_filename)
        print(f"Loaded features shape: {loaded_features.shape}")
        
        # Verify strict 1D shape requirement
        if loaded_features.ndim == 1 and loaded_features.shape[0] == 1280:
             print("Shape verification passed: (1280,)")
        else:
             print(f"Shape verification failed! Expected (1280,), got {loaded_features.shape}")

        # Clean up
        try:
            os.remove(output_filename)
            print("Test file removed.")
        except Exception as e:
            print(f"Could not remove test file: {e}")
    else:
        print("Error: File was not generated.")
