import torch
import timm
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from captum.attr import LayerGradCam

def create_feature_extractor():
    """
    Creates an EfficientNet-B0 model for feature extraction.
    Removes the classification head by setting num_classes=0.
    """
    # Load pretrained efficientnet_b0 model
    # num_classes=0 removes the final classification layer
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    model.eval()
    return model

def explain_image(image_tensor, model, original_image, output_path="gradcam_output.jpg"):
    """
    Generates a GradCAM attribution for the input image and saves the result.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (1, 3, 224, 224).
        model (torch.nn.Module): The feature extractor model.
        original_image (numpy.ndarray): The original image (H, W, 3) in BGR format (for OpenCV).
        output_path (str): Path to save the combined image.
    """
    
    # 1. Setup LayerGradCam on the final convolutional layer
    # For efficientnet_b0 in timm, the final conv layer before pooling is often 'conv_head'
    # or 'bn2' depending on exact architecture, but 'conv_head' is standard for efficientnets.
    layer_gc = LayerGradCam(model, model.conv_head)
    
    # 2. Determine target for attribution
    # Since we have no classification head (output is feature vector),
    # we target the feature with the highest activation.
    with torch.no_grad():
        output = model(image_tensor)
        target_index = output.argmax(dim=1).item()
        
    print(f"Targeting feature index: {target_index}")

    # 3. Compute attribution
    # attribute() computes gradients with respect to the target
    # The output is spatial attribution at the resolution of the target layer (e.g., 7x7)
    attr = layer_gc.attribute(image_tensor, target=target_index)
    
    # 4. Upsample attribution to match input image size (224x224)
    # LayerGradCam returns shape (1, 1, H_layer, W_layer)
    attr = torch.nn.functional.interpolate(attr, size=(224, 224), mode='bilinear', align_corners=False)
    
    # 5. Process attribution for visualization
    # Convert to numpy, remove batch/channel dims -> (224, 224)
    attr_np = attr.squeeze().detach().cpu().numpy()
    
    # ReLU equivalent (only positive relevance) - GradCAM does this essentially, but ensure >= 0
    attr_np = np.maximum(attr_np, 0)
    
    # Normalize to 0-1 range
    if attr_np.max() > 0:
        attr_np = attr_np / attr_np.max()
        
    # 6. Create Heatmap
    # Convert to uint8 0-255
    heatmap = np.uint8(255 * attr_np)
    # Apply colormap (COLORMAP_JET is common for heatmaps)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 7. Overlay on original image
    # Ensure original image is resized to 224x224 if it isn't already (it should be roughly)
    if original_image.shape[:2] != (224, 224):
         original_image = cv2.resize(original_image, (224, 224))
         
    # Weighted sum: 0.5 * Original + 0.5 * Heatmap
    combined_image = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    
    # 8. Save
    cv2.imwrite(output_path, combined_image)
    print(f"Saved GradCAM visualization to {output_path}")

if __name__ == '__main__':
    # 1. Download a sample image
    # Using a stable URL for a sample image (e.g. from COCO or similar)
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Example cat image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        print("Downloaded sample image.")
    except Exception as e:
        print(f"Failed to download image: {e}")
        # Create a dummy image if download fails
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print("Created dummy image instead.")

    # 2. Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0) # Add batch dimension -> (1, 3, 224, 224)
    
    # Prepare original image for OpenCV (BGR format)
    # We resize/crop manually to match what the tensor sees for visualization alignment
    img_resized = transforms.CenterCrop(224)(transforms.Resize(256)(img))
    original_image_np = np.array(img_resized)
    original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)

    # 3. Create Model
    print("Creating model...")
    model = create_feature_extractor()

    # 4. Run Explainability
    print("Running explainability...")
    explain_image(input_tensor, model, original_image_np, output_path="gradcam_output.jpg")

    print("Done.")
