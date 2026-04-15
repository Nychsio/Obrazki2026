import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OpenFakeDataset(Dataset):
    """
    Custom Dataset for loading the ComplexDataLab/OpenFake dataset.
    Loads full dataset into memory (no streaming).
    """
    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.dataset_id = "ComplexDataLab/OpenFake"
        
        # Load the dataset without streaming (full dataset in memory)
        print(f"Loading OpenFake dataset ({split})...")
        self.hf_dataset = load_dataset(self.dataset_id, split=self.split)
        print(f"Dataset loaded: {len(self.hf_dataset)} samples")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        # Detailed safety check on image key
        if "image" not in sample:
            # Return a dummy sample if image key is missing
            dummy_image = torch.zeros(3, 224, 224)
            dummy_label = -1
            return dummy_image, dummy_label
            
        image = sample["image"]
        # Convert PIL Image to RGB NumPy array
        image_np = np.array(image.convert("RGB"))
        
        # Extract label, assume 'label' key exists
        label = sample.get("label", -1)

        if self.transform:
            # Apply Albumentations transform
            # Albumentations expects 'image' keyword argument
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            # Fallback: simple conversion if no transform provided
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, label

def get_transforms():
    """
    Returns the albumentations transform pipeline including strong augmentations.
    """
    return A.Compose([
        # Strong augmentations
        A.ImageCompression(quality_range=(60, 100), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Standard blur
        # var_limit=(10, 50) -> std ~= (sqrt(10), sqrt(50)) / 255
        A.GaussNoise(std_range=(0.0124, 0.0277), p=0.5), # Add noise
        
        # Essential specific processing
        A.Resize(224, 224), # Resize to standard efficientnet size (implied requirement usually)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

if __name__ == '__main__':
    # Initial test
    print("Initializing dataset...")
    transforms = get_transforms()
    dataset = OpenFakeDataset(split="train", transform=transforms)
    
    # Create DataLoader with multiple workers to test sharding
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)
    
    print("Testing data loading with num_workers=2...")
    try:
        for i, (images, labels) in enumerate(dataloader):
            print(f"Batch {i}: Images {images.shape}, Labels {labels}")
            if i >= 2:
                break
        print("Success! Data loaded correctly.")
    except Exception as e:
        print(f"Error during data loading: {e}")
