import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OpenFakeDataset(IterableDataset):
    """
    Custom IterableDataset for streaming the ComplexDataLab/OpenFake dataset.
    Supports multi-worker DataLoaders by sharding the stream.
    """
    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.dataset_id = "ComplexDataLab/OpenFake"

    def __iter__(self):
        # Determine worker information for proper sharding
        worker_info = torch.utils.data.get_worker_info()
        
        # Load the dataset in streaming mode
        # Note: If streaming=True, load_dataset returns an IterableDataset
        hf_dataset = load_dataset(self.dataset_id, split=self.split, streaming=True)
        
        # If running with multiple workers, shard the dataset to avoid duplicates
        if worker_info is not None:
            # Calculate total number of workers across all nodes (usually 1 node, so num_workers)
            # Use strict sharding based on worker ID
            # In streaming mode, this splits the underlying file shards or examples
            hf_dataset = hf_dataset.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id
            )

        for sample in hf_dataset:
            # Detailed safety check on image key
            if "image" not in sample:
                continue
                
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

            yield image_tensor, label

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
