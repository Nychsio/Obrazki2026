import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import os
from tqdm import tqdm

from src.rgb.data import OpenFakeDataset, get_transforms
from src.rgb.feature_extractor import create_feature_extractor

class RGBClassifier(nn.Module):
    """
    Binary classifier using efficientnet_b0 as backbone.
    """
    def __init__(self):
        super(RGBClassifier, self).__init__()
        # Load feature extractor (EfficientNet-B0 with num_classes=0)
        self.backbone = create_feature_extractor()
        
        # 1280 is the feature dimension of EfficientNet-B0
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 1)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Classify
        logits = self.classifier(features)
        return logits

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, writer, steps_per_epoch=1000):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch} Training")
    
    # Since it's an IterableDataset, we might iterate until StopIteration or a fixed number of steps
    # We use enumerate just to count steps, but relies on dataloader to stop or we break manually
    
    step = 0
    for i, (inputs, labels) in enumerate(dataloader):
        if step >= steps_per_epoch:
            break
            
        inputs = inputs.to(device)
        
        # Convert tuple to tensor if necessary
        if isinstance(labels, (list, tuple)):
            # Map string labels to integers: 'real' -> 0, 'fake' -> 1
            if len(labels) > 0 and isinstance(labels[0], str):
                labels = [0 if l.lower() == 'real' else 1 for l in labels]
            labels = torch.as_tensor(labels)
            
        labels = labels.to(device).float().unsqueeze(1) # BCE needs float and shape (N, 1)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        if step % 10 == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * steps_per_epoch + step)
            
        pbar.update(1)
        step += 1

    pbar.close()
    avg_loss = running_loss / step if step > 0 else 0
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, writer, steps_per_val=200):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    pbar = tqdm(total=steps_per_val, desc=f"Epoch {epoch} Validation")
    
    with torch.no_grad():
        step = 0
        for i, (inputs, labels) in enumerate(dataloader):
            if step >= steps_per_val:
                break
                
            inputs = inputs.to(device)
            
            # Convert tuple to tensor if necessary
            if isinstance(labels, (list, tuple)):
                # Map string labels to integers: 'real' -> 0, 'fake' -> 1
                if len(labels) > 0 and isinstance(labels[0], str):
                    labels = [0 if l.lower() == 'real' else 1 for l in labels]
                labels = torch.as_tensor(labels)
                
            labels = labels.to(device).float().unsqueeze(1)

            # Cast to float32 for metric calculation stability if needed, usually float16 is fine for inference
            # but usually validation is fast enough in full precision too. keeping autocast for consistency.
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            # Use sigmoid to get probabilities for ROC-AUC
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = labels.cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(targets)
            
            pbar.update(1)
            step += 1

    pbar.close()
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate Metrics
    # Handle cases where only one class is present in batch to avoid sklearn errors
    try:
        if len(np.unique(all_labels)) > 1:
            val_roc_auc = roc_auc_score(all_labels, all_preds)
        else:
            val_roc_auc = 0.5 # Default fallback
            
        # Threshold at 0.5 for F1 score
        binary_preds = (all_preds > 0.5).astype(int)
        val_f1 = f1_score(all_labels, binary_preds)
    except Exception as e:
        print(f"Metric calculation error: {e}")
        val_roc_auc = 0.0
        val_f1 = 0.0

    avg_loss = running_loss / step if step > 0 else 0
    
    writer.add_scalar('Loss/validation', avg_loss, epoch)
    writer.add_scalar('Metrics/F1', val_f1, epoch)
    writer.add_scalar('Metrics/ROC_AUC', val_roc_auc, epoch)
    
    print(f"Validation Results - Loss: {avg_loss:.4f}, F1: {val_f1:.4f}, ROC_AUC: {val_roc_auc:.4f}")
    
    return val_roc_auc

def main():
    # Configuration
    BATCH_SIZE = 32
    NUM_WORKERS = 0 # Adjusted to 0 for Windows compatibility
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    STEPS_PER_EPOCH = 20 # Adjust based on dataset size and time constraints (streaming)
    VAL_STEPS = 100
    CHECKPOINT_PATH = "best_rgb_model.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/rgb_experiment_1')
    
    # Data Setup
    transforms = get_transforms()
    
    # Note: Ensure the split names exist in the dataset. Common are 'train', 'validation', 'test'.
    # OpenFake usually has 'train' and 'test', so we use 'test' for validation here.
    try:
        train_dataset = OpenFakeDataset(split="train", transform=transforms)
        val_dataset = OpenFakeDataset(split="test", transform=transforms)
    except Exception as e:
        print(f"Warning: split not found ({e}), using 'train' for validation (debug mode)")
        val_dataset = OpenFakeDataset(split="train", transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model Setup
    model = RGBClassifier().to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed Precision Scaler
    scaler = GradScaler()
    
    best_roc_auc = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nExample Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, 
            epoch, writer, steps_per_epoch=STEPS_PER_EPOCH
        )
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_roc_auc = validate(
            model, val_loader, criterion, device, 
            epoch, writer, steps_per_val=VAL_STEPS
        )
        
        # Checkpointing
        if val_roc_auc > best_roc_auc:
            print(f"Validation ROC-AUC improved ({best_roc_auc:.4f} -> {val_roc_auc:.4f}). Saving model...")
            best_roc_auc = val_roc_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'roc_auc': best_roc_auc,
            }, CHECKPOINT_PATH)
        else:
            print(f"Validation ROC-AUC did not improve (Best: {best_roc_auc:.4f})")

    writer.close()
    print("Training complete.")

if __name__ == '__main__':
    main()
