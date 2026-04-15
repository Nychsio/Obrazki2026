import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import os
import sys

# Importujemy szybki DataLoader ze streamingu z RGB (żeby użyć OpenFake!)
from src.models.rgb.data import OpenFakeDataset, get_transforms
from torch.utils.data import DataLoader

# Twoje importy modelu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.gradient_pca.model import GradientPCADetector

def train_pca():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam trening Gradient PCA na urządzeniu: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    model = GradientPCADetector(device=device).to(device)
    
    # === SOTA DATA LOADING (Koniec z blokowaniem GPU!) ===
    transforms = get_transforms()
    train_dataset = OpenFakeDataset(split="train", transform=transforms)
    val_dataset = OpenFakeDataset(split="validation", transform=transforms)
    
    # num_workers=4 i pin_memory=True sprawiają, że GPU nie czeka na procesor
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # Optymalizator
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Skoro mamy dataset streamingowy (terabajty), używamy kroków!
    steps_per_epoch = 250
    val_steps = 50
    
    best_auc = 0.0
    
    for epoch in range(12):
        print(f"\nEpoch {epoch+1}/12 [Train]")
        model.train()
        train_loss = 0
        
        train_iter = iter(train_loader)
        for step in tqdm(range(steps_per_epoch)):
            try:
                images, labels = next(train_iter)
            except Exception:
                continue # Tarcza Ochronna!
                
            images, labels = images.to(device, non_blocking=True), labels.float().view(-1, 1).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Train Loss: {train_loss/steps_per_epoch:.4f}")
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_iter = iter(val_loader)
        
        with torch.no_grad():
            for step in tqdm(range(val_steps), desc="[Validation]"):
                try:
                    images, labels = next(val_iter)
                except Exception:
                    continue
                    
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                val_preds.extend(probs)
                val_labels.extend(labels.numpy())
                
        val_auc = roc_auc_score(val_labels, val_preds)
        val_bin_preds = [1 if p > 0.5 else 0 for p in val_preds]
        val_f1 = f1_score(val_labels, val_bin_preds)
        
        print(f"Val ROC-AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "checkpoints/best_pca_model.pt")
            print("🌟 Zapisano nowy, potężny SOTA model!")

if __name__ == "__main__":
    train_pca()