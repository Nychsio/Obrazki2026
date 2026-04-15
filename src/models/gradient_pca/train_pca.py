import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import numpy as np
import os
import sys
import platform
import PIL

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.gradient_pca.model import GradientPCADetector
from src.models.rgb.data import OpenFakeDataset, get_transforms
from torch.utils.data import DataLoader

def train_pca():
    # === 🚀 GPU OPTIMIZATION FOR RTX 3090 Ti (Ampere) 🚀 ===
    # Włączamy TF32 (TensorFloat-32) - sprzętowe przyspieszenie mnożenia macierzy bez utraty jakości
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Włączamy autotuning algorytmów konwolucyjnych dla tego konkretnego GPU
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Uruchamiam HYPER-SOTA Gradient PCA na: {device} (TF32: WŁĄCZONE)")

    os.makedirs("checkpoints", exist_ok=True)
    
    # Inicjalizacja modelu i konwersja do formatu channels_last (Wymagane dla maksymalnej prędkości Tensor Cores)
    model = GradientPCADetector(device=device).to(device)
    model = model.to(memory_format=torch.channels_last)
    
    # Maksymalizujemy CPU Workers, żeby nadążyć za kartą
    workers = 0 if platform.system() == 'Windows' else 8
    
    # 🔥 Zwiększamy strumień danych, bo RTX 3090 Ti ma 24GB VRAM!
    batch_size = 192 
    
    transforms = get_transforms()
    train_dataset = OpenFakeDataset(split="train", transform=transforms)
    val_dataset = OpenFakeDataset(split="test", transform=transforms)
    
    # Dodajemy persistent_workers, żeby nie marnować czasu na tworzenie procesów pomiędzy epokami
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=workers, 
        pin_memory=True,
        persistent_workers=(workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=workers, 
        pin_memory=True,
        persistent_workers=(workers > 0)
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # Zostawiamy 250 kroków, ale teraz w jednym kroku przetwarzamy 3x więcej zdjęć!
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
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)
            except (IOError, OSError, ValueError, PIL.UnidentifiedImageError):
                continue
                
            # Konwersja obrazów do channels_last w locie dla Tensor Cores
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            
            # Kuloodporna konwersja (zamienia teksty "fake"/"1" na float 1.0, resztę na 0.0)
            numeric_labels = [1.0 if str(l).lower() in ['fake', '1', 'true', '1.0'] else 0.0 for l in labels]
            labels = torch.tensor(numeric_labels, dtype=torch.float32).view(-1, 1).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels.to(dtype=logits.dtype))
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        print(f"Train Loss: {train_loss/steps_per_epoch:.4f}")
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_preds, val_labels_list = [], []
        val_iter = iter(val_loader)
        valid_steps_done = 0
        
        with torch.no_grad():
            for step in tqdm(range(val_steps), desc="[Validation]"):
                try:
                    images, labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    images, labels = next(val_iter)
                except (IOError, OSError, ValueError, PIL.UnidentifiedImageError):
                    continue
                    
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                
                # Kuloodporna konwersja dla walidacji
                numeric_labels = [1.0 if str(l).lower() in ['fake', '1', 'true', '1.0'] else 0.0 for l in labels]
                labels_tensor = torch.tensor(numeric_labels, dtype=torch.float32)
                labels_gpu = labels_tensor.view(-1, 1).to(device, non_blocking=True)
                
                with autocast():
                    logits = model(images)
                    v_loss = criterion(logits, labels_gpu.to(dtype=logits.dtype))
                    probs = torch.sigmoid(logits)
                
                val_loss += v_loss.item()
                valid_steps_done += 1
                
                val_preds.extend(probs.cpu().numpy().flatten().tolist())
                val_labels_list.extend(labels_tensor.cpu().numpy().flatten().tolist())
                
                val_loss += v_loss.item()
                valid_steps_done += 1
                
                val_preds.extend(probs.cpu().numpy().flatten().tolist())
                val_labels_list.extend(labels_tensor.cpu().numpy().flatten().tolist())
                
        avg_val_loss = val_loss / max(1, valid_steps_done)
        val_auc = roc_auc_score(val_labels_list, val_preds)
        
        # Optymalizacja progu F1 (Youden's J)
        fpr, tpr, thresholds = roc_curve(val_labels_list, val_preds)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        val_bin_preds = [1 if p > optimal_threshold else 0 for p in val_preds]
        val_f1 = f1_score(val_labels_list, val_bin_preds)
        
        print(f"Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} (Opt. Threshold: {optimal_threshold:.2f})")
        
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "checkpoints/best_pca_model.pt")
            print("🌟 Zapisano nowy, POTĘŻNY model PCA (TF32 Optimized)!")

if __name__ == "__main__":
    train_pca()