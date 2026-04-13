import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import os
import sys

# Allow direct execution from the project root via `python src/models/gradient_pca/train_pca.py`
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.gradient_pca.model import GradientPCADetector
from src.data.data_loader import get_dataloaders

def train_pca():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam trening Gradient PCA na urządzeniu: {device}")

    # Utworzenie katalogu checkpoints jeśli nie istnieje
    os.makedirs("checkpoints", exist_ok=True)

    # Inicjalizacja modelu i przeniesienie na GPU/CPU
    model = GradientPCADetector(device=device).to(device)
    
    # Ładowanie danych (zwiększony batch_size dla RTX 3090 Ti)
    train_loader, val_loader = get_dataloaders(batch_size=64, train_size=8000, val_size=2000)
    
    # Konfiguracja treningu: AdamW i BCEWithLogitsLoss (optymalne dla klasyfikacji binarnej)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 10  # Zwiększone dla pełnego treningu
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        num_batches = 0  # <--- DODANO: Ręczny licznik batchy

        # Pętla po batchach z progress barem
        pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs}")
        for batch in pbar:
            # Rozpakowanie elastyczne
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('pixel_values'))
                labels = batch.get('label', batch.get('labels'))
            else:
                images = batch[0]
                labels = batch[1]

            if images is None or labels is None:
                continue

            # --- SANITACJA TENSORA OBRAZU ---
            # 1. Jeśli tensor ma 3 wymiary [B, H, W], dodaj wymiar kanału [B, 1, H, W]
            if images.dim() == 3:
                images = images.unsqueeze(1)

            # 2. Wymuszamy 3 kanały (pseudokolor), żeby zadowolić ekstraktor Sobela
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[1] == 2:
                # Jeśli to FFT (amplituda, faza), odrzucamy fazę i powielamy amplitudę
                images = images[:, 0:1, :, :].repeat(1, 3, 1, 1)
            # -----------------------------------

            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1  # <--- DODANO: Zwiększamy licznik

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- ZMIENIONY BLOK METRYK NA KONIEC EPOKI ---
        # Używamy num_batches zamiast len(train_loader)
        if num_batches > 0:
            epoch_loss = running_loss / num_batches
        else:
            print("Uwaga: Nie przetworzono żadnych danych w epoce!")
            continue

        auc = roc_auc_score(all_labels, all_preds)
        
        # Predykcje binarne (próg 0.5)
        bin_preds = [1 if p > 0.5 else 0 for p in all_preds]
        f1 = f1_score(all_labels, bin_preds)

        print(f"Koniec epoki {epoch+1} | Loss: {epoch_loss:.4f} | ROC-AUC: {auc:.4f} | F1-Score: {f1:.4f}")

        # Walidacja
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images = batch.get('image', batch.get('pixel_values'))
                    labels = batch.get('label', batch.get('labels'))
                else:
                    images = batch[0]
                    labels = batch[1]
                
                if images is None or labels is None:
                    continue
                
                # Sanitacja tensora obrazu
                if images.dim() == 3:
                    images = images.unsqueeze(1)
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                elif images.shape[1] == 2:
                    images = images[:, 0:1, :, :].repeat(1, 3, 1, 1)
                
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)
                
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1
                
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
        
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            val_auc = roc_auc_score(val_labels, val_preds)
            val_bin_preds = [1 if p > 0.5 else 0 for p in val_preds]
            val_f1 = f1_score(val_labels, val_bin_preds)
            print(f"Val Loss: {avg_val_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val F1-Score: {val_f1:.4f}")
        else:
            avg_val_loss = float('inf')
            print("Uwaga: Brak danych walidacyjnych!")
        
        # Checkpointing (zapisywanie najlepszego modelu na podstawie val loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_gradient_pca_model.pt")
            print(">>> Zapisano nowe, najlepsze wagi modelu!")

if __name__ == "__main__":
    train_pca()