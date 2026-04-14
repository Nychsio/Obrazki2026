import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.data.data_loader import get_dataloaders
from src.models.fft_detector.model import FFTResNetDetector


def rgb_to_fft_two_channel(inputs: torch.Tensor) -> torch.Tensor:
    """
    Konwertuje batch obrazów [B, 3, H, W] do reprezentacji FFT [B, 2, H, W]
    (kanały: log-amplituda oraz faza), w pełni tensorowo na aktualnym urządzeniu.
    """
    if inputs.ndim != 4:
        raise ValueError(f"Oczekiwano tensora 4D [B, C, H, W], otrzymano: {tuple(inputs.shape)}")

    channels = inputs.size(1)
    if channels == 2:
        return inputs

    if channels == 3:
        # Luminancja w standardzie ITU-R BT.601
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
        gray = (inputs * rgb_weights).sum(dim=1)
    elif channels == 1:
        gray = inputs.squeeze(1)
    else:
        raise ValueError(f"Nieobsługiwana liczba kanałów wejściowych: {channels}")

    fft = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    amplitude = torch.log1p(torch.abs(fft_shifted))
    phase = torch.angle(fft_shifted)

    return torch.stack((amplitude, phase), dim=1)

def safe_dataloader(dataloader):
    """Tarcza MLOps: Generator pomijający uszkodzone paczki danych z Hugging Face"""
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except Exception as e:
            print(f"\n⚠️ Tarcza włączona: Pominięto uszkodzony plik! ({e})")
            continue

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Utworzenie katalogu checkpoints jeśli nie istnieje
    os.makedirs("checkpoints", exist_ok=True)

    model = FFTResNetDetector(num_classes=1).to(device)
    train_loader, val_loader = get_dataloaders(batch_size=64, train_size=8000, val_size=2000)  # Zwiększone dla RTX 3090 Ti
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    
    # --- NOWOŚĆ: LR Scheduler (zgodnie z audytem) ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    epochs = 12  # Zwiększone dla pełnego treningu
    
    # --- NOWOŚĆ: Parametry do Early Stopping ---
    best_val_loss = float('inf')
    patience = 4
    trigger_times = 0
    
    print("Rozpoczęcie zaawansowanego treningu...")
    
    for epoch in range(epochs):
        # --- FAZA TRENINGU ---
        model.train()
        running_loss = 0.0
        
        # Używamy safe_dataloader, żeby zepsute zdjęcia nas nie wysadziły w powietrze
        progress_bar = tqdm(safe_dataloader(train_loader), desc=f"Epoka {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            inputs_rgb = batch['image'].to(device)
            inputs_fft = rgb_to_fft_two_channel(inputs_rgb)
            if isinstance(batch['label'], (list, tuple)):
                labels = torch.tensor([int(x) for x in batch['label']]).to(device).float().unsqueeze(1)
            else:
                labels = batch['label'].to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs_rgb, inputs_fft)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs_rgb.size(0)
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # Obliczanie średniego loss na podstawie faktycznej liczby próbek
        total_train_samples = 0
        for batch in train_loader:
            total_train_samples += batch['image'].size(0)
        epoch_train_loss = running_loss / total_train_samples if total_train_samples > 0 else 0.0 
        
        # --- FAZA WALIDACJI ---
        model.eval()
        val_loss = 0.0
        
        # Listy do zbierania wyników dla scikit-learn
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(safe_dataloader(val_loader), desc=f"Epoka {epoch+1}/{epochs} [Val]", leave=False):
                inputs_rgb = batch['image'].to(device)
                inputs_fft = rgb_to_fft_two_channel(inputs_rgb)
                if isinstance(batch['label'], (list, tuple)):
                    labels = torch.tensor([int(x) for x in batch['label']]).to(device).float().unsqueeze(1)
                else:
                    labels = batch['label'].to(device).float().unsqueeze(1)
                
                outputs = model(inputs_rgb, inputs_fft)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs_rgb.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Zbieranie danych (przenosimy na CPU dla numpy)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        # Obliczanie średniego val loss na podstawie faktycznej liczby próbek walidacyjnych
        total_val_samples = 0
        for batch in val_loader:
            total_val_samples += batch['image'].size(0)
        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0.0
        
        # --- NOWOŚĆ: Obliczanie zaawansowanych metryk ---
        # Spłaszczamy listy
        y_true = np.array(all_labels).flatten()
        y_prob = np.array(all_probs).flatten()
        y_pred = np.array(all_preds).flatten()
        
        val_acc = (y_pred == y_true).mean() * 100
        
        # Zabezpieczenie na wypadek, gdyby w batchu była tylko jedna klasa
        try:
            val_auc = roc_auc_score(y_true, y_prob)
            val_precision = precision_score(y_true, y_pred, zero_division=0)
            val_recall = recall_score(y_true, y_pred, zero_division=0)
            val_f1 = f1_score(y_true, y_pred, zero_division=0)
        except ValueError:
            val_auc, val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0, 0.0
        
        print(f"\n-> EPOKA {epoch+1} PODSUMOWANIE:")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"   Val Acc: {val_acc:.2f}% | AUC-ROC: {val_auc:.4f} | F1: {val_f1:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f}")
        
        # Aktualizacja Schedulera
        scheduler.step(epoch_val_loss)
        
        # --- NOWOŚĆ: Early Stopping i Checkpointing ---
        if epoch_val_loss < best_val_loss:
            print(f"   🌟 Poprawa Val Loss ({best_val_loss:.4f} -> {epoch_val_loss:.4f}). Zapisywanie najlepszego modelu...")
            best_val_loss = epoch_val_loss
            trigger_times = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'metrics': {'acc': val_acc, 'auc': val_auc, 'f1': val_f1}
            }, "checkpoints/fft_detector_best.pth")
        else:
            trigger_times += 1
            print(f"   ⚠️ Brak poprawy. Early stopping counter: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("🛑 Early stopping przerwany trening, model przestał się uczyć.")
                break

if __name__ == "__main__":
    train()