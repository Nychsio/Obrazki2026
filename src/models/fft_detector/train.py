import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.data.data_loader import get_dataloaders
from src.models.fft_detector.model import FFTResNetDetector

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    model = FFTResNetDetector(num_classes=1).to(device)
    train_loader, val_loader = get_dataloaders(batch_size=32, train_size=4000, val_size=1000)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    
    # --- NOWOŚĆ: LR Scheduler (zgodnie z audytem) ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    epochs = 15 
    
    # --- NOWOŚĆ: Parametry do Early Stopping ---
    best_val_loss = float('inf')
    patience = 4
    trigger_times = 0
    
    print("Rozpoczęcie zaawansowanego treningu...")
    
    for epoch in range(epochs):
        # --- FAZA TRENINGU ---
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            inputs = batch['image'].to(device)
            if isinstance(batch['label'], (list, tuple)):
                labels = torch.tensor([int(x) for x in batch['label']]).to(device).float().unsqueeze(1)
            else:
                labels = batch['label'].to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        epoch_train_loss = running_loss / 4000 
        
        # --- FAZA WALIDACJI ---
        model.eval()
        val_loss = 0.0
        
        # Listy do zbierania wyników dla scikit-learn
        all_labels = []
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoka {epoch+1}/{epochs} [Val]", leave=False):
                inputs = batch['image'].to(device)
                if isinstance(batch['label'], (list, tuple)):
                    labels = torch.tensor([int(x) for x in batch['label']]).to(device).float().unsqueeze(1)
                else:
                    labels = batch['label'].to(device).float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Zbieranie danych (przenosimy na CPU dla numpy)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
        epoch_val_loss = val_loss / 1000
        
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
            }, "fft_detector_best.pth")
        else:
            trigger_times += 1
            print(f"   ⚠️ Brak poprawy. Early stopping counter: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("🛑 Early stopping przerwany trening, model przestał się uczyć.")
                break

if __name__ == "__main__":
    train()