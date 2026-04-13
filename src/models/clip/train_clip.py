import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import os

# Importy z Twoich lokalnych plików
from semantic_judge import SemanticJudgeCLIP
from clip_streamer import CLIPDataStreamer

def train_clip():
    # 1. Konfiguracja
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam trening na: {device}")
    
    batch_size = 32  # CLIP może potrzebować mniejszego batch_size ze względu na duży model
    max_steps_per_epoch = 2000  # Zwiększone dla pełnego treningu
    val_steps = 400             # Zwiększone dla pełnego treningu
    epochs = 12                 # Zwiększone dla pełnego treningu
    patience = 4                # Early Stopping: ile epok czekać bez poprawy
    
    # 2. Inicjalizacja Modelu i Danych
    model = SemanticJudgeCLIP(freeze_backbone=True).to(device)
    streamer = CLIPDataStreamer(batch_size=batch_size)
    
    train_loader = streamer.create_dataloader(split="train")
    val_loader = streamer.create_dataloader(split="validation") # Zakładam, że OpenFake ma ten split
    
    # 3. Optymalizator i Strata
    criterion = nn.BCEWithLogitsLoss() # Stabilne numerycznie połączenie Sigmoida i BCE
    # Optymalizujemy TYLKO głowę klasyfikującą, bo reszta jest zamrożona!
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Zmienne do Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    # 4. Główna Pętla
    for epoch in range(epochs):
        print(f"\n=== Epoka {epoch+1}/{epochs} ===")
        
        # --- TRENING ---
        model.train()
        train_loss = 0.0
        actual_train_steps = 0
        
        # Używamy manualnego licznika kroków ze względu na streaming
        train_iter = iter(train_loader)
        for step in tqdm(range(max_steps_per_epoch), desc="Trening"):
            try:
                pixel_values, labels = next(train_iter)
            except StopIteration:
                break # Koniec strumienia
                
            pixel_values = pixel_values.to(device)
            labels = labels.to(device=device, dtype=torch.float32).view(-1, 1)
            
            optimizer.zero_grad()
            logits = model(pixel_values).view(-1, 1)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            actual_train_steps += 1
            
        avg_train_loss = train_loss / actual_train_steps if actual_train_steps > 0 else 0
        
        # --- WALIDACJA ---
        model.eval()
        val_loss = 0.0
        actual_val_steps = 0
        all_labels = []
        all_preds = []
        all_probs = []
        
        val_iter = iter(val_loader)
        with torch.no_grad():
            for step in tqdm(range(val_steps), desc="Walidacja"):
                try:
                    pixel_values, labels = next(val_iter)
                except StopIteration:
                    break
                    
                pixel_values = pixel_values.to(device)
                labels = labels.to(device=device, dtype=torch.float32).view(-1, 1)
                
                logits = model(pixel_values).view(-1, 1)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                actual_val_steps += 1
                
                # Zmiana logitów na prawdopodobieństwa (Sigmoid) dla metryk
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_labels.extend(labels.cpu().numpy().reshape(-1))
                all_preds.extend(preds.cpu().numpy().reshape(-1))
                all_probs.extend(probs.cpu().numpy().reshape(-1))
                
        avg_val_loss = val_loss / actual_val_steps if actual_val_steps > 0 else 0
        scheduler.step()
        
        # --- METRYKI (ROC-AUC i F1) ---
        # ROC-AUC: Jak dobrze model separuje klasy niezależnie od progu (im bliżej 1.0, tym lepiej)
        # F1-Score: Średnia harmoniczna precyzji i czułości (ważne przy niezbalansowanych danych)
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val ROC-AUC: {auc:.4f} | Val F1-Score: {f1:.4f}")
        
        # --- EARLY STOPPING & ZAPIS ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Zapisujemy pełny model (klasyfikator + backbone) dla kompletności
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'roc_auc': auc,
                'f1_score': f1
            }, "checkpoints/clip_model_best.pth")
            print("🌟 Zapisano nowy najlepszy model!")
        else:
            epochs_no_improve += 1
            print(f"Brak poprawy od {epochs_no_improve} epok.")
            
        if epochs_no_improve >= patience:
            print("\n⏹️ Early Stopping zadziałał. Przerywam trening, aby uniknąć overfittingu.")
            break

if __name__ == "__main__":
    train_clip()