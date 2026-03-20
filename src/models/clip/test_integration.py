import torch
import torch.nn as nn
import torch.optim as optim
from semantic_judge import SemanticJudgeCLIP
from clip_streamer import CLIPDataStreamer

def run_sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SZYBKI TEST INTEGRACYJNY (Device: {device}) ===")

    try:
        # 1. Inicjalizacja
        print("[1/4] Ładowanie modelu Sędziego...")
        model = SemanticJudgeCLIP(freeze_backbone=True).to(device)
        model.train() # Ustawienie trybu treningowego (uruchamia Dropout itp.)

        print("[2/4] Podłączanie strumienia danych...")
        # Bierzemy mały batch (np. 8) dla szybkości
        streamer = CLIPDataStreamer(batch_size=8)
        train_loader = streamer.create_dataloader(split="train")
        
        # Pobieramy DOKŁADNIE JEDEN batch
        train_iter = iter(train_loader)
        pixel_values, labels = next(train_iter)
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        
        print(f"      -> Kształt pikseli: {pixel_values.shape} (Oczekiwane: [8, 3, 224, 224])")
        print(f"      -> Kształt etykiet: {labels.shape} (Oczekiwane: [8, 1])")

        # 3. Definicja straty i optymalizatora
        criterion = nn.BCEWithLogitsLoss()
        # Używamy wyższego Learning Rate (1e-3) celowo, by szybko wymusić overfitting
        optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

        print("[3/4] Test przejścia w przód (Forward Pass)...")
        logits = model(pixel_values)
        print(f"      -> Kształt wyjścia: {logits.shape} (Oczekiwane: [8, 1])")

        print("[4/4] Test pętli uczącej (Overfit on 1 Batch)...")
        print("      Oczekiwany rezultat: Strata (Loss) powinna drastycznie spadać.")
        
        # Pętla kręci się w kółko NA TYM SAMYM batchu
        for i in range(15):
            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            print(f"      Iteracja {i+1:02d} | Loss: {loss.item():.4f}")

        print("\n✅ TEST ZAKOŃCZONY SUKCESEM. Pipeline jest całkowicie szczelny!")

    except Exception as e:
        print(f"\n❌ BLĄD KRYTYCZNY. Zatrzymano test: {e}")

if __name__ == "__main__":
    run_sanity_check()