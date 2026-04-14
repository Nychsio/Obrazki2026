import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from semantic_judge import SemanticJudgeCLIP
from clip_streamer import CLIPDataStreamer

def evaluate_model(model_weights_path="checkpoints/clip_classifier_best.pth", test_steps=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rozpoczynam ewaluację na: {device}")
    
    # 1. Inicjalizacja modelu i wczytanie wag
    model = SemanticJudgeCLIP(freeze_backbone=True).to(device)
    try:
        model.classifier.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("✅ Pomyślnie wczytano wagi modelu.")
    except FileNotFoundError:
        print("❌ Błąd: Nie znaleziono pliku z wagami. Upewnij się, że model został wytrenowany.")
        return

    model.eval()

    # 2. Inicjalizacja strumienia danych (split 'test')
    batch_size = 32
    streamer = CLIPDataStreamer(batch_size=batch_size)
    
    # OpenFake używa splitu 'test' zamiast 'validation'.
    # Fallback na 'validation' pozostaje dla kompatybilności wstecznej i jest mapowany w streamerze na 'test'.
    try:
        test_loader = streamer.create_dataloader(split="test")
    except Exception as e:
        print(f"Nie udało się załadować splitu 'test', próbuję aliasu 'validation'. Błąd: {e}")
        test_loader = streamer.create_dataloader(split="validation")

    criterion = nn.BCEWithLogitsLoss()
    
    test_loss = 0.0
    actual_test_steps = 0
    all_labels = []
    all_preds = []
    all_probs = []

    # 3. Pętla testująca
    test_iter = iter(test_loader)
    with torch.no_grad():
        for step in tqdm(range(test_steps), desc="Testowanie (Ewaluacja)"):
            try:
                pixel_values, labels = next(test_iter)
            except StopIteration:
                print("\nOsiągnięto koniec strumienia danych.")
                break
                
            pixel_values = pixel_values.to(device)
            labels = labels.to(device=device, dtype=torch.float32).view(-1, 1)
            
            logits = model(pixel_values).view(-1, 1)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            actual_test_steps += 1
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy().reshape(-1))
            all_preds.extend(preds.cpu().numpy().reshape(-1))
            all_probs.extend(probs.cpu().numpy().reshape(-1))

    # 4. Obliczanie metryk
    # Poprawne obliczenie średniej straty: dzielimy przez faktyczną liczbę kroków
    avg_test_loss = test_loss / actual_test_steps if actual_test_steps > 0 else 0
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    
    # Zabezpieczona confusion matrix z jawnymi labelami
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback dla przypadków gdy macierz nie jest 2x2
        tn = fp = fn = tp = 0
        print("⚠️  Uwaga: Confusion matrix nie jest 2x2, ustawiono wszystkie wartości na 0")

    # 5. Raport
    print("\n" + "="*40)
    print("📊 RAPORT Z EWALUACJI SĘDZIEGO CLIP")
    print("="*40)
    print(f"Przeanalizowano próbek: {len(all_labels)}")
    print(f"Test Loss:  {avg_test_loss:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print("-" * 40)
    print("Macierz pomyłek (Confusion Matrix):")
    print(f"True Negatives (Real jako Real): {tn}")
    print(f"False Positives (Real jako Fake): {fp}  <-- FAŁSZYWE ALARMY")
    print(f"False Negatives (Fake jako Real): {fn}  <-- PRZEOCZENIA")
    print(f"True Positives (Fake jako Fake): {tp}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()