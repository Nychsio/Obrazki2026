import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def test_single_image(image_path, model_weights_path):
    print("🔧 Ładowanie modelu i procesora CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ładujemy procesor (taki sam jak podczas treningu)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Ładujemy nasz zdefiniowany model Sędziego Semantycznego
    from src.models.clip.semantic_judge import SemanticJudgeCLIP
    model = SemanticJudgeCLIP()
    
    # 1. Wgrywamy cały słownik Checkpointu
    checkpoint = torch.load(model_weights_path, map_location=device, weights_only=False)
    
    # 2. Wyciągamy z niego same wagi szkieletu (backbone) modelu CLIP
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Wagi z RunPoda (Epoka: {checkpoint.get('epoch', 'Nieznana')}) załadowane pomyślnie!")
        print(f"🌟 Parametry zapisane z RunPoda - ROC-AUC: {checkpoint.get('roc_auc', 'Brak'):.4f}, F1: {checkpoint.get('f1_score', 'Brak'):.4f}")
    except Exception as e:
        print(f"❌ Błąd ładowania wag: {e}")
        return

    model.to(device)
    model.eval()

    print(f"🖼️ Analiza obrazu: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Błąd otwierania obrazka: {e}")
        return
    
    # 1. Przetwarzamy TYLKO obrazek (bez tekstów)
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 2. Puszczamy przez Sędziego
    with torch.no_grad():
        # Podajemy same piksele do Twojego SemanticJudgeCLIP
        # (jeśli wyrzuci błąd, że nie zna 'pixel_values', zmień na: outputs = model(inputs.pixel_values) )
        outputs = model(pixel_values=inputs.pixel_values)
        
        # 3. Zakładam, że model wypluwa 1 logit (binarna klasyfikacja), więc liczymy Sigmoid (0 do 1)
        prob = torch.sigmoid(outputs).item()

    # 4. Werdykt! (Powyżej 0.5 to zazwyczaj Fake w takich projektach)
    print("\n📊 --- WYNIKI SĘDZIEGO ---")
    if prob > 0.5:
        print(f"🛑 FAKE (AI Generated) | Pewność: {prob * 100:.2f}%")
    else:
        print(f"✅ REAL (Authentic)    | Pewność: {(1 - prob) * 100:.2f}%")

if __name__ == "__main__":
    # ==========================================
    # ⚠️ MIEJSCE NA TWOJE ZMIANY:
    # ==========================================
    
    # 1. Ścieżka do jakiegoś obrazka z internetu (skopiuj go do folderu z projektem)
    TEST_IMAGE = "koleszka.jpg" 
    
    # 2. Ścieżka do Twojego pliku, który pobrałeś z RunPoda (te 336 MB)
    WEIGHTS = "moj_wytrenowany_clip.pth" 
    
    test_single_image(TEST_IMAGE, WEIGHTS)