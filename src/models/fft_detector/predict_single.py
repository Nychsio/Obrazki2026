import torch
from PIL import Image
import os
from torchvision import transforms
import torch.nn.functional as F

# Importy Twoich modułów
from src.models.fft_detector.model import FFTResNetDetector
from src.models.fft_detector.transforms import ComplexFourierTransform

def predict_image(image_path):
    # 1. Ustawienia urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # 2. Inicjalizacja modelu i ładowanie wag
    model = FFTResNetDetector(num_classes=1).to(device)
    checkpoint_path = "fft_detector_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"BŁĄD: Nie znaleziono pliku wag w {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model załadowany pomyślnie.")

    # 3. Przygotowanie obrazu (Transformacje)
    # Musimy powtórzyć dokładnie to samo, co działo się podczas treningu
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        ComplexFourierTransform()
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device) # Dodajemy wymiar batcha [1, 1, 224, 224]
    except Exception as e:
        print(f"BŁĄD podczas wczytywania obrazu: {e}")
        return

    # 4. Predykcja
    with torch.no_grad():
        output = model(input_tensor)
        # Przekształcamy logit na prawdopodobieństwo (0-1)
        probability = torch.sigmoid(output).item()
        
    # 5. Wynik
    # 1.0 = Fake, 0.0 = Real (zgodnie z naszym treningiem)
    label = "FAKE (AI Generated)" if probability > 0.5 else "REAL (Authentic)"
    confidence = probability if probability > 0.5 else (1 - probability)

    print("\n" + "="*30)
    print(f"ANALIZA OBRAZU: {os.path.basename(image_path)}")
    print(f"WERDYKT: {label}")
    print(f"PEWNOŚĆ: {confidence * 100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    # Tutaj podaj ścieżkę do swojego pliku JPG!
    path_to_test = "test_image.jpg" 
    predict_image(path_to_test)