import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.fft_detector.model import FFTResNetDetector
from src.data.data_loader import get_streaming_dataloader

def test_and_explain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # 1. Inicjalizacja modelu i wczytanie wag z PoC
    model = FFTResNetDetector(num_classes=1).to(device)
    checkpoint_path = "src/models/fft_detector/poc_fft_checkpoint.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Sprawdzamy, czy model został zapisany jako słownik z metadanymi
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # Klasyczny fallback
        print("Wagi modelu załadowane pomyślnie!")
    except Exception as e:
        print(f"Błąd ładowania wag: {e}")
        return

    model.eval()

    # 2. Pobranie próbki
    print("Pobieranie próbki testowej...")
    loader = get_streaming_dataloader(batch_size=1)
    batch = next(iter(loader))
    input_tensor = batch['image'].to(device)
    
    # --- TO JEST POPRAWIONY BLOK (LINIA 42+) ---
    if isinstance(batch['label'], (list, tuple)):
        label = int(batch['label'][0])
    else:
        label = int(batch['label'].item())
    # ------------------------------------------

    
    # 3. Predykcja
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred_class = "Fake" if prob > 0.5 else "Real"
        true_class = "Fake" if label == 1 else "Real"
        
        print(f"\n--- WYNIK ---")
        print(f"Prawdziwa klasa: {true_class}")
        print(f"Predykcja modelu: {pred_class} (Pewność: {prob*100:.2f}%)")

    # 4. Grad-CAM
    print("\nGenerowanie mapy Grad-CAM...")
    target_layers = [model.backbone.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Wizualizacja
    img_freq = input_tensor.squeeze().cpu().numpy()
    img_freq_norm = (img_freq - img_freq.min()) / (img_freq.max() - img_freq.min() + 1e-8)
    img_freq_rgb = cv2.cvtColor(np.float32(img_freq_norm), cv2.COLOR_GRAY2RGB)
    
    visualization = show_cam_on_image(img_freq_rgb, grayscale_cam, use_rgb=True)
    
    # 5. Zapis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_freq_norm, cmap='gray')
    plt.title(f"Widmo Fouriera (Prawda: {true_class})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM (Predykcja: {pred_class})")
    plt.axis('off')
    
    out_path = "portfolio_xai_result.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Zapisano wizualizację do pliku: {out_path}")

if __name__ == "__main__":
    test_and_explain()