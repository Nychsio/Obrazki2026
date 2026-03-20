import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Importy z Twojego projektu
from src.models.fft_detector.model import FFTResNetDetector
from src.data.data_loader import get_streaming_dataloader

def generate_heatmap(model_path="fft_detector_epoch_1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Ładowanie przetrenowanego modelu
    model = FFTResNetDetector(num_classes=1).to(device)
    # Wczytanie wag (zastąp nazwę pliku tą, która Ci się zapisze)
    # parametr map_location upewnia się, że działa nawet jeśli trenowałeś na GPU, a odpalasz na CPU
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Wagi modelu załadowane poprawnie.")
    except Exception as e:
        print(f"Błąd ładowania wag (czy model na pewno się wytrenował?): {e}")
        return

    model.eval() # Tryb ewaluacji (wyłącza np. Dropout)

    # 2. Definicja warstwy docelowej dla Grad-CAM
    # W ResNet18 zazwyczaj sprawdzamy ostatnią warstwę konwolucyjną przed klasyfikatorem
    target_layers = [model.backbone.layer4[-1]]

    # 3. Inicjalizacja Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Pobranie jednej próbki testowej
    loader = get_streaming_dataloader(batch_size=1)
    batch = next(iter(loader))
    input_tensor = batch['image'].to(device)
    label = batch['label'].item()

    # 5. Generowanie mapy aktywacji
    # Grad-CAM zwraca maskę w skali szarości [0, 1]
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # 6. Wizualizacja: Nałożenie heatmapy na nasze widmo Fouriera
    # Krok A: Pobieramy obraz z tensora [1, 1, 224, 224] do formatu numpy [224, 224]
    img_freq = input_tensor.squeeze().cpu().numpy()
    
    # Krok B: Konwersja jednokanałowego widma Fouriera na 3 kanały (wymóg biblioteki do wizualizacji)
    img_freq_rgb = cv2.cvtColor(img_freq, cv2.COLOR_GRAY2RGB)

    # Krok C: Nałożenie ciepłych kolorów na widmo
    visualization = show_cam_on_image(img_freq_rgb, grayscale_cam, use_rgb=True)

    # 7. Zapisanie/Wyświetlenie wyniku
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_freq_rgb)
    plt.title(f"Widmo Fouriera (Klasa: {'Fake' if label==1 else 'Real'})")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM XAI")
    plt.axis('off')

    plt.savefig("portfolio_xai_showcase.png", bbox_inches='tight')
    print("Zapisano grafikę XAI jako portfolio_xai_showcase.png!")

if __name__ == "__main__":
    generate_heatmap()