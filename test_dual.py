import torch
from src.models.fft_detector.model import FFTResNetDetector

def test_dual_stream_model():
    print("🔧 Inicjalizacja testu modelu Dual-Stream...")
    
    try:
        # 1. Tworzymy nasz nowy model
        model = FFTResNetDetector(num_classes=1)
        model.eval() # Tryb testowy (wyłącza Dropout)
        print("✅ Model utworzony pomyślnie!")

        # 2. Tworzymy sztuczne paczki danych (Dummy Data) symulujące to, co wypluwa DataLoader
        batch_size = 4
        # RGB: 4 obrazki, 3 kanały (Kolor), 224x224 piksele
        dummy_rgb = torch.randn(batch_size, 3, 224, 224) 
        # FFT: 4 obrazki, 2 kanały (Amplituda i Faza), 224x224 piksele
        dummy_fft = torch.randn(batch_size, 2, 224, 224) 

        print(f"📦 Przygotowano batch próbny:")
        print(f"   -> RGB shape: {dummy_rgb.shape}")
        print(f"   -> FFT shape: {dummy_fft.shape}")

        # 3. Próbny strzał (Forward pass)
        print("🚀 Odpalamy dane przez oba strumienie modelu...")
        with torch.no_grad():
            outputs = model(dummy_rgb, dummy_fft)
            
        print(f"✅ Sukces! Model połączył strumienie. Kształt wyjścia: {outputs.shape}")
        
        # Oczekujemy, że z 4 obrazków wypluje 4 decyzje [4, 1]
        if outputs.shape == (batch_size, 1):
            print("🏆 TEST ZALICZONY WZOROWO! Architektura gotowa na RunPoda.")
        else:
            print("⚠️ Zły wymiar wyjściowy!")

    except Exception as e:
        print(f"❌ TEST OBLANY (Błąd): {e}")

if __name__ == "__main__":
    test_dual_stream_model()