# Memory Bank - Projekt Detekcji Deepfake i Obrazów Wygenerowanych przez AI

## Przegląd Projektu

Projekt implementuje system detekcji obrazów wygenerowanych przez AI (deepfake) wykorzystujący różne podejścia:
1. **Analiza szumów (Noise)**: Wykrywanie artefaktów szumowych charakterystycznych dla generowanych obrazów
2. **Analiza cech RGB**: Wykorzystanie wstępnie wytrenowanego EfficientNet-B0 do klasyfikacji semantycznej
3. **Analiza częstotliwościowa (FFT)**: Detekcja artefaktów w dziedzinie częstotliwości za pomocą transformaty Fouriera
4. **Model CLIP**: Fine-tuning modelu CLIP do klasyfikacji binarnej
5. **Gradient PCA**: Ekstrakcja cech gradientowych i analiza macierzy kowariancji

## Kluczowe Komponenty

### Modele
1. **Noise Binary Classifier** (`src/models/noise/`)
   - Architektura: CNN na szumach resztkowych
   - Filtr: High-pass Gaussian filter
   - Zapisany model: `best_noise_model.pt`

2. **RGB Classifier** (`src/models/rgb/`)
   - Backbone: EfficientNet-B0 (timm)
   - Głowa klasyfikacyjna: Linear(1280→1)
   - Zapisany model: `best_rgb_model.pt`

3. **FFT ResNet Detector** (`src/models/fft_detector/`)
   - Backbone: ResNet18 z modyfikacją dla 2 kanałów (amplituda + faza)
   - Transformacja: ComplexFourierTransform
   - Zapisany model: `fft_detector_best.pth`

4. **Semantic Judge CLIP** (`src/models/clip/`)
   - Model: CLIP ViT-Base-Patch32
   - Głowa klasyfikacyjna: Linear(768→256→1)
   - Status: Wymaga poprawek (zobacz audyt)

5. **Gradient PCA Extractor** (`src/models/gradient_pca/`)
   - Ekstraktor: GradientCovarianceExtractor
   - Cechy: Macierze kowariancji gradientów (2x2 → 4-wymiarowe)
   - Status: Implementacja podstawowa ukończona

### Dane
- **Dataset**: OpenFake (ComplexDataLab/OpenFake) z Hugging Face
- **Tryb**: Streaming dla efektywnego ładowania dużych zbiorów
- **Klasy**: Real (0) vs Fake (1)
- **Transformacje**: Augmentacje (kompresja, rozmycie, szum) dla treningu

### Narzędzia
- **Trening**: Mixed precision training, TensorBoard logging
- **XAI**: Grad-CAM, attention maps (CLIP)
- **Ewaluacja**: ROC-AUC, F1-Score, precision, recall
- **Deployment**: Docker, konfiguracje YAML

## Status Projektu

### ✅ Działające komponenty
1. Model Noise - pełna implementacja
2. Model RGB - pełna implementacja z XAI (Grad-CAM)
3. Model FFT - podstawowa implementacja (wymaga poprawek)
4. Model Gradient PCA - implementacja podstawowa ukończona
5. Pipeline danych - streaming z OpenFake

### ⚠️ Wymagające poprawek
1. Model CLIP - błędy w obliczeniach loss, brak shuffle danych
2. Model FFT - błędy syntaktyczne w train.py, brak walidacji
3. Dokumentacja - niekompletna w niektórych obszarach

### 🚀 Planowane rozszerzenia
1. Ensemble modeli (noise + RGB + FFT + CLIP + Gradient PCA)
2. Real-time API REST
3. Multi-modal detection (metadane + obraz)
4. Extended datasets

## Struktura Katalogów

```
.memory/
├── overview.md              # Ten plik - główna dokumentacja projektu
├── frontend.md              # Dokumentacja frontendu React
├── deployment.md            # Dokumentacja backendu FastAPI i inference
├── quickstart.md            # Szybki start dla nowych developerów
├── models/                  # Dokumentacja poszczególnych modeli ML
│   ├── clip_model.md        # Model CLIP - analiza semantyczna
│   ├── fft_model.md         # Model FFT - analiza częstotliwościowa
│   ├── rgb_model.md         # Model RGB - analiza pikseli
│   ├── noise_model.md       # Model Noise - analiza szumów
│   └── gradient_pca_model.md # Model PCA - analiza gradientów
├── data/                    # Dokumentacja danych
│   └── dataset.md           # Dataset OpenFake i preprocessing
├── training/                # Procesy treningowe (w budowie)
├── evaluation/              # Metryki i ewaluacja (w budowie)
├── deployment/              # Pliki źródłowe deployment (kopie)
│   └── app/                 # Kopie plików FastAPI dla referencji
└── issues/                  # Zidentyfikowane problemy
    └── known_issues.md      # Lista znanych problemów i TODO
```

## Szybkie Odwołania

### Uruchomienie treningu
```bash
# Model noise
python src/models/noise/train.py

# Model RGB  
python src/models/rgb/train.py

# Model FFT
python src/models/fft_detector/train.py

# Model CLIP
python src/models/clip/train_clip.py

# Test Gradient PCA
python src/models/gradient_pca/simple_test.py
```

### Inferencja
```bash
# RGB model
python src/models/rgb/inference.py test_image.jpg --model_path best_rgb_model.pt

# FFT model
python src/models/fft_detector/predict_single.py test_image.jpg

# XAI (Grad-CAM)
python src/models/rgb/explain.py

# Gradient PCA features
python -c "from src.models.gradient_pca import GradientCovarianceExtractor; import torch; extractor = GradientCovarianceExtractor(); img = torch.randn(1, 3, 224, 224); features = extractor(img); print(f'Features shape: {features.shape}')"
```

### Środowisko
```bash
# Conda
conda env create -f environment.yml

# Pip
pip install -r requirements.txt
```

## Kontakt i Wsparcie

- **Repozytorium**: git@github.com:Nychsio/Obrazki2026.git
- **Ostatni commit**: 73c95817137ac92e89138addc89646ee02bf20ad
- **Środowisko**: Python 3.10, PyTorch 2.2+, CUDA 12.1 (opcjonalnie)

---

*Memory Bank zaktualizowany: 2026-04-16*  
*Cel: Centralne źródło wiedzy o projekcie dla nowych developerów i utrzymanie kontekstu*
