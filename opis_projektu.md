# Dokumentacja Projektu: Detekcja Deepfake i Obrazów Wygenerowanych przez AI

## Przegląd Projektu

Projekt implementuje system detekcji obrazów wygenerowanych przez AI (deepfake) wykorzystujący różne podejścia: analizę szumów (noise), cech RGB oraz modele oparte na CLIP i FFT. System jest zbudowany w Pythonie z użyciem PyTorch i ma na celu identyfikację obrazów rzeczywistych vs. wygenerowanych przez AI.

## Struktura Projektu

```
g:/obrazki/
├── src/
│   ├── noise/              # Model analizy szumów
│   │   ├── model.py        # Architektura modelu noise
│   │   └── train.py        # Skrypt treningowy noise
│   ├── rgb/                # Model analizy cech RGB
│   │   ├── data.py         # Dataset i transformacje
│   │   ├── feature_extractor.py  # Ekstraktor cech EfficientNet
│   │   ├── train.py        # Skrypt treningowy RGB
│   │   ├── inference.py    # Inferencja dla pojedynczych obrazów
│   │   └── explain.py      # Explainable AI (Grad-CAM)
│   ├── models/
│   │   ├── clip/           # Modele oparte na CLIP
│   │   └── fft_detector/   # Detektor oparty na FFT
│   ├── data/               # Ładowanie danych
│   ├── training/           # Narzędzia treningowe
│   ├── utils/              # Narzędzia pomocnicze
│   └── xai/                # Explainable AI
├── configs/                # Konfiguracje
├── docker/                 # Konfiguracja Docker
├── notebooks/              # Eksperymenty i analizy
├── runs/                   # Logi TensorBoard
│   ├── noise_experiment_1/
│   └── rgb_experiment_1/
└── tests/                  # Testy jednostkowe
```

## Model Noise (Analiza Szumów)

### Opis Działania

Model noise analizuje szumy resztkowe w obrazach, które są charakterystyczne dla różnych metod generowania obrazów. Wykorzystuje filtr górnoprzepustowy do wyodrębnienia szumów, a następnie klasyfikuje je za pomocą sieci konwolucyjnej.

### Architektura

1. **Filtr górnoprzepustowy** (`high_pass_filter`):
   - Odejmuje rozmyty obraz Gaussa od oryginalnego
   - Parametry: `kernel_size=5`, `sigma=1.0`
   - Wynik: tensor resztkowy z szumami

2. **Sieć konwolucyjna** (`NoiseBinaryClassifier`):
   - 4 warstwy konwolucyjne (32, 64, 128, 256 filtrów)
   - Funkcja aktywacji ReLU
   - Warstwy pooling (MaxPool2d)
   - AdaptiveAvgPool2d do redukcji wymiarów
   - Warstwy liniowe: 256 → 128 → 1

### Pliki Powiązane

- **`src/noise/model.py`**: Definicja modelu i filtra
- **`src/noise/train.py`**: Skrypt treningowy z walidacją
- **`best_noise_model.pt`**: Zapisane wagi najlepszego modelu
- **`noise_features.npy`**: Przykładowe cechy wyekstrahowane

### Proces Treningowy

1. **Dane**: Dataset OpenFake (ComplexDataLab/OpenFake)
2. **Transformacje**: Standardowe augmentacje (rozmycie, kompresja, szum)
3. **Funkcja straty**: BCEWithLogitsLoss
4. **Optymalizator**: Adam (lr=1e-4)
5. **Metryki**: ROC-AUC, F1-Score
6. **Zapis**: Model jest zapisywany gdy ROC-AUC się poprawia

### Użycie

```python
from src.noise.model import NoiseBinaryClassifier

model = NoiseBinaryClassifier()
model.load_state_dict(torch.load("best_noise_model.pt"))
logits = model(image_tensor)  # Przewidywanie
```

## Model RGB (Analiza Cech Wizualnych)

### Opis Działania

Model RGB wykorzystuje wstępnie wytrenowany EfficientNet-B0 jako ekstraktor cech, a następnie dodaje prostą głowę klasyfikacyjną. Analizuje cechy wizualne na poziomie semantycznym.

### Architektura

1. **Backbone**: EfficientNet-B0 (timm)
   - Wstępnie wytrenowany na ImageNet
   - `num_classes=0` - usunięta warstwa klasyfikacyjna
   - Zwraca wektor cech o długości 1280

2. **Klasyfikator**:
   - Dropout (0.2)
   - Warstwa liniowa: 1280 → 1

### Pliki Powiązane

- **`src/rgb/feature_extractor.py`**: Tworzenie ekstraktora cech
- **`src/rgb/data.py`**: Dataset OpenFake z transformacjami
- **`src/rgb/train.py`**: Skrypt treningowy RGB
- **`src/rgb/inference.py`**: Inferencja dla pojedynczych obrazów
- **`src/rgb/explain.py`**: Explainable AI z Grad-CAM
- **`best_rgb_model.pt`**: Zapisane wagi najlepszego modelu

### Proces Treningowy

1. **Dane**: Dataset OpenFake w trybie streaming
2. **Transformacje**:
   - Resize do 224x224
   - Augmentacje: kompresja, rozmycie Gaussa, szum
   - Normalizacja ImageNet
3. **Funkcja straty**: BCEWithLogitsLoss
4. **Optymalizator**: Adam (lr=1e-4)
5. **Mixed Precision**: Używany gdy dostępna GPU
6. **TensorBoard**: Logowanie metryk

### Inferencja

```bash
python src/rgb/inference.py test_image.jpg --model_path best_rgb_model.pt
```

### Explainable AI (Grad-CAM)

Model RGB wspiera wizualizację ważnych regionów obrazu poprzez Grad-CAM:
- Cel: warstwa `conv_head` EfficientNet
- Heatmap nakładana na oryginalny obraz
- Zapis do `gradcam_output.jpg`

## Dataset OpenFake

### Charakterystyka

- **Źródło**: ComplexDataLab/OpenFake (Hugging Face)
- **Tryb**: Streaming - dane ładowane na bieżąco
- **Klasy**: Real (0) vs Fake (1)
- **Rozmiar**: ~100k obrazów treningowych

### Przetwarzanie

```python
from src.rgb.data import OpenFakeDataset, get_transforms

transforms = get_transforms()
dataset = OpenFakeDataset(split="train", transform=transforms)
```

### Transformacje

1. **Augmentacje treningowe**:
   - ImageCompression (60-100% jakości)
   - GaussianBlur (3-7 kernel)
   - GaussNoise (var 10-50)

2. **Preprocessing**:
   - Resize 224x224
   - Normalizacja (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

## Modele Dodatkowe

### CLIP-based Model

Lokalizacja: `src/models/clip/`

- **Architektura**: Fine-tuning CLIP (ViT-Base-Patch32)
- **Zastosowanie**: Klasyfikacja semantyczna
- **Pliki**: `semantic_judge.py`, `train_clip.py`, `evaluate_clip.py`

### FFT Detector

Lokalizacja: `src/models/fft_detector/`

- **Architektura**: Analiza częstotliwościowa (FFT)
- **Zastosowanie**: Wykrywanie artefaktów w dziedzinie częstotliwości
- **Pliki**: `model.py`, `train.py`, `predict_single.py`

## Środowisko i Zależności

### Główne Biblioteki

- **PyTorch**: Framework deep learning
- **timm**: Pretrained models (EfficientNet)
- **albumentations**: Augmentacje obrazów
- **datasets**: Ładowanie danych z Hugging Face
- **scikit-learn**: Metryki ewaluacyjne
- **captum**: Explainable AI

### Instalacja

```bash
conda env create -f environment.yml
# lub
pip install -r requirements.txt
```

## Trening i Ewaluacja

### Uruchomienie Treningu

```bash
# Model noise
python src/noise/train.py

# Model RGB
python src/rgb/train.py

# Model CLIP
python src/models/clip/train_clip.py
```

### Monitorowanie

- **TensorBoard**: Logi w `runs/noise_experiment_1/` i `runs/rgb_experiment_1/`
- **Metryki**: Loss, F1-Score, ROC-AUC
- **Checkpointing**: Automatyczny zapis najlepszych modeli

### Ewaluacja

```bash
# Inferencja na pojedynczym obrazie
python src/rgb/inference.py test_image.jpg

# Explainable AI
python src/rgb/explain.py
```

## Wyniki i Metryki

### Kluczowe Metryki

1. **ROC-AUC**: Powierzchnia pod krzywą ROC
2. **F1-Score**: Średnia harmoniczna precyzji i recall
3. **Loss**: Binary Cross Entropy z logits

### Strategia Walidacji

- **Dataset**: Podział train/test z OpenFake
- **Early Stopping**: Na podstawie ROC-AUC
- **Checkpointing**: Zapis przy poprawie ROC-AUC

## Problemy i Rozwiązania

### Wyzwania

1. **Streaming danych**: Brak shuffle w strumieniu
   - Rozwiązanie: Implementacja buffer shuffle

2. **Mixed Precision**: Stabilność numeryczna
   - Rozwiązanie: Użycie GradScaler z autocast

3. **Explainable AI**: Dostępność attention maps
   - Rozwiązanie: Grad-CAM zamiast attention

4. **Cache danych**: Powolne ładowanie
   - Rozwiązanie: Lokalne cache dla często używanych danych

### Optymalizacje

1. **Batch Size**: 32 dla kompromisu pamięć/szybkość
2. **Num Workers**: 0 dla Windows (kompatybilność)
3. **Mixed Precision**: Aktywowane dla GPU
4. **Gradient Accumulation**: Możliwość implementacji dla większych batchy

## Przyszły Rozwój

### Planowane Funkcjonalności

1. **Ensemble modeli**: Łączenie predykcji noise i RGB
2. **Multi-modal detection**: Dodanie analizy metadanych
3. **Real-time detection**: API REST dla strumienia wideo
4. **Extended datasets**: Więcej źródeł danych treningowych

### Ulepszenia Techniczne

1. **Distributed training**: Trening wielo-GPU
2. **Quantization**: Optymalizacja dla deployment
3. **ONNX export**: Kompatybilność między frameworkami
4. **CI/CD**: Automatyzacja testów i deploymentu

## Wnioski

Projekt implementuje kompleksowy system detekcji deepfake wykorzystujący różne podejścia:

1. **Model Noise**: Skuteczny w wykrywaniu artefaktów szumowych
2. **Model RGB**: Skupia się na cechach semantycznych
3. **Modularna architektura**: Łatwe dodawanie nowych modeli
4. **Explainable AI**: Przejrzystość decyzji modelu

System jest gotowy do dalszego rozwoju i integracji w aplikacjach produkcyjnych do detekcji obrazów wygenerowanych przez AI.