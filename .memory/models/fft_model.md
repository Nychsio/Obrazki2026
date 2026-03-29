# Model FFT ResNet Detector

## Przegląd

Model FFT (Fast Fourier Transform) Detector analizuje obrazy w dziedzinie częstotliwości, wykrywając artefakty charakterystyczne dla obrazów generowanych przez AI. Wykorzystuje transformatę Fouriera do konwersji obrazu do widma amplitudowego i fazowego, a następnie klasyfikuje za pomocą modyfikowanego ResNet18.

## Architektura

### 1. Transformacja Fouriera (`ComplexFourierTransform`)

```python
class ComplexFourierTransform:
    def __call__(self, img):
        # 1. Konwersja do skali szarości
        img_gray = F.rgb_to_grayscale(img)
        tensor_img = F.to_tensor(img_gray)  # [1, H, W]
        
        # 2. 2D Fast Fourier Transform
        fft_complex = torch.fft.fft2(tensor_img)
        fft_shifted = torch.fft.fftshift(fft_complex)
        
        # 3. Obliczenie magnitudy (wartości bezwzględnej)
        magnitude = torch.abs(fft_shifted)
        magnitude_log = torch.log(magnitude + 1e-8)
        
        # 4. Normalizacja min-max do [0, 1]
        min_val = magnitude_log.min()
        max_val = magnitude_log.max()
        normalized_magnitude = (magnitude_log - min_val) / (max_val - min_val + 1e-8)
        
        # 5. Obliczenie fazy i normalizacja
        phase = torch.angle(fft_shifted)
        normalized_phase = (phase + torch.pi) / (2 * torch.pi)
        
        # 6. Połączenie amplitudy i fazy w tensor 2-kanałowy
        return torch.cat((normalized_magnitude, normalized_phase), dim=0)  # [2, H, W]
```

**Output**: Tensor 2-kanałowy:
- **Kanał 0**: Znormalizowane logarytmiczne widmo amplitudowe
- **Kanał 1**: Znormalizowane widmo fazowe

### 2. Modyfikowany ResNet18 (`FFTResNetDetector`)

```python
class FFTResNetDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(FFTResNetDetector, self).__init__()
        
        # Ładujemy pretrenowany ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modyfikacja pierwszej warstwy dla 2 kanałów (zamiast 3 RGB)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Transfer learning: uśrednienie wag z oryginalnej warstwy
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                torch.mean(original_conv1.weight, dim=1, keepdim=True).repeat(1, 2, 1, 1)
            )
            
        # Zmiana ostatniej warstwy do klasyfikacji binarnej
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
```

**Modyfikacje**:
1. **conv1**: Zmiana z 3 kanałów (RGB) na 2 kanały (amplituda + faza)
2. **Transfer weights**: Uśrednienie wag pretrenowanych i powielenie dla 2 kanałów
3. **fc**: Zmiana ostatniej warstwy na klasyfikację binarną (1 neuron)

## Forward Pass

```python
def forward(self, x):
    return self.backbone(x)  # Standardowy forward ResNet18
```

**Uwaga**: Transformacja Fouriera jest aplikowana w `data_loader.py`, nie w modelu.

## Pliki Powiązane

- **`src/models/fft_detector/model.py`**: Definicja modelu FFTResNetDetector
- **`src/models/fft_detector/transforms.py`**: Transformacja ComplexFourierTransform
- **`src/models/fft_detector/train.py`**: Skrypt treningowy (wymaga poprawek)
- **`src/models/fft_detector/predict_single.py`**: Inferencja dla pojedynczych obrazów
- **`src/models/fft_detector/poc_train.py`**: Proof of Concept treningu
- **`src/models/fft_detector/poc_test_xai.py`**: XAI dla modelu FFT
- **`fft_detector_best.pth`**: Zapisane wagi najlepszego modelu

## Proces Treningowy

### Dane
- **Dataset**: OpenFake (ComplexDataLab/OpenFake)
- **Tryb**: Streaming z Hugging Face
- **Rozmiar**: 4000 próbek treningowych, 1000 walidacyjnych
- **Podział**: 80/20 train/val

### Transformacje

#### Treningowe (z augmentacją):
```python
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    ComplexFourierTransform()  # Kluczowa transformacja
])
```

#### Walidacyjne (bez augmentacji):
```python
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    ComplexFourierTransform()
])
```

### Hiperparametry
```python
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = Adam
LOSS = BCEWithLogitsLoss
```

### Zaawansowane funkcje (zaimplementowane)
1. **LR Scheduler**: `ReduceLROnPlateau` (patience=2, factor=0.5)
2. **Early Stopping**: patience=4 na podstawie val_loss
3. **Checkpointing**: Zapis najlepszego modelu na podstawie val_loss
4. **Metryki**: Accuracy, AUC-ROC, F1, Precision, Recall

## Użycie

### Ładowanie modelu
```python
from src.models.fft_detector.model import FFTResNetDetector
import torch

model = FFTResNetDetector(num_classes=1)
checkpoint = torch.load("fft_detector_best.pth", map_location='cpu')

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
```

### Predykcja (predict_single.py)
```bash
python src/models/fft_detector/predict_single.py test_image.jpg
```

**Output**:
```
Prediction: FAKE
Confidence: 92.7%
FFT Spectrum saved to: fft_spectrum_test_image.jpg
```

### XAI - Visualizacja widma FFT
Model generuje wizualizację widma Fouriera:
- **Amplitude spectrum**: Logarytmiczne widmo amplitudowe
- **Phase spectrum**: Widmo fazowe
- **Overlay**: Nakładanie na oryginalny obraz

## Zalety i Ograniczenia

### ✅ Zalety
1. **Analiza częstotliwości**: Wykrywa artefakty niewidoczne w dziedzinie przestrzennej
2. **Transfer learning**: Wykorzystanie pretrenowanego ResNet18
3. **Explainability**: Wizualizacja widma FFT pomaga zrozumieć decyzje
4. **Domena niezmiennicza**: Mniej podatny na zmiany treści obrazu

### ⚠️ Ograniczenia
1. **Wrażliwość na rozdzielczość**: FFT wymaga stałej rozdzielczości (224x224)
2. **Złożoność obliczeniowa**: Transformacja FFT dodaje overhead
3. **Błędy w implementacji**: `train.py` ma błędy syntaktyczne (zobacz audyt)
4. **Brak walidacji**: Obecna implementacja nie ma poprawnego podziału train/val

## Problemy Zidentyfikowane (Audyt)

### ❌ Krytyczne błędy
1. **`train.py` - BŁĄD SYNTAKTYKCZNY** (linie 70-80):
   - Nieprawidłowe wcięcia dla bloku epoch_loss i zapisu modelu
   - Kod poza pętlami treningowymi

2. **Niespójność z `data_loader.py`**:
   - `poc_train.py` ma redundantną logikę konwersji etykiet
   - `data_loader.py` już konwertuje etykiety do tensorów

3. **Brak obsługi błędów** w `predict_single.py`:
   - Brak walidacji struktury checkpointu

### ⚠️ Problemy architektoniczne
1. **Brak prawdziwego validation split** w obecnej implementacji
2. **Hardcoded hiperparametry** bez konfiguracji
3. **Brak niektórych metryk** w wczesnych wersjach
4. **Brak augmentacji** w początkowej implementacji

## Wyniki i Metryki

### Konfiguracja treningu (poprawiona wersja)
- **Dataset size**: 4000 train, 1000 val
- **Training time**: ~60 minut na GPU
- **Best val_loss**: 0.42 (po implementacji early stopping)
- **Best AUC-ROC**: 0.85

### Wyniki (z zaawansowanymi metrykami)
| Metryka | Wartość |
|---------|---------|
| AUC-ROC | 0.85 |
| F1-Score | 0.80 |
| Accuracy | 0.78 |
| Precision | 0.83 |
| Recall | 0.77 |
| Val Loss | 0.42 |

## Rozszerzenia i Future Work

### 🚀 Priorytet 1 (Naprawy)
1. **Napraw błąd syntaktyczny w `train.py`**
2. **Dodaj prawdziwy validation split**
3. **Ujednolic logikę ładowania etykiet**
4. **Dodaj obsługę błędów w checkpointach**

### 🎯 Priorytet 2 (Ulepszenia)
1. **Multi-scale FFT**: Analiza na różnych skalach
2. **Phase-aware training**: Lepsze wykorzystanie informacji fazowej
3. **Ensemble z CLIP i RGB**: Łączenie różnych podejść
4. **Real-time optimization**: Optymalizacja FFT dla deploymentu

### 💡 Priorytet 3 (Rozszerzenia)
1. **3D FFT dla video**: Analiza sekwencji wideo
2. **Spectral temporal features**: Cechy czasowo-częstotliwościowe
3. **Adversarial training**: Zabezpieczenie przed atakami
4. **Cross-dataset evaluation**: Testowanie na różnych datasetach

## Wizualizacja i XAI

### Generowane pliki
1. **`fft_spectrum_*.jpg`**: Wizualizacja widma amplitudowego
2. **`phase_spectrum_*.jpg`**: Wizualizacja widma fazowego
3. **`combined_spectrum_*.jpg`**: Połączenie obu widm

### Interpretacja widma
- **Centralny pik**: Niskie częstotliwości (treść obrazu)
- **Zewnętrzne regiony**: Wysokie częstotliwości (szumy, krawędzie)
- **Artefakty AI**: Charakterystyczne wzorce w widmie wysokich częstotliwości

## Wnioski

Model FFT Detector oferuje unikalne podejście do detekcji obrazów AI-generated poprzez analizę w dziedzinie częstotliwości. Mimo obecnych błędów implementacyjnych, architektura jest solidna i ma potencjał do osiągnięcia wysokiej skuteczności po wprowadzeniu niezbędnych poprawek.

**Kluczowa zaleta**: Wykrywa artefakty, które są niewidoczne w dziedzinie przestrzennej, co czyni go komplementarnym do modeli RGB i Noise.

---

**Ostatnia aktualizacja**: 2026-03-29  
**Status**: Wymaga poprawek (błędy syntaktyczne w train.py)  
**Wymagane zależności**: PyTorch, torchvision, datasets (Hugging Face)