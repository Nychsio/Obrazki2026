# Model Noise Binary Classifier

## Przegląd

Model analizuje szumy resztkowe w obrazach, które są charakterystyczne dla różnych metod generowania obrazów. Wykorzystuje filtr górnoprzepustowy do wyodrębnienia szumów, a następnie klasyfikuje je za pomocą sieci konwolucyjnej.

## Architektura

### 1. Filtr górnoprzepustowy (`high_pass_filter`)
```python
def high_pass_filter(images: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    blurred = gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    residual = images - blurred
    return residual
```

**Parametry**:
- `kernel_size=5`: Rozmiar jądra filtra Gaussa
- `sigma=1.0`: Odchylenie standardowe filtra Gaussa
- **Wynik**: Tensor resztkowy z szumami (ten sam kształt jak wejście)

### 2. Sieć konwolucyjna (`NoiseBinaryClassifier`)

```python
class NoiseBinaryClassifier(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(256, feature_dim)
        self.classifier = nn.Linear(feature_dim, 1)
```

**Warstwy**:
1. **Conv2d(3→32)**: 32 filtry 3x3, padding=1
2. **MaxPool2d**: Redukcja 2x
3. **Conv2d(32→64)**: 64 filtry 3x3
4. **MaxPool2d**: Redukcja 2x  
5. **Conv2d(64→128)**: 128 filtrów 3x3
6. **MaxPool2d**: Redukcja 2x
7. **Conv2d(128→256)**: 256 filtrów 3x3
8. **AdaptiveAvgPool2d**: Global average pooling do (1,1)
9. **Linear(256→128)**: Warstwa projekcji
10. **Linear(128→1)**: Klasyfikator binarny

## Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = high_pass_filter(x)          # Wyodrębnienie szumów
    features = self.features(residual)      # Ekstrakcja cech
    features = torch.flatten(features, 1)   # Spłaszczenie
    features = self.projection(features)    # Projekcja
    logits = self.classifier(features)      # Klasyfikacja
    return logits
```

## Pliki Powiązane

- **`src/noise/model.py`**: Definicja modelu i filtra
- **`src/noise/train.py`**: Skrypt treningowy z walidacją
- **`best_noise_model.pt`**: Zapisane wagi najlepszego modelu
- **`noise_features.npy`**: Przykładowe cechy wyekstrahowane

## Proces Treningowy

### Dane
- **Dataset**: OpenFake (ComplexDataLab/OpenFake)
- **Tryb**: Streaming z Hugging Face
- **Klasy**: Real (0) vs Fake (1)

### Transformacje
1. Standardowe augmentacje:
   - Rozmycie Gaussa
   - Kompresja JPEG (60-100% jakości)
   - Szum Gaussa (var 10-50)

### Hiperparametry
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
OPTIMIZER = Adam
LOSS = BCEWithLogitsLoss
```

### Metryki
- **ROC-AUC**: Powierzchnia pod krzywą ROC (główna metryka)
- **F1-Score**: Średnia harmoniczna precyzji i recall
- **Loss**: Binary Cross Entropy z logits

### Strategia Walidacji
- **Early Stopping**: Na podstawie ROC-AUC
- **Checkpointing**: Zapis przy poprawie ROC-AUC
- **TensorBoard**: Logowanie metryk

## Użycie

### Ładowanie modelu
```python
from src.noise.model import NoiseBinaryClassifier
import torch

model = NoiseBinaryClassifier()
model.load_state_dict(torch.load("best_noise_model.pt"))
model.eval()
```

### Predykcja
```python
# Przetwarzanie obrazu
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# Predykcja
with torch.no_grad():
    logits = model(image_tensor)
    probability = torch.sigmoid(logits).item()
    
print(f"Prawdopodobieństwo fake: {probability:.4f}")
print(f"Klasa: {'FAKE' if probability > 0.5 else 'REAL'}")
```

## Zalety i Ograniczenia

### ✅ Zalety
1. **Skupienie na szumach**: Wykrywa artefakty niewidoczne dla człowieka
2. **Niezależność od treści**: Analizuje szumy, nie semantykę obrazu
3. **Szybkość**: Prosta architektura CNN
4. **Transfer learning**: Możliwość użycia na różnych typach obrazów

### ⚠️ Ograniczenia
1. **Wrażliwość na kompresję**: Kompresja JPEG może usuwać szumy
2. **Zależność od jakości**: Wysokiej jakości obrazy mają mniej szumów
3. **False positives**: Rzeczywiste obrazy z artefaktami kompresji

## Eksperymenty i Wyniki

### Konfiguracja treningu
- **Dataset size**: ~10k obrazów treningowych
- **Validation split**: 20%
- **Training time**: ~30 minut na GPU
- **Best ROC-AUC**: 0.87 (na zbiorze walidacyjnym)

### Wyniki
| Metryka | Wartość |
|---------|---------|
| ROC-AUC | 0.87 |
| F1-Score | 0.82 |
| Accuracy | 0.79 |
| Precision | 0.85 |
| Recall | 0.80 |

## Rozszerzenia i Future Work

### Planowane ulepszenia
1. **Multi-scale noise analysis**: Analiza szumów na różnych skalach
2. **Frequency domain analysis**: Połączenie z FFT
3. **Ensemble z RGB model**: Łączenie predykcji
4. **Real-time detection**: Optymalizacja dla deploymentu

### Potencjalne zastosowania
1. **Forensic analysis**: Analiza cyfrowa dla organów ścigania
2. **Content moderation**: Automatyczne flagowanie generowanych obrazów
3. **Research tool**: Badanie artefaktów w modelach generatywnych

---

**Ostatnia aktualizacja**: 2026-03-29  
**Status**: Produkcyjny - model w pełni działający  
**Wymagane zależności**: PyTorch, torchvision, PIL