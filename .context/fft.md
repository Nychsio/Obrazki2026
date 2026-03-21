# Moduł FFT Detector - Dokumentacja Techniczna

> **Źródło prawdy**: Ta dokumentacja została wygenerowana na podstawie analizy faktycznego kodu źródłowego projektu i zastępuje wszelkie wcześniejsze założenia.

## Przegląd

Moduł `fft_detector` to detektor obrazów generowanych przez AI (deepfake) oparty na analizie widma Fouriera. Wykorzystuje **Fast Fourier Transform (FFT)** do ekstrakcji cech częstotliwościowych, które następnie klasyfikuje sieć neuronowa oparta na architekturze ResNet18.

### Główna idea

Obrazy generowane przez AI (np. przez GAN-y, diffusion models) często zawierają charakterystyczne artefakty w wysokich częstotliwościach, które są niewidoczne gołym okiem, ale wyraźnie ujawniają się w widmie Fouriera. Ten detektor wykorzystuje tę właściwość do klasyfikacji binarnej: **Fake vs Real**.

---

## Architektura

### Schemat przepływu danych

```
[Obraz RGB] 
    ↓
[Resize 224x224]
    ↓
[Konwersja do skali szarości]
    ↓
[2D FFT (torch.fft.fft2)]
    ↓
[Shift zerowej częstotliwości (fftshift)]
    ↓
[Magnitude (wartość bezwzględna)]
    ↓
[Skala logarytmiczna]
    ↓
[Normalizacja min-max → [0,1]]
    ↓
[ResNet18 (1 kanał wejściowy)]
    ↓
[Klasyfikacja binarna (Sigmoid)]
```

---

## Komponenty

### 1. `transforms.py` - Transformacja Fouriera

**Klasa:** `FourierMagnitudeTransform`

Konwertuje obraz RGB na znormalizowane widmo amplitudowe Fouriera.

#### Algorytm krok po kroku:

| Krok | Operacja | Implementacja |
|------|----------|---------------|
| 1 | Konwersja RGB → grayscale | `torchvision.transforms.functional.rgb_to_grayscale()` |
| 2 | Konwersja na tensor | `F.to_tensor()` → kształt `[1, H, W]` |
| 3 | 2D FFT | `torch.fft.fft2()` |
| 4 | Shift DC na środek | `torch.fft.fftshift()` |
| 5 | Obliczenie magnitudy | `torch.abs()` |
| 6 | Skala logarytmiczna | `torch.log(magnitude + 1e-8)` |
| 7 | Normalizacja min-max | `(x - min) / (max - min + eps)` → zakres `[0, 1]` |

#### Parametry i optymalizacje:

- **Epsilon** (`1e-8`): Zapobiega `log(0)` i dzieleniu przez zero
- **Wyjście**: Tensor `[1, H, W]` w zakresie `[0, 1]`
- **Edge case**: Jeśli `max == min`, zwraca tensor zerowy

```python
# Kluczowy fragment kodu
fft_complex = torch.fft.fft2(tensor_img)
fft_shifted = torch.fft.fftshift(fft_complex)
magnitude = torch.abs(fft_shifted)
magnitude_log = torch.log(magnitude + 1e-8)
```

---

### 2. `model.py` - Architektura sieci

**Klasa:** `FFTResNetDetector`

Klasyfikator binarny oparty na pretrenowanym ResNet18.

#### Modyfikacje względem standardowego ResNet18:

| Warstwa | Oryginalna | Zmodyfikowana |
|---------|------------|---------------|
| `conv1` | 3 kanały RGB | 1 kanał (grayscale FFT) |
| `fc` | 1000 klas (ImageNet) | 1 klasa (binarna) |

#### Transfer Learning:

```python
# Zachowanie wiedzy z ImageNet przez uśrednienie wag po kanałach
self.backbone.conv1.weight = nn.Parameter(
    torch.mean(original_conv1.weight, dim=1, keepdim=True)
)
```

#### Parametry modelu:

- **Backbone**: `resnet18` z wagami `ResNet18_Weights.DEFAULT` (pretrenowany na ImageNet)
- **Wejście**: `[B, 1, 224, 224]` - widmo Fouriera
- **Wyjście**: `[B, 1]` - logit (przed sigmoid)
- **Interpretacja**: `sigmoid(output) > 0.5` → Fake, inaczej → Real

---

### 3. `train.py` / `poc_train.py` - Trening

#### Hiperparametry:

| Parametr | `train.py` | `poc_train.py` (PoC) |
|----------|------------|----------------------|
| Batch size | 32 | 16 |
| Learning rate | 0.001 | 0.001 |
| Optymalizator | Adam | Adam |
| Funkcja straty | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Epoki | 15 | 3 |
| Próbki | pełny dataset | 1000 (streaming) |

#### Device Agnostic Code:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon
else:
    device = torch.device("cpu")
```

#### Checkpointing:

- `train.py`: `fft_detector_epoch_{N}.pth` - tylko `state_dict`
- `poc_train.py`: `poc_fft_checkpoint.pth` - słownik z `model_state_dict` i `loss`

---

### 4. `predict_single.py` - Predykcja

Skrypt do klasyfikacji pojedynczego obrazu.

#### Pipeline:

1. Wczytanie obrazu (`PIL.Image`)
2. Resize do 224x224
3. Transformacja FFT (`FourierMagnitudeTransform`)
4. Dodanie wymiaru batcha: `[1, 1, 224, 224]`
5. Forward przez model
6. Sigmoid → prawdopodobieństwo
7. Próg 0.5 → klasyfikacja

#### Wyjście:

```
ANALIZA OBRAZU: test_image.jpg
WERDYKT: FAKE (AI Generated)
PEWNOŚĆ: 87.34%
```

---

### 5. `poc_test_xai.py` - Explainable AI

Wizualizacja Grad-CAM pokazująca, które regiony widma Fouriera wpłynęły na decyzję modelu.

#### Użyte biblioteki:

- `pytorch_grad_cam` (Grad-CAM)
- `opencv-python` (cv2)
- `matplotlib`

#### Warstwa docelowa:

```python
target_layers = [model.backbone.layer4[-1]]  # Ostatni blok ResNet
```

#### Wyjście:

Plik `portfolio_xai_result.png` z dwoma panelami:
1. Widmo Fouriera (grayscale)
2. Mapa Grad-CAM nałożona na widmo

---

### 6. `data_loader.py` - Ładowanie danych

**Funkcja:** `get_streaming_dataloader(batch_size=32)`

#### Źródło danych:

- **Dataset**: `ComplexDataLab/OpenFake` (Hugging Face)
- **Split**: `train`
- **Tryb**: streaming (nie pobiera całego datasetu)
- **Limit**: 1000 próbek (PoC)

#### Mapowanie etykiet:

| String | Wartość numeryczna |
|--------|-------------------|
| `"fake"` / `"generated"` | 1 |
| `"real"` / `"authentic"` | 0 |

---

## Wnioski z kodu

### Faktyczna optymalizacja FFT:

1. **Biblioteka**: `torch.fft` (natywny PyTorch, GPU-accelerated)
   - NIE używa `numpy.fft` ani `scipy.fft`
   - Pełna kompatybilność z GPU (CUDA) i autograd

2. **Optymalizacje w transformacji**:
   - Konwersja na grayscale przed FFT (redukcja obliczeń 3x)
   - Epsilon w log i normalizacji dla stabilności numerycznej
   - Min-max normalizacja dla stabilnego treningu sieci

3. **Transfer Learning**:
   - Uśrednienie wag `conv1` zamiast losowej inicjalizacji
   - Zachowuje wiedzę z ImageNet dla ekstrakcji cech

### Gdzie FFT jest wywoływane:

| Lokalizacja | Kontekst |
|-------------|----------|
| `transforms.py:FourierMagnitudeTransform.__call__()` | **Główna implementacja** |
| `data_loader.py:apply_transforms()` | Podczas ładowania batchy |
| `predict_single.py:preprocess` | Predykcja pojedynczego obrazu |
| `smoke_test.py:transform()` | Testy jednostkowe |

### Parametry kluczowe:

| Parametr | Wartość | Lokalizacja |
|----------|---------|-------------|
| Rozmiar wejścia | 224×224 | `transforms.Resize()` |
| Epsilon (log) | 1e-8 | `FourierMagnitudeTransform` |
| Epsilon (normalizacja) | 1e-8 | `FourierMagnitudeTransform` |
| Próg klasyfikacji | 0.5 | `predict_single.py`, `poc_test_xai.py` |

---

## Brakujące elementy / Możliwe rozszerzenia

> **Uwaga**: Oryginalny plik `.context/FFT.md` był **pusty**, więc nie można zweryfikować założeń Gemini. Poniżej lista potencjalnych rozszerzeń na podstawie dobrych praktyk:

### Niezaimplementowane (potencjalne TODO):

1. **Augmentacje danych** - brak augmentacji w pipeline treningu
2. **Validation split** - brak podziału na train/val, tylko pełny trening
3. **Early stopping** - brak mechanizmu wczesnego zatrzymania
4. **Scheduler LR** - brak harmonogramu uczenia (np. CosineAnnealing)
5. **Metryki** - tylko accuracy i loss, brak precision/recall/F1/AUC
6. **Logging** - brak integracji z TensorBoard/W&B
7. **Konfiguracja** - hiperparametry hardcoded w kodzie
8. **Testy jednostkowe** - tylko `smoke_test.py`, brak pytest

### Architektura - możliwe ulepszenia:

1. Użycie widma fazowego oprócz magnitudy
2. Multi-scale FFT (różne rozdzielczości)
3. Azymutalna analiza widma (jak w DIRE paper)
4. Ensemble z innymi detektorami (np. CLIP)

---

## Zależności

```
torch
torchvision
datasets (Hugging Face)
pytorch-grad-cam
opencv-python
matplotlib
numpy
Pillow
tqdm
```

---

## Przykłady użycia

### Trening PoC:

```bash
python -m src.models.fft_detector.poc_train
```

### Predykcja pojedynczego obrazu:

```bash
python -m src.models.fft_detector.predict_single
```

### Generowanie wizualizacji XAI:

```bash
python -m src.models.fft_detector.poc_test_xai
```

### Smoke test:

```bash
python src/models/fft_detector/smoke_test.py
```

---

## Pliki checkpointów

| Plik | Opis |
|------|------|
| `poc_fft_checkpoint.pth` | Wagi z treningu PoC (3 epoki, 1000 próbek) |
| `fft_detector_epoch_{N}.pth` | Wagi z pełnego treningu (epoka N) |

---

*Dokumentacja wygenerowana automatycznie na podstawie analizy kodu źródłowego.*
*Ostatnia aktualizacja: 2026-03-21*
