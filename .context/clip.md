# Dokumentacja Techniczna: Moduł CLIP

## 1. Przegląd Techniczny

### Wersja modelu CLIP
- **Model podstawowy**: `openai/clip-vit-base-patch32`
- **Architektura**: Vision Transformer (ViT-Base) z patch size 32x32
- **Wymiary wejściowe**: 224x224 pikseli, 3 kanały RGB
- **Wymiary wyjściowe**: Wektor embedding o rozmiarze 768
- **Cel użycia**: Binary classification (Real vs Fake) dla wykrywania obrazów wygenerowanych przez AI i cyfrowych manipulacji

### Charakterystyka implementacji
- **Transfer Learning**: Zamrożony backbone CLIP (feature extraction)
- **Fine-tuning**: Tylko głowa klasyfikacyjna (linear layers)
- **Tryb pracy**: Vision-only (bez komponentu tekstowego CLIP)
- **Reprezentacja**: Użycie `pooler_output` z vision_model CLIP jako globalnej reprezentacji obrazu

## 2. Struktura Kodu

### Kluczowe moduły i klasy

#### `semantic_judge.py` - Główny model klasyfikacyjny
```python
class SemanticJudgeCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_backbone=True):
        # Inicjalizacja procesora CLIPProcessor
        # Ładowanie vision_model z CLIPModel
        # Konfiguracja głowy klasyfikacyjnej
        # Zamrażanie wag backbone (opcjonalnie)
    
    def forward(self, pixel_values):
        # Ekstrakcja cech przez vision_model
        # Pobranie pooler_output
        # Klasyfikacja przez sekwencję linear layers
```

**Architektura głowy klasyfikacyjnej**:
```
Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 1)
```

#### `clip_streamer.py` - Strumień danych
```python
class CLIPDataStreamer:
    def __init__(self, dataset_name="ComplexDataLab/OpenFake", 
                 model_name="openai/clip-vit-base-patch32", batch_size=32):
        # Inicjalizacja CLIPProcessor
        # Konfiguracja datasetu
    
    def get_stream(self, split="train"):
        # Ładowanie datasetu w trybie streaming
        # Shuffle dla danych treningowych
    
    def collate_fn(self, batch):
        # Przetwarzanie obrazów przez CLIPProcessor
        # Konwersja etykiet (string → float)
        # Przygotowanie tensorów
    
    def create_dataloader(self, split="train"):
        # Tworzenie DataLoader z funkcją collate
```

#### `train_clip.py` - Skrypt treningowy
```python
def train_clip():
    # Konfiguracja urządzenia (CUDA/CPU)
    # Inicjalizacja modelu i danych
    # Definicja loss (BCEWithLogitsLoss)
    # Optymalizator (AdamW) tylko dla klasyfikatora
    # Scheduler (CosineAnnealingLR)
    # Pętla treningowa z early stopping
    # Obliczanie metryk (ROC-AUC, F1-Score)
```

#### `evaluate_clip.py` - Ewaluacja modelu
```python
def evaluate_model(model_weights_path="checkpoints/clip_classifier_best.pth", test_steps=500):
    # Ładowanie wag modelu
    # Inicjalizacja strumienia testowego
    # Obliczanie metryk: ROC-AUC, F1-Score, Accuracy, Confusion Matrix
    # Generowanie raportu
```

#### `xai_clip.py` - Explainable AI
```python
class CLIPExplainer:
    def __init__(self, model_path="checkpoints/clip_classifier_best.pth"):
        # Ładowanie modelu
        # Inicjalizacja procesora
    
    def generate_heatmap(self, image_path, save_path="heatmap_output.jpg"):
        # Generowanie attention maps z ostatniej warstwy Transformer
        # Wizualizacja heatmap na oryginalnym obrazie
        # Dodawanie tekstu z wynikiem klasyfikacji
```

#### `test_integration.py` - Testy integracyjne
```python
def run_sanity_check():
    # Test forward pass
    # Overfitting na pojedynczym batchu
    # Weryfikacja kształtów tensorów
```

## 3. Pipeline Danych

### Przetwarzanie wstępne (Preprocessing)
1. **Ładowanie obrazu**: `PIL.Image.open().convert("RGB")`
2. **Transformacja przez CLIPProcessor**:
   - Resize do 224x224 pikseli
   - Normalizacja z użyciem mean/std specyficznych dla CLIP
   - Konwersja do tensora PyTorch
3. **Formatowanie batcha**: `[batch_size, 3, 224, 224]`

### Przetwarzanie etykiet
```python
# Konwersja string → float dla etykiet
if isinstance(val, str):
    labels.append(1.0 if "fake" in val.lower() or "ai" in val.lower() else 0.0)
else:
    labels.append(float(val))
```

### Źródło danych
- **Dataset**: `ComplexDataLab/OpenFake` (HuggingFace)
- **Tryb**: Streaming (bez cache'owania na dysku)
- **Splity**: train, validation, test
- **Struktura**: Obrazy + etykiety binarne (Real/Fake)

### Flow danych
```
OpenFake Dataset → Streaming Loader → CLIPProcessor → 
→ Batch Tensor → Model CLIP → Features → Classifier → Output
```

## 4. Zależności

### Biblioteki kluczowe dla modułu CLIP

#### Core ML Framework
- `torch>=2.2.0` - Podstawowy framework deep learning
- `torchvision>=0.17.0` - Przetwarzanie obrazów i transformacje

#### Model Hubs & Transformers
- `transformers>=4.38.2` - **KLUCZOWA**: Implementacja CLIPModel i CLIPProcessor
- `datasets>=2.18.0` - Ładowanie datasetu OpenFake w trybie streaming
- `huggingface-hub>=0.21.4` - Dostęp do modeli HuggingFace
- `safetensors>=0.4.0` - Bezpieczne ładowanie wag modelu

#### Computer Vision
- `opencv-python-headless` - Przetwarzanie obrazów dla XAI (heatmaps)
- `pillow>=10.2.0` - Operacje na obrazach PIL

#### Data Science & Utilities
- `scikit-learn>=1.4.1` - Metryki ewaluacyjne (ROC-AUC, F1-Score, Confusion Matrix)
- `numpy>=1.26.0` - Operacje numeryczne
- `matplotlib>=3.8.0` - Wizualizacja (używana pośrednio)
- `tqdm>=4.66.0` - Progress bars dla pętli treningowych

#### XAI & Interpretability
- `captum>=0.7.0` - Zadeklarowana, ale **NIE UŻYWANA** w obecnej implementacji
  - Obecna implementacja używa natywnych attention maps z CLIP
  - Captum byłby lepszy dla Grad-CAM, Integrated Gradients

### Specyficzne importy w kodzie

#### `semantic_judge.py`
```python
from transformers import CLIPModel, CLIPProcessor  # KLUCZOWE
import torch.nn as nn
```

#### `clip_streamer.py`
```python
from transformers import CLIPProcessor  # KLUCZOWE
from datasets import load_dataset  # KLUCZOWE
from torch.utils.data import DataLoader
```

#### `train_clip.py` / `evaluate_clip.py`
```python
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix  # KLUCZOWE
```

#### `xai_clip.py`
```python
import cv2  # KLUCZOWE dla heatmaps
from PIL import Image
```

### Wersje i kompatybilność
- **CLIPProcessor**: Wymaga transformers >=4.0.0 dla poprawnej obsługi CLIP
- **Streaming datasets**: Wymaga datasets >=2.0.0
- **Attention maps**: Zależne od implementacji vision_model w transformers

## 5. Integracje

### Brak integracji z bazami wektorowymi
**Analiza**: W obecnej implementacji **NIE MA** integracji z:
- FAISS (Facebook AI Similarity Search)
- ChromaDB
- Qdrant
- Pinecone

**Powód**: System działa jako klasyfikator binarny, a nie system retrieval/semantic search. Embeddingi są używane bezpośrednio do klasyfikacji, a nie do wyszukiwania podobieństw.

### Potencjalne rozszerzenia
1. **FAISS**: Do k-nearest neighbors na embeddingach
2. **Chroma**: Do przechowywania i wyszukiwania podobnych obrazów
3. **Weaviate**: Do graph-based similarity search

### Integracja z HuggingFace
- **Model Hub**: Pobieranie pre-trenowanego CLIP
- **Dataset Hub**: Ładowanie OpenFake dataset
- **SafeTensors**: Bezpieczne ładowanie wag

## 6. Uwagi Techniczne

### Zamrożenie wag (Freeze Backbone)
```python
if freeze_backbone:
    for param in self.clip.parameters():
        param.requires_grad = False
```
- **Zalety**: Szybszy trening, mniej pamięci, zapobiega catastrophic forgetting
- **Wady**: Ograniczona zdolność adaptacji do specyficznej domeny

### Reprezentacja obrazu
```python
pooled_output = outputs.pooler_output  # Wymaga weryfikacji
```
- **Uwaga**: Należy sprawdzić czy `vision_model` CLIP faktycznie zwraca `pooler_output`
- **Alternatywa**: `last_hidden_state[:, 0]` (CLS token)

### Tryb streaming danych
- **Zalety**: Brak limitów pamięci, praca z dużymi datasetami
- **Wady**: Brak cache'owania, powolne ładowanie przy każdym epokowaniu
- **Brak shuffle**: Krytyczny problem dla jakości treningu

### XAI Implementation
- **Metoda**: Attention maps z ostatniej warstły Transformer
- **Rozdzielczość**: 7x7 (49 patchy) → interpolacja do oryginalnego rozmiaru
- **Ograniczenia**: Niska rozdzielczość, artefakty interpolacji

## 7. Problemy i Limity

### Zidentyfikowane problemy
1. **Błąd w obliczeniach loss** w `evaluate_clip.py`
2. **Brak shuffle** dla danych treningowych
3. **Założenie o `pooler_output`** bez weryfikacji
4. **Brak obsługi edge cases** w confusion matrix
5. **Niska rozdzielczość heatmaps** (7x7 → interpolacja)

### Ograniczenia architektoniczne
1. **Vision-only**: Brak wykorzystania komponentu tekstowego CLIP
2. **Binary classification**: Ograniczenie do 2 klas
3. **Fixed input size**: 224x224, brak obsługi różnych rozdzielczości
4. **No data augmentation**: Brak augmentacji w pipeline'ie

## 8. Rekomendacje

### Krótkoterminowe poprawki
1. Naprawić obliczenia loss w `evaluate_clip.py`
2. Dodać shuffle do danych treningowych
3. Zweryfikować dostępność `pooler_output`
4. Dodać obsługę błędów w confusion matrix

### Długoterminowe ulepszenia
1. Zaimplementować augmentację danych (albumentations)
2. Dodać mixed precision training
3. Rozważyć unfreezing ostatnich warstw CLIP
4. Zaimplementować Grad-CAM z Captum zamiast attention maps
5. Dodać integrację z FAISS dla similarity search

---

**Ostatnia aktualizacja**: Analiza oparta na kodzie z `src/models/clip/`  
**Stan**: Implementacja działająca, ale wymagająca poprawek w kluczowych obszarach  
**Gotowość produkcyjna**: Średnia - wymaga walidacji założeń i naprawy błędów