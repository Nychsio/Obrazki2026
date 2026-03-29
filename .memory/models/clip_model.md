# Model Semantic Judge CLIP

## Przegląd

Model CLIP (Contrastive Language-Image Pre-training) wykorzystuje fine-tuning vision model z CLIP (ViT-Base-Patch32) z dodaną głową klasyfikacyjną do detekcji obrazów wygenerowanych przez AI. Model analizuje obrazy na poziomie semantycznym, wykorzystując wiedzę pretrenowaną na parach obraz-tekst.

## Architektura

### 1. Backbone: CLIP Vision Transformer

```python
class SemanticJudgeCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_backbone=True):
        super(SemanticJudgeCLIP, self).__init__()
        
        # Ładowanie modelu CLIP i procesora
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Ekstrakcja vision model z CLIP
        self.vision_model = self.clip.vision_model
        
        # Głowa klasyfikacyjna
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),    # CLIP ViT-Base output dimension: 768
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)       # Binary classification
        )
        
        # Zamrażanie backbone (opcjonalnie)
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False
```

**Parametry**:
- **Model**: `openai/clip-vit-base-patch32` (ViT-Base z patch size 32)
- **Output dimension**: 768 (wymiar embeddingu CLIP)
- **Freeze backbone**: Domyślnie zamrożony dla transfer learning

### 2. Forward Pass

```python
def forward(self, pixel_values):
    # Przejście przez vision model CLIP
    outputs = self.vision_model(pixel_values=pixel_values)
    
    # Użycie pooler_output jako globalnej reprezentacji obrazu
    # UWAGA: Wymaga weryfikacji czy vision_model zwraca pooler_output
    pooled_output = outputs.pooler_output
    
    # Klasyfikacja
    logits = self.classifier(pooled_output)
    return logits
```

**Uwaga**: Istnieje problem z założeniem, że `vision_model` zwraca `pooler_output`. Alternatywnie można użyć `last_hidden_state[:, 0]` (CLS token).

## Pliki Powiązane

- **`src/models/clip/semantic_judge.py`**: Główna definicja modelu
- **`src/models/clip/train_clip.py`**: Skrypt treningowy
- **`src/models/clip/evaluate_clip.py`**: Ewaluacja modelu
- **`src/models/clip/clip_streamer.py`**: Strumień danych
- **`src/models/clip/xai_clip.py`**: Explainable AI (attention maps)
- **`src/models/clip/test_integration.py`**: Testy integracyjne

## Proces Treningowy

### Dane
- **Dataset**: OpenFake (ComplexDataLab/OpenFake)
- **Tryb**: Streaming z Hugging Face
- **Klasy**: Real (0) vs Fake (1)
- **Problem**: Brak shuffle danych treningowych

### Przetwarzanie przez CLIPProcessor

```python
# W clip_streamer.py
def collate_fn(self, batch):
    images = [item['image'] for item in batch]
    
    # Przetwarzanie przez CLIPProcessor
    inputs = self.processor(
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    # Konwersja etykiet
    labels = []
    for item in batch:
        val = item['label']
        if isinstance(val, str):
            labels.append(1.0 if "fake" in val.lower() or "ai" in val.lower() else 0.0)
        else:
            labels.append(float(val))
    
    return {
        'pixel_values': inputs['pixel_values'],
        'labels': torch.tensor(labels).float().unsqueeze(1)
    }
```

### Hiperparametry
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
MAX_STEPS_PER_EPOCH = 1000  # Problem: fixed bez sprawdzania końca strumienia
OPTIMIZER = AdamW (tylko dla klasyfikatora)
LOSS = BCEWithLogitsLoss
SCHEDULER = CosineAnnealingLR
```

### Problemy w Implementacji

#### ❌ Krytyczne błędy:
1. **`evaluate_clip.py` - BŁĄD MATEMATYCZNY**:
   ```python
   avg_test_loss = test_loss / len(all_labels) * batch_size  # ŹLE
   # Powinno być: test_loss / (len(all_labels) / batch_size)
   ```

2. **Brak shuffle danych**:
   - Streaming dataset bez shuffle prowadzi do overfittingu na kolejności

3. **Założenie o `pooler_output`**:
   - Niezweryfikowane czy `vision_model` CLIP zwraca `pooler_output`

4. **Fixed steps per epoch**:
   - `max_steps_per_epoch=1000` bez sprawdzania `StopIteration`

## Użycie

### Ładowanie modelu
```python
from src.models.clip.semantic_judge import SemanticJudgeCLIP
import torch

model = SemanticJudgeCLIP(freeze_backbone=True)
checkpoint = torch.load("checkpoints/clip_classifier_best.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Predykcja
```python
from PIL import Image
from src.models.clip.semantic_judge import SemanticJudgeCLIP

model = SemanticJudgeCLIP()
image = Image.open("test_image.jpg").convert("RGB")

# Przetwarzanie przez CLIPProcessor
inputs = model.processor(images=image, return_tensors="pt")

# Predykcja
with torch.no_grad():
    logits = model(inputs['pixel_values'])
    probability = torch.sigmoid(logits).item()

print(f"Prawdopodobieństwo fake: {probability:.4f}")
```

### XAI - Attention Maps
```bash
python src/models/clip/xai_clip.py --image_path test_image.jpg
```

**Generuje**: `heatmap_output.jpg` - attention maps z ostatniej warstwy Transformer nakładane na obraz.

## Zalety i Ograniczenia

### ✅ Zalety
1. **Wiedza semantyczna**: CLIP pretrenowany na 400M par obraz-tekst
2. **Transfer learning**: Możliwość fine-tuningu na specyficznej domenie
3. **Multi-modal potential**: Możliwość rozszerzenia o komponent tekstowy
4. **Attention visualization**: Natywne attention maps do interpretacji

### ⚠️ Ograniczenia
1. **Błędy implementacyjne**: Krytyczne błędy w obliczeniach loss
2. **Brak shuffle**: Streaming danych bez shuffle
3. **Założenia API**: Niezweryfikowane założenia o outputach CLIP
4. **Rozdzielczość heatmaps**: Niska rozdzielczość (7x7) po interpolacji

## Wyniki i Metryki

### Konfiguracja treningu (obecna)
- **Dataset**: OpenFake streaming
- **Training samples**: ~32k (max_steps=1000, batch_size=32)
- **Validation samples**: ~3.2k
- **Training time**: ~90 minut na GPU
- **Freeze backbone**: Tak (tylko klasyfikator trenowany)

### Wyniki (przed poprawkami)
| Metryka | Wartość (obecna) | Uwagi |
|---------|------------------|-------|
| ROC-AUC | 0.82 | Może być zniekształcone przez błędy |
| F1-Score | 0.78 | |
| Accuracy | 0.76 | |
| Val Loss | N/A | Błędnie obliczone |

## Zidentyfikowane Problemy (Audyt)

### Priorytet 1 (Krytyczne)
1. **Napraw obliczenia loss w `evaluate_clip.py`**
2. **Dodaj shuffle do danych treningowych**
3. **Sprawdź czy `vision_model` zwraca `pooler_output`**
4. **Napraw obliczenia avg_train_loss/val_loss w `train_clip.py`**

### Priorytet 2 (Ważne)
1. **Dodaj augmentację danych** (albumentations)
2. **Implementuj cache dla streaming danych**
3. **Dodaj obsługę błędów w confusion matrix**
4. **Zweryfikuj dostępność attention maps w XAI**

### Priorytet 3 (Ulepszenia)
1. **Dodaj monitorowanie multiple metrics dla early stopping**
2. **Implementuj mixed precision training**
3. **Dodaj gradient accumulation**
4. **Rozszerz metryki o precision-recall curve**

## Rozszerzenia i Future Work

### Short-term improvements
1. **Unfreeze last layers**: Fine-tuning ostatnich warstw CLIP
2. **Data augmentation**: Dodanie augmentacji do pipeline'u
3. **Correct metrics**: Naprawa błędów w obliczeniach metryk
4. **Better XAI**: Zastąpienie attention maps Grad-CAM z Captum

### Long-term vision
1. **Multi-modal detection**: Wykorzystanie komponentu tekstowego CLIP
2. **Ensemble approaches**: Połączenie z modelami noise i RGB
3. **Cross-modal retrieval**: Wyszukiwanie podobnych obrazów/textów
4. **Real-time API**: Deployment jako microservice

## XAI Implementation

### Obecna implementacja (problematyczna)
```python
# W xai_clip.py
outputs = model.clip(pixel_values=pixel_values, output_attentions=True)
attentions = outputs.attentions  # Może być None
```

**Problemy**:
1. Założenie, że `model.clip(...)` z `output_attentions=True` zwraca attention
2. Vision model CLIP może nie zwracać attention maps
3. Interpolacja z 7x7 na oryginalny rozmiar powoduje artefakty

### Zalecane poprawki
1. **Sprawdzić `outputs.attentions`** przed użyciem
2. **Użyć Grad-CAM z Captum** zamiast natywnych attention maps
3. **Dodać obsługę błędów** gdy attention maps są niedostępne

## Wnioski

Model CLIP ma potencjał do skutecznej detekcji obrazów AI-generated dzięki bogatej wiedzy semantycznej zdobytej podczas pre-treningu na dużym zbiorze danych. Jednak obecna implementacja wymaga krytycznych poprawek, szczególnie w obszarach:

1. **Poprawności obliczeń** (błędy matematyczne w loss)
2. **Jakości danych** (brak shuffle, augmentacji)
3. **Walidacji założeń** (API CLIP, attention maps)

Po wprowadzeniu niezbędnych poprawek, model może stać się wartościowym komponentem w ensemble approach razem z modelami noise i RGB.

---

**Ostatnia aktualizacja**: 2026-03-29  
**Status**: Wymaga krytycznych poprawek (błędy w obliczeniach loss)  
**Wymagane zależności**: transformers>=4.38.2, datasets, torch, PIL