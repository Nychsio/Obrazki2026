# Dokumentacja: AI-Generated Image and Digital Manipulation Detector (CLIP-based)

## Przegląd Architektury

Projekt implementuje detektor obrazów wygenerowanych przez AI i cyfrowych manipulacji wykorzystujący architekturę opartą na CLIP (Contrastive Language-Image Pre-training). System składa się z następujących komponentów:

### 1. Główne Moduły

#### `semantic_judge.py` - Model Główny
- **Architektura**: Fine-tuning vision model z CLIP (ViT-Base-Patch32) z dodaną głową klasyfikacyjną
- **Wejście**: Obrazy 224x224 RGB (przetworzone przez CLIPProcessor)
- **Wyjście**: Logity dla klasyfikacji binarnej (Real vs Fake)
- **Transfer Learning**: Domyślnie zamrożony backbone CLIP, trenowana tylko głowa klasyfikacyjna
- **Warstwy klasyfikatora**: Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→1)

#### `clip_streamer.py` - Strumień Danych
- **Źródło danych**: Dataset OpenFake (ComplexDataLab/OpenFake) w trybie streaming
- **Przetwarzanie**: CLIPProcessor (resize 224x224, normalizacja)
- **Batchowanie**: Dynamiczne ładowanie danych bez cache'owania na dysku
- **Problem**: Brak shuffle danych treningowych

#### `train_clip.py` - Trening Modelu
- **Optymalizator**: AdamW (tylko dla klasyfikatora, lr=1e-4)
- **Funkcja straty**: BCEWithLogitsLoss (stabilna numerycznie)
- **Scheduler**: CosineAnnealingLR
- **Early Stopping**: Based on validation loss (patience=3)
- **Metryki**: ROC-AUC, F1-Score obliczane na zbiorze walidacyjnym

#### `evaluate_clip.py` - Ewaluacja Modelu
- **Metryki**: ROC-AUC, F1-Score, Accuracy, Confusion Matrix
- **Testowanie**: Na zbiorze test/validation
- **Problemy**: Błędy w obliczaniu loss

#### `xai_clip.py` - Explainable AI
- **Metoda**: Attention maps z ostatniej warstwy Transformer
- **Wizualizacja**: Heatmap nakładana na oryginalny obraz
- **Problemy**: Założenia o dostępności attention maps

#### `test_integration.py` - Testy Integracyjne
- **Cel**: Sanity check pipeline'u
- **Metoda**: Overfitting na pojedynczym batchu

## Zidentyfikowane Problemy i "AI Hallucinations"

### Krytyczne Błędy Logiczne

#### 1. **semantic_judge.py**
   - **Problem**: Użycie `pooler_output` z vision_model CLIP - wymaga weryfikacji czy vision_model faktycznie zwraca pooler_output
   - **Ryzyko**: Model może używać nieoptymalnej reprezentacji obrazu
   - **Rozwiązanie**: Sprawdzić dokumentację CLIP vision_model lub użyć `last_hidden_state[:, 0]` (CLS token)

#### 2. **train_clip.py**
   - **Problem**: Fixed `max_steps_per_epoch=1000` bez sprawdzania końca strumienia
   - **Konsekwencja**: Jeśli strumień ma <1000 batchy, pętla się zakończy wcześniej, ale loss jest dzielone przez 1000
   - **Rozwiązanie**: Liczyć rzeczywistą liczbę kroków lub użyć `while True` z break na StopIteration
   
   - **Problem**: Brak shuffle danych treningowych
   - **Konsekwencja**: Model uczy się na sekwencyjnych danych, co prowadzi do overfittingu na kolejności
   - **Rozwiązanie**: Implementacja buffer shuffle dla streaming danych

#### 3. **evaluate_clip.py**
   - **Krytyczny błąd**: `avg_test_loss = test_loss / len(all_labels) * batch_size`
   - **Matematycznie niepoprawne**: Powinno być `test_loss / (len(all_labels) / batch_size)` lub `test_loss / actual_steps`
   - **Konsekwencja**: Zniekształcone wartości loss, nieporównywalne między różnymi batch sizes
   
   - **Problem**: `confusion_matrix().ravel()` bez sprawdzenia kształtu
   - **Ryzyko**: Błąd jeśli macierz nie jest 2x2 (np. gdy wszystkie przewidywania tej samej klasy)
   - **Rozwiązanie**: Dodać walidację lub użyć `confusion_matrix(..., labels=[0,1])`

#### 4. **clip_streamer.py**
   - **Problem**: Brak augmentacji danych
   - **Konsekwencja**: Ograniczona generalizacja modelu
   - **Rozwiązanie**: Dodanie augmentacji za pomocą albumentations
   
   - **Problem**: Brak obsługi cache'owania dla streaming danych
   - **Konsekwencja**: Powolne ładowanie przy każdym epokowaniu
   - **Rozwiązanie**: Implementacja lokalnego cache lub użycie `datasets.cache`

#### 5. **xai_clip.py**
   - **Problem**: Założenie, że `model.clip(...)` z `output_attentions=True` zwraca attention
   - **Ryzyko**: Vision model CLIP może nie zwracać attention maps
   - **Rozwiązanie**: Sprawdzić czy `outputs.attentions` nie jest `None`
   
   - **Problem**: Interpolacja heatmap z 7x7 na oryginalny rozmiar
   - **Konsekwencja**: Artefakty i niedokładna lokalizacja
   - **Rozwiązanie**: Użycie bardziej zaawansowanych metod jak Grad-CAM z Captum

### Problemy Wydajnościowe

1. **Zamrożenie całego backbone**: Dobre dla szybkiego treningu, ale może ograniczać osiągi
2. **Streaming bez cache**: Każda epoka ładuje dane od nowa
3. **Brak mixed precision**: Możliwość przyspieszenia treningu na GPU
4. **Brak gradient accumulation**: Ograniczenie przy małych batch sizes

### Ryzyko Overfitting

1. **Early stopping tylko na val_loss**: Brak monitorowania AUC/F1
2. **Brak regularizacji poza dropout**: Brak weight decay w zamrożonych warstwach
3. **Bias w danych**: OpenFake dataset może mieć systematyczne różnice między real/fake
4. **Brak cross-validation**: Ocena na pojedynczym zbiorze walidacyjnym

### Problemy z Metrykami

1. **ROC-AUC dla danych niezbalansowanych**: AUC może być mylące przy dużym imbalance
2. **F1-Score z progiem 0.5**: Stały próg może nie być optymalny
3. **Brak precision-recall curve**: Ważniejsze dla detection tasks
4. **Brak confidence intervals**: Punktowe estymaty bez niepewności

## Zalecenia Poprawkowe

### Priorytet 1 (Krytyczne)
1. **Popraw obliczenia loss w evaluate_clip.py**
2. **Dodaj shuffle do danych treningowych**
3. **Sprawdź czy vision_model zwraca pooler_output**
4. **Napraw obliczenia avg_train_loss/val_loss w train_clip.py**

### Priorytet 2 (Ważne)
1. **Dodaj augmentację danych**
2. **Implementuj cache dla streaming danych**
3. **Dodaj obsługę błędów w confusion matrix**
4. **Zweryfikuj dostępność attention maps w XAI**

### Priorytet 3 (Ulepszenia)
1. **Dodaj monitorowanie multiple metrics dla early stopping**
2. **Implementuj mixed precision training**
3. **Dodaj gradient accumulation**
4. **Rozszerz metryki o precision-recall curve**

## Zależności i Wersje

### Wymagane pakiety (requirements.txt)
- `torch`, `torchvision`: Framework deep learning
- `transformers==4.38.2`: Model CLIP i processor (wymagana wersja 4.38.2)
- `datasets==2.18.0`: Ładowanie danych OpenFake
- `albumentations`: Augmentacja obrazów
- `captum`: Explainable AI (Grad-CAM)
- `scikit-learn==1.4.1post1`: Metryki ewaluacyjne

### Konflikty wersji
- Aktualna transformers 4.57.1 vs wymagana 4.38.2
- Potencjalne problemy z API changes

## Struktura Projektu

```
g:/obrazki/
├── src/models/clip/
│   ├── semantic_judge.py    # Model główny
│   ├── train_clip.py        # Skrypt treningowy
│   ├── evaluate_clip.py     # Ewaluacja
│   ├── clip_streamer.py     # Strumień danych
│   ├── xai_clip.py          # Explainable AI
│   └── test_integration.py  # Testy
├── configs/                 # Konfiguracje
├── data/                   # Dane lokalne
├── checkpoints/            # Zapisane wagi
└── notebooks/              # Eksperymenty
```

## Uwagi Końcowe

Kod wykazuje typowe "AI hallucinations":
1. Założenia o API bez weryfikacji (pooler_output, attention maps)
2. Błędy matematyczne w obliczeniach metryk
3. Brak obsługi edge cases (pusty strumień, macierz 1x1)
4. Niedostateczna walidacja danych wejściowych

Mimo to, architektura jest zasadna i projekt ma solidne podstawy. Większość problemów można naprawić stosunkowo prostymi poprawkami. Kluczowe jest przetestowanie założeń dotyczących CLIP vision_model API przed dalszym rozwojem.