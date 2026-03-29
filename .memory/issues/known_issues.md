# Zidentyfikowane Problemy w Projekcie

## Przegląd

Ten dokument zawiera listę wszystkich zidentyfikowanych problemów w projekcie, pogrupowanych według priorytetu i komponentu. Problemy są wynikiem audytu kodu i analizy implementacji.

## Priorytety

### 🚨 PRIORYTET 1 (KRYTYCZNE)
Problemy uniemożliwiające poprawne działanie systemu lub prowadzące do błędnych wyników.

### ⚠️ PRIORYTET 2 (WAŻNE)
Problemy wpływające na jakość, wydajność lub utrzymywalność kodu.

### 💡 PRIORYTET 3 (ULEPSZENIA)
Sugestie ulepszeń i optymalizacji, które nie są krytyczne.

---

## Model CLIP

### 🚨 PRIORYTET 1
1. **`evaluate_clip.py` - BŁĄD MATEMATYCZNY W OBLICZENIACH LOSS**
   - **Lokalizacja**: `src/models/clip/evaluate_clip.py`
   - **Problem**: `avg_test_loss = test_loss / len(all_labels) * batch_size`
   - **Poprawka**: `avg_test_loss = test_loss / (len(all_labels) / batch_size)` lub `test_loss / actual_steps`
   - **Konsekwencja**: Zniekształcone wartości loss, nieporównywalne między różnymi batch sizes

2. **BRAK SHUFFLE DANYCH TRENINGOWYCH**
   - **Lokalizacja**: `src/models/clip/clip_streamer.py`
   - **Problem**: Streaming dataset bez shuffle prowadzi do overfittingu na kolejności danych
   - **Poprawka**: Implementacja buffer shuffle dla streaming danych
   - **Konsekwencja**: Model uczy się na sekwencyjnych danych, słaba generalizacja

3. **ZAŁOŻENIE O `pooler_output` BEZ WERYFIKACJI**
   - **Lokalizacja**: `src/models/clip/semantic_judge.py`
   - **Problem**: Użycie `outputs.pooler_output` bez sprawdzenia czy `vision_model` CLIP zwraca tę wartość
   - **Poprawka**: Sprawdzenie dokumentacji CLIP lub użycie `last_hidden_state[:, 0]` (CLS token)
   - **Konsekwencja**: `AttributeError` jeśli `pooler_output` nie istnieje

### ⚠️ PRIORYTET 2
4. **FIXED `max_steps_per_epoch=1000` BEZ SPRAWDZANIA KONCA STRUMIENIA**
   - **Lokalizacja**: `src/models/clip/train_clip.py`
   - **Problem**: Jeśli strumień ma <1000 batchy, pętla się zakończy wcześniej, ale loss jest dzielone przez 1000
   - **Poprawka**: Liczenie rzeczywistej liczby kroków lub użycie `while True` z break na `StopIteration`

5. **BRAK OBSŁUGI BŁĘDÓW W CONFUSION MATRIX**
   - **Lokalizacja**: `src/models/clip/evaluate_clip.py`
   - **Problem**: `confusion_matrix().ravel()` bez sprawdzenia kształtu (błąd gdy wszystkie przewidywania tej samej klasy)
   - **Poprawka**: Dodać walidację lub użyć `confusion_matrix(..., labels=[0,1])`

6. **ZAŁOŻENIE O ATTENTION MAPS W XAI**
   - **Lokalizacja**: `src/models/clip/xai_clip.py`
   - **Problem**: Założenie, że `model.clip(...)` z `output_attentions=True` zwraca attention
   - **Poprawka**: Sprawdzić czy `outputs.attentions` nie jest `None`

### 💡 PRIORYTET 3
7. **BRAK AUGMENTACJI DANYCH**
   - **Lokalizacja**: `src/models/clip/clip_streamer.py`
   - **Poprawka**: Dodanie augmentacji za pomocą albumentations

8. **BRAK CACHE'OWANIA DLA STREAMING DANYCH**
   - **Poprawka**: Implementacja lokalnego cache lub użycie `datasets.cache`

---

## Model FFT

### 🚨 PRIORYTET 1
9. **`train.py` - BŁĄD SYNTAKTYKCZNY (NIE PRAWIDŁOWE WCIĘCIA)**
   - **Lokalizacja**: `src/models/fft_detector/train.py` (linie 70-80)
   - **Problem**: Kod poza pętlami treningowymi (epoch_loss i zapis modelu)
   - **Poprawka**: Poprawienie wcięć, przeniesienie kodu we właściwe miejsce
   - **Konsekwencja**: Błąd wykonania lub nieprawidłowe działanie treningu

10. **NIESPÓJNOŚĆ Z `data_loader.py`**
    - **Lokalizacja**: `src/models/fft_detector/poc_train.py` (linie 24-30)
    - **Problem**: Redundantna logika konwersji etykiet (`data_loader.py` już konwertuje)
    - **Poprawka**: Usunięcie redundantnego kodu, użycie spójnego formatu

### ⚠️ PRIORYTET 2
11. **BRAK OBSŁUGI BŁĘDÓW DLA MODELU W `predict_single.py`**
    - **Lokalizacja**: `src/models/fft_detector/predict_single.py`
    - **Problem**: Brak walidacji struktury checkpointu, może powodować `KeyError`
    - **Poprawka**: Dodanie obsługi różnych formatów checkpointów

12. **BRAK PRAWDZIWEGO VALIDATION SPLIT**
    - **Problem**: Trening odbywa się tylko na danych treningowych, brak ewaluacji
    - **Poprawka**: Implementacja podziału 80/20 lub 70/30 train/val

### 💡 PRIORYTET 3
13. **HARDCODED HIPERPARAMETRY**
    - **Poprawka**: Przeniesienie do pliku konfiguracyjnego YAML/JSON

14. **BRAK AUGMENTACJI DANYCH**
    - **Poprawka**: Dodanie augmentacji do pipeline treningu

---

## Wspólne Problemy

### ⚠️ PRIORYTET 2
15. **BRAK MIXED PRECISION TRAINING DLA WSZYSTKICH MODELI**
    - **Problem**: Tylko model RGB używa mixed precision
    - **Poprawka**: Zaimplementować dla noise, FFT i CLIP modeli

16. **BRAK GRADIENT ACCUMULATION**
    - **Problem**: Ograniczenie przy małych batch sizes
    - **Poprawka**: Dodanie gradient accumulation dla wszystkich modeli

17. **BRAK OBSŁUGI WINDOWS (num_workers=0)**
    - **Problem**: Wymuszone `num_workers=0` dla Windows
    - **Poprawka**: Implementacja fallback mechanism

### 💡 PRIORYTET 3
18. **BRAK ENSEMBLE MODELI**
    - **Poprawka**: Połączenie predykcji noise, RGB, FFT i CLIP

19. **BRAK REAL-TIME API**
    - **Poprawka**: Implementacja REST API dla batch processing

20. **BRAK DOCKER COMPOSE DLA PEŁNEGO SYSTEMU**
    - **Poprawka**: Docker Compose z wszystkimi modelami i API

---

## Problemy z Danymi

### ⚠️ PRIORYTET 2
21. **STREAMING BEZ SHUFFLE (WSZYSTKIE MODELE)**
    - **Problem**: Wszystkie modele używają streaming bez shuffle
    - **Poprawka**: Globalna implementacja buffer shuffle

22. **NIESPÓJNOŚĆ ETYKIET**
    - **Problem**: Różne formaty etykiet w różnych batchach
    - **Poprawka**: Ujednolicone mapowanie we wszystkich data loaderach

### 💡 PRIORYTET 3
23. **BRAK LOCAL CACHE**
    - **Poprawka**: Cache często używanych danych na dysku

24. **BRAK DATA VERSIONING**
    - **Poprawka**: System wersjonowania datasetów

---

## Problemy z Dokumentacją

### ⚠️ PRIORYTET 2
25. **NIEKOMPLETNA DOKUMENTACJA**
    - **Problem**: Brak dokumentacji dla niektórych komponentów
    - **Poprawka**: Uzupełnienie brakującej dokumentacji

26. **BRAK INSTALACJI KROK PO KROKU**
    - **Poprawka**: Dodanie szczegółowego guide instalacyjnego

### 💡 PRIORYTET 3
27. **BRAK EXAMPLE NOTEBOOKÓW**
    - **Poprawka**: Dodanie Jupyter notebooks z przykładami użycia

28. **BRAK API DOCUMENTATION**
    - **Poprawka**: Automatyczna generacja dokumentacji API

---

## Problemy z Testami

### ⚠️ PRIORYTET 2
29. **BRAK TESTÓW JEDNOSTKOWYCH**
    - **Problem**: Bardzo ograniczona liczba testów
    - **Poprawka**: Dodanie testów jednostkowych dla kluczowych komponentów

30. **BRAK TESTÓW INTEGRACYJNYCH**
    - **Poprawka**: Testy całego pipeline'u od danych do predykcji

### 💡 PRIORYTET 3
31. **BRAK TESTÓW WYDAJNOŚCIOWYCH**
    - **Poprawka**: Benchmarki wydajnościowe dla różnych hardware

32. **BRAK TESTÓW Z RÓŻNYMI DATASETAMI**
    - **Poprawka**: Testy cross-dataset evaluation

---

## Plan Naprawy

### Faza 1: Krytyczne poprawki (1-2 tygodnie)
1. Naprawa błędów matematycznych w CLIP (`evaluate_clip.py`)
2. Poprawienie błędów syntaktycznych w FFT (`train.py`)
3. Implementacja shuffle dla streaming danych
4. Weryfikacja API CLIP (`pooler_output`)

### Faza 2: Ważne poprawki (2-3 tygodnie)
1. Dodanie validation split dla wszystkich modeli
2. Implementacja mixed precision dla wszystkich modeli
3. Ujednolicenie mapowania etykiet
4. Dodanie testów jednostkowych

### Faza 3: Ulepszenia (3-4 tygodnie)
1. Implementacja ensemble modeli
2. Dodanie REST API
3. Rozszerzenie dokumentacji
4. Optymalizacja wydajności

---

## Status Śledzenia

| ID | Problem | Priorytet | Status | Przypisany do | Data zgłoszenia |
|----|---------|-----------|--------|---------------|-----------------|
| 1 | Błąd obliczeń loss w CLIP | 🚨 P1 | **OTWARTY** | - | 2026-03-29 |
| 2 | Brak shuffle danych | 🚨 P1 | **OTWARTY** | - | 2026-03-29 |
| 3 | Założenie o pooler_output | 🚨 P1 | **OTWARTY** | - | 2026-03-29 |
| 9 | Błąd syntaktyczny w FFT train.py | 🚨 P1 | **OTWARTY** | - | 2026-03-29 |
| 4 | Fixed steps w CLIP | ⚠️ P2 | **OTWARTY** | - | 2026-03-29 |
| 11 | Brak obsługi błędów w FFT predict | ⚠️ P2 | **OTWARTY** | - | 2026-03-29 |
| 15 | Brak mixed precision | ⚠️ P2 | **OTWARTY** | - | 2026-03-29 |
| 7 | Brak augmentacji CLIP | 💡 P3 | **OTWARTY** | - | 2026-03-29 |
| 18 | Brak ensemble | 💡 P3 | **OTWARTY** | - | 2026-03-29 |

---

## Uwagi

1. **Priorytety są dynamiczne** - mogą się zmieniać w zależności od potrzeb projektu
2. **Niektóre problemy są powiązane** - naprawa jednego może rozwiązać kilka innych
3. **Zalecane podejście iteracyjne** - małe, przyrostowe poprawki zamiast dużych refactorów
4. **Testowanie po każdej zmianie** - aby uniknąć regresji

---

**Ostatnia aktualizacja**: 2026-03-29  
**Autor audytu**: Cline (AI Assistant)  
**Następny przegląd**: 2026-04-05