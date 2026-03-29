# Przewodnik Szybkiego Startu

## Wymagania Wstępne

### Systemowe
- **Python**: 3.10 lub nowszy
- **PyTorch**: 2.2.0+ (z CUDA jeśli dostępne)
- **RAM**: Minimum 8GB (16GB zalecane)
- **Dysk**: Minimum 10GB wolnego miejsca

### Dla GPU (opcjonalnie)
- **NVIDIA GPU**: z co najmniej 6GB VRAM
- **CUDA**: 12.1 (kompatybilne z PyTorch 2.2+)
- **cuDNN**: 8.9+

## Szybka Instalacja

### Opcja 1: Conda (zalecana)
```bash
# Klonowanie repozytorium
git clone git@github.com:Nychsio/Obrazki2026.git
cd Obrazki2026

# Tworzenie środowiska Conda
conda env create -f environment.yml

# Aktywacja środowiska
conda activate ai-forensics-env
```

### Opcja 2: Pip
```bash
# Klonowanie repozytorium
git clone git@github.com:Nychsio/Obrazki2026.git
cd Obrazki2026

# Instalacja zależności
pip install -r requirements.txt
```

### Opcja 3: Docker
```bash
# Budowanie obrazu
docker build -t ai-forensics -f docker/Dockerfile .

# Uruchomienie kontenera
docker run -it --gpus all -v $(pwd):/workspace ai-forensics
```

## Weryfikacja Instalacji

### Test podstawowych zależności
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA dostępne: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "from transformers import __version__; print(f'transformers: {__version__}')"
```

### Test ładowania danych
```bash
python -c "
from datasets import load_dataset
dataset = load_dataset('ComplexDataLab/OpenFake', split='train', streaming=True)
sample = next(iter(dataset))
print(f'Dataset załadowany: {sample.keys()}')
print(f'Typ obrazu: {type(sample[\"image\"])}')
"
```

## Szybkie Testy Modeli

### 1. Model Noise (najszybszy)
```bash
# Trening (demonstracyjny - mały dataset)
python src/noise/train.py

# Test na przykładowym obrazie
python -c "
import torch
from src.noise.model import NoiseBinaryClassifier
from PIL import Image
import torchvision.transforms as T

# Ładowanie modelu
model = NoiseBinaryClassifier()
model.load_state_dict(torch.load('best_noise_model.pt', map_location='cpu'))
model.eval()

# Przetwarzanie obrazu
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Predykcja
with torch.no_grad():
    logits = model(image_tensor)
    probability = torch.sigmoid(logits).item()
    
print(f'Prawdopodobieństwo fake: {probability:.4f}')
print(f'Klasa: {\"FAKE\" if probability > 0.5 else \"REAL\"}')
"
```

### 2. Model RGB (najbardziej zaawansowany)
```bash
# Inferencja na pojedynczym obrazie
python src/rgb/inference.py test_image.jpg --model_path best_rgb_model.pt

# XAI - Grad-CAM
python src/rgb/explain.py

# Wynik: gradcam_output.jpg
```

### 3. Model FFT (wymaga poprawek)
```bash
# Uwaga: train.py ma błędy syntaktyczne
# Można użyć poc_train.py dla demonstracji
python src/models/fft_detector/poc_train.py

# Predykcja (działa)
python src/models/fft_detector/predict_single.py test_image.jpg
```

### 4. Model CLIP (wymaga poprawek)
```bash
# Uwaga: evaluate_clip.py ma błędy w obliczeniach loss
# Test integracyjny (działa)
python src/models/clip/test_integration.py
```

## Przykładowy Workflow

### Krok 1: Przygotowanie danych
```bash
# Pobierz przykładowe obrazy testowe
curl -o test_image.jpg https://example.com/sample.jpg
curl -o test_image1.jpg https://example.com/sample2.jpg
```

### Krok 2: Uruchomienie wszystkich modeli
```bash
# Utwórz skrypt porównawczy
cat > compare_models.py << 'EOF'
import subprocess
import sys

models = [
    ("Noise", "python src/noise/train.py --quick_test"),
    ("RGB", "python src/rgb/inference.py test_image.jpg --model_path best_rgb_model.pt"),
    ("FFT", "python src/models/fft_detector/predict_single.py test_image.jpg"),
]

print("=== PORÓWNANIE MODELI ===")
for name, cmd in models:
    print(f"\n--- {name} Model ---")
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Błędy: {result.stderr[:200]}...")
    except Exception as e:
        print(f"Błąd: {e}")
EOF

python compare_models.py
```

### Krok 3: Analiza wyników
```bash
# Sprawdź wygenerowane pliki
ls -la *.jpg *.png *.pth *.pt

# Oczekiwane pliki:
# - gradcam_output.jpg (RGB model XAI)
# - fft_spectrum_*.jpg (FFT model visualization)
# - best_*.pt (zapisane modele)
# - test_image.jpg (przykładowy obraz)
```

## Rozwiązywanie Problemów

### Częste problemy i rozwiązania

#### 1. Brak pamięci GPU
```bash
# Zmniejsz batch size
export BATCH_SIZE=16

# Użyj CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 2. Błędy z Hugging Face datasets
```bash
# Użyj trybu offline jeśli masz problemy z siecią
export HF_DATASETS_OFFLINE=1

# Lub użyj cache
export HF_HOME=/path/to/cache
```

#### 3. Problemy z Windows
```bash
# Ustaw num_workers=0 we wszystkich DataLoaderach
# Edytuj pliki train.py i zmień:
# num_workers=0  # zamiast num_workers=4
```

#### 4. Błędy z zależnościami
```bash
# Zaktualizuj pip
python -m pip install --upgrade pip

# Zainstaluj ponownie zależności
pip install -r requirements.txt --force-reinstall
```

## Monitorowanie i Debugowanie

### TensorBoard (dla modeli RGB i Noise)
```bash
# Uruchom TensorBoard
tensorboard --logdir runs/

# Otwórz w przeglądarce: http://localhost:6006
```

### Proste logowanie
```bash
# Sprawdź logi treningu
tail -f training.log  # Linux/Mac
Get-Content -Wait training.log  # Windows PowerShell
```

### Test wydajności
```bash
# Benchmark inference time
python -c "
import time
import torch
from src.rgb.feature_extractor import create_feature_extractor

model = create_feature_extractor()
dummy_input = torch.randn(1, 3, 224, 224)

# Warmup
for _ in range(10):
    _ = model(dummy_input)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    _ = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times.append(time.time() - start)

print(f'Średni czas inference: {sum(times)/len(times)*1000:.2f}ms')
print(f'FPS: {1/(sum(times)/len(times)):.2f}')
"
```

## Następne Kroki

### Dla developerów
1. **Przeczytaj dokumentację modeli** w `.memory/models/`
2. **Sprawdź znane problemy** w `.memory/issues/known_issues.md`
3. **Eksperymentuj z hiperparametrami** w `configs/`
4. **Dodaj testy** w `tests/`

### Dla researcherów
1. **Eksperymentuj z ensemble** modeli
2. **Testuj na różnych datasetach**
3. **Implementuj nowe metody XAI**
4. **Publikuj wyniki** w `notebooks/`

### Dla production
1. **Napraw krytyczne błędy** (priorytet 1)
2. **Zoptymalizuj wydajność**
3. **Stwórz REST API**
4. **Przygotuj Docker deployment**

## Pomoc i Wsparcie

### Debugowanie
```bash
# Pełny test środowiska
python scripts/test_environment.py

# Test poszczególnych komponentów
python -m pytest tests/ -v

# Sprawdź zużycie GPU
nvidia-smi  # Linux
# lub
torch.cuda.memory_summary()  # Python
```

### Logi i raporty
- **TensorBoard**: `runs/` - metryki treningu
- **Checkpointy**: `best_*.pt`, `fft_detector_best.pth` - wagi modeli
- **Wizualizacje**: `*_output.jpg` - wyniki XAI
- **Logi**: Sprawdź output w terminalu

### Kontakt
- **Issues**: GitHub Issues w repozytorium
- **Documentation**: `.memory/` - memory bank projektu
- **Examples**: `notebooks/` - przykłady użycia

---

**Uwaga**: Ten projekt jest w fazie aktywnego rozwoju. Niektóre komponenty (FFT, CLIP) wymagają poprawek. Zacznij od modeli Noise i RGB które są najbardziej stabilne.

**Gotowość produkcyjna**:
- ✅ Noise Model: Gotowy
- ✅ RGB Model: Gotowy (z XAI)
- ⚠️ FFT Model: Wymaga poprawek (błędy syntaktyczne)
- ⚠️ CLIP Model: Wymaga poprawek (błędy w obliczeniach)

---

**Ostatnia aktualizacja**: 2026-03-29  
**Wersja**: 1.0.0-beta  
**Stan**: Aktywny rozwój z znanymi problemami