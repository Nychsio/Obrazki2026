# Deployment - FastAPI Backend i Inference Engine

## Przegląd

Deployment projektu AuthentiScan składa się z backendu FastAPI, który służy jako REST API dla frontendu oraz Inference Engine, który ładuje i uruchamia modele ML. System umożliwia analizę obrazów w czasie rzeczywistym, integrację z modelami językowymi (LLM) oraz generowanie wizualizacji XAI.

## Architektura Systemu

### Komponenty
1. **FastAPI Server** (`deployment/app/main.py`) - REST API endpointy
2. **Inference Engine** (`deployment/app/engine.py`) - Ładowanie i uruchamianie modeli ML
3. **Model Registry** (`deployment/app/models/`) - Wagi wytrenowanych modeli
4. **Environment Configuration** - Zmienne środowiskowe i konfiguracja

### Stack Technologiczny
- **Backend Framework**: FastAPI (Python 3.10+)
- **ML Framework**: PyTorch 2.2+, Transformers, OpenCV
- **XAI Libraries**: Captum, Matplotlib
- **LLM Integration**: OpenAI SDK (DeepSeek API)
- **Image Processing**: PIL/Pillow, OpenCV
- **API Documentation**: Auto-generated Swagger/OpenAPI

## Struktura Projektu

```
deployment/
├── app/
│   ├── main.py              # FastAPI aplikacja i endpointy
│   ├── engine.py            # Inference Engine - główna logika ML
│   ├── dashboard.py         # Dashboard monitoring (opcjonalny)
│   └── models/              # Wagi wytrenowanych modeli
│       ├── best_rgb_model.pt
│       ├── best_noise_model.pt
│       ├── fft_detector_best.pth
│       └── best_pca_model.pt
├── requirements.txt         # Zależności Python
└── .env.example            # Template zmiennych środowiskowych
```

## Inference Engine

### Klasa `InferenceEngine`

Główna klasa odpowiedzialna za:
- Ładowanie modeli ML z dysku
- Przetwarzanie obrazów wejściowych
- Uruchamianie inferencji na GPU/CPU
- Generowanie wizualizacji XAI
- Integracja z różnymi architekturami modeli

### Ładowanie Modeli

```python
class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        
        # Transformacje dla różnych modeli
        self.base_transform = transforms.Compose([...])
        self.fft_transform = transforms.Compose([...])
        self.rgb_transform = transforms.Compose([...])
        self.noise_transform = transforms.Compose([...])
        
        # Ładowanie modeli
        self._load_clip()
        self._load_pca()
        self._load_rgb()
        self._load_noise()
```

### Obsługiwane Modele

#### 1. Model CLIP (Zero-Shot)
- **Architektura**: CLIP ViT-Base-Patch32
- **Zadanie**: Analiza semantyczna i logiczna
- **Prompt Engineering**: 
  ```python
  labels = [
      "a real, natural, authentic photograph without any edits", 
      "an impossible, surreal, AI-generated image with logical mistakes"
  ]
  ```
- **XAI**: Attention maps z ostatniej warstny transformer

#### 2. Model RGB (EfficientNet-B0)
- **Architektura**: EfficientNet-B0 z timm
- **Zadanie**: Analiza artefaktów na poziomie pikseli
- **XAI**: Grad-CAM z użyciem Captum
- **Normalizacja**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### 3. Model Noise (CNN na szumach)
- **Architektura**: Custom CNN dla szumów resztkowych
- **Zadanie**: Detekcja anomalii w szumie PRNU
- **XAI**: Saliency maps z gradientami
- **Filtr**: High-pass Gaussian filter

#### 4. Model FFT (ResNet18)
- **Architektura**: ResNet18 zmodyfikowany dla 2 kanałów (amplituda + faza)
- **Transformacja**: ComplexFourierTransform
- **Zadanie**: Analiza artefaktów w dziedzinie częstotliwości
- **XAI**: Wizualizacja widma FFT

#### 5. Model Gradient PCA
- **Architektura**: GradientCovarianceExtractor + klasyfikator
- **Zadanie**: Analiza wariancji gradientów
- **XAI**: Elipsy wariancji kowariancji
- **Cechy**: Macierze kowariancji 2x2 (4-wymiarowe)

## FastAPI Endpointy

### Główny Endpoint: `/api/v1/analyze`

```python
@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analizuje przesłany obraz przy użyciu wszystkich modeli ML.
    
    Parameters:
    - file: UploadFile - Obraz do analizy (JPEG, PNG, WebP)
    
    Returns:
    - JSON z wynikami wszystkich modeli i werdyktem LLM
    """
```

#### Flow Request-Response:
1. **Przesłanie obrazu** → multipart/form-data
2. **Przetworzenie obrazu** → PIL.Image, konwersja RGB
3. **Inferencja modeli** → równoległe uruchomienie 5 modeli
4. **Generowanie XAI** → base64 encoded visualizations
5. **LLM Integration** → werdykt DeepSeek na podstawie wyników
6. **Response** → JSON z kompletnymi wynikami

### Struktura Odpowiedzi

```json
{
  "status": "success",
  "predictions": {
    "rgb_prob": 0.85,
    "clip_prob": 0.72,
    "fft_prob": 0.91,
    "noise_prob": 0.68,
    "pca_prob": 0.79,
    "rgb_gradcam": "base64_string",
    "clip_vis": "base64_string",
    "fft_vis": "base64_string",
    "noise_vis": "base64_string",
    "pca_vis": "base64_string",
    "llm_verdict": "Analiza wskazuje na wysokie prawdopodobieństwo..."
  }
}
```

## XAI (Explainable AI) Visualizations

### 1. Grad-CAM dla Modelu RGB
- **Technika**: LayerGradCam z Captum
- **Warstwa docelowa**: `model.backbone.conv_head`
- **Visualization**: Heatmap nałożony na oryginalny obraz
- **Paleta kolorów**: JET colormap

### 2. Attention Maps dla CLIP
- **Technika**: Self-attention z ostatniej warstny ViT
- **Extraction**: `outputs.vision_model_output.attentions[-1]`
- **Aggregation**: Średnia ze wszystkich attention heads
- **Visualization**: INFERNO colormap

### 3. FFT Spectrum Visualization
- **Technika**: 2D FFT + magnitude spectrum
- **Processing**: Log scaling, normalization
- **Visualization**: VIRIDIS colormap
- **Interpretacja**: Jasne piki = anomalie częstotliwościowe

### 4. Noise Saliency Maps
- **Technika**: Gradient-based saliency
- **Processing**: Absolute gradients, max pooling
- **Visualization**: OCEAN colormap
- **Interpretacja**: Obszary z nienaturalnym szumem

### 5. PCA Covariance Ellipses
- **Technika**: Analiza macierzy kowariancji gradientów
- **Visualization**: Elipsy wariancji na wykresie 2D
- **Styl**: Dark theme z niebieskimi elipsami
- **Interpretacja**: Okrągła elipsa = brak artefaktów

## LLM Integration (DeepSeek)

### Konfiguracja
```python
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)
```

### Prompt Engineering
```python
prompt = f"""
Jesteś ekspertem ds. cyberbezpieczeństwa i detekcji deepfake'ów. 
Przeanalizuj poniższe wyniki z 4 modeli sztucznej inteligencji...

Wyniki modeli (prawdopodobieństwo, że to FAKE):
- Model CLIP: {predictions.get('clip_prob', 0) * 100:.1f}%
- Model FFT: {predictions.get('fft_prob', 0) * 100:.1f}%
- Model RGB: {predictions.get('rgb_prob', 0) * 100:.1f}%
- Model Noise: {predictions.get('noise_prob', 0) * 100:.1f}%

Napisz werdykt w języku polskim, w profesjonalnym ale przystępnym tonie.
"""
```

### Parametry LLM
- **Model**: `deepseek-chat`
- **Max Tokens**: 200
- **Temperature**: 0.3 (niska dla spójności)
- **System Prompt**: "Jesteś pomocnym asystentem analitykiem."

## Konfiguracja Środowiska

### Wymagania Systemowe
- **Python**: 3.10+
- **PyTorch**: 2.2+ (z CUDA 12.1 dla GPU)
- **RAM**: Minimum 8GB (16GB recommended)
- **VRAM**: 4GB+ dla inferencji na GPU
- **Disk Space**: 2GB+ dla modeli

### Instalacja Zależności

```bash
cd deployment
pip install -r requirements.txt
```

### requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.2.0
torchvision==0.17.0
transformers==4.36.2
openai==1.12.0
pillow==10.1.0
opencv-python==4.9.0.80
matplotlib==3.8.2
captum==0.7.0
python-dotenv==1.0.0
numpy==1.24.3
scipy==1.11.4
timm==0.9.12
```

### Zmienne Środowiskowe (.env)
```env
# DeepSeek API Key (wymagane)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Opcjonalne - konfiguracja GPU
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Logging
LOG_LEVEL=INFO
```

## Uruchomienie Serwera

### Development Mode
```bash
cd deployment
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode (z Gunicorn)
```bash
cd deployment
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring i Logging

### Dashboard (`dashboard.py`)
- Monitorowanie obciążenia GPU/CPU
- Statystyki requestów
- Cache hit rates
- Error tracking

### Logging Configuration
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Performance Optimization

### Caching
- Cache model loading (singleton pattern)
- Cache XAI visualizations
- LRU cache dla często używanych obrazów

### Batch Processing
```python
# Planowane: batch inference dla wielu obrazów
async def analyze_batch(files: List[UploadFile]):
    # Parallel processing with asyncio
    pass
```

### GPU Optimization
- Mixed precision inference (FP16)
- CUDA graph capture dla powtarzalnych operacji
- Memory pooling dla tensorów

## Bezpieczeństwo

### Input Validation
- Sprawdzanie rozmiaru pliku (max 10MB)
- Walidacja typu MIME
- Sanity check rozdzielczości
- Protection against malformed images

### API Security
- Rate limiting (planowane)
- API key authentication (dla produkcji)
- CORS configuration
- Request timeout (30s)

### Model Security
- Sandboxed inference
- Checksum verification dla wag modeli
- Protection against model inversion attacks

## Known Issues i Rozwój

### ✅ Działające
- Podstawowy inference pipeline
- XAI dla wszystkich modeli
- LLM integration
- REST API z FastAPI

### ⚠️ Wymagające Poprawek
1. **Model FFT**: Brak załadowanych wag (komentarz w kodzie)
2. **Error Handling**: Brak comprehensive error recovery
3. **Batch Processing**: Brak obsługi wielu obrazów
4. **Model Versioning**: Brak systemu wersjonowania modeli

### 🚀 Roadmap
1. **Model Registry**: Centralne zarządzanie wersjami modeli
2. **A/B Testing**: Testowanie nowych modeli na produkcji
3. **Auto-scaling**: Dynamiczne skalowanie w chmurze
4. **Model Compression**: Quantization, pruning dla mobile
5. **Edge Deployment**: ONNX Runtime, TensorRT
6. **Monitoring**: Prometheus + Grafana dashboard
7. **CI/CD**: Automated model deployment pipeline

## Troubleshooting

### Common Issues

#### 1. Brak klucza DeepSeek API
```
ValueError: ⚠️ BŁĄD KRYTYCZNY: Brak klucza DEEPSEEK_API_KEY w pliku .env!
```
**Rozwiązanie**: Utwórz plik `.env` z kluczem API

#### 2. Brak pamięci GPU
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Rozwiązanie**:
- Zmniejsz batch size
- Użyj `torch.cuda.empty_cache()`
- Włącz mixed precision (FP16)

#### 3. Wolne ładowanie modeli
**Rozwiązanie**:
- Cache modeli w pamięci
- Użyj `weights_only=True` dla bezpieczeństwa
- Parallel loading przy starcie

#### 4. Błędy XAI generation
**Rozwiązanie**:
- Sprawdź wymiary tensorów
- Upewnij się, że gradienty są włączone
- Debug z podstawowymi obrazami testowymi

## Deployment Strategies

### 1. Local Development
```bash
# Frontend + Backend
cd frontend/frontend && npm run dev  # localhost:5173
cd deployment && uvicorn app.main:app --reload  # localhost:8000
```

### 2. Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: ./deployment
    ports:
      - "8000:8000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./deployment/app/models:/app/app/models
  
  frontend:
    build: ./frontend/frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend
```

### 3. Kubernetes
- Deployment z HPA (Horizontal Pod Autoscaler)
- ConfigMaps dla zmiennych środowiskowych
- PersistentVolume dla wag modeli
- Ingress dla routingu

### 4. Serverless (AWS Lambda)
- Container image deployment
- API Gateway integration
- S3 dla wag modeli
- CloudWatch dla logów

---

*Dokumentacja zaktualizowana: 2026-04-16*  
*Autor: Zespół AuthentiScan*  
*Cel: Kompleksowy przewodnik po deployment i inference pipeline*