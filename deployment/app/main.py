from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from deployment.app.engine import InferenceEngine

app = FastAPI(
    title="Detektor Obrazów AI - Meta-Klasyfikator",
    description="API do fuzji wyników z modeli: Noise, RGB, FFT, CLIP oraz Gradient PCA",
    version="1.0.0"
)

# Inicjalizacja silnika przy starcie serwera
engine = InferenceEngine()

@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Plik musi być obrazem.")
    
    try:
        # Odczyt obrazu z pamięci
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Wywołanie zintegrowanego systemu
        predictions = engine.predict(image)
        
        # Tu w przyszłości dodamy wywołanie modelu Ensemble (Sędziego)
        
        return {
            "status": "success",
            "filename": file.filename,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd analizy: {str(e)}")