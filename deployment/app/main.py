from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deployment.app.engine import InferenceEngine
from PIL import Image
import io
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Wczytanie zmiennych z pliku .env
load_dotenv()

app = FastAPI(title="Detektor Obrazów AI - API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine()

# Bezpieczne pobranie klucza ze zmiennych środowiskowych
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("⚠️ BŁĄD KRYTYCZNY: Brak klucza DEEPSEEK_API_KEY w pliku .env!")

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 1. Pobranie predykcji z modeli ML
    predictions = engine.predict(image)
    
    # 2. Generowanie werdyktu przez LLM (DeepSeek)
    prompt = f"""
    Jesteś ekspertem ds. cyberbezpieczeństwa i detekcji deepfake'ów. 
    Przeanalizuj poniższe wyniki z 4 modeli sztucznej inteligencji badających jedno zdjęcie i napisz krótki, 
    zrozumiały dla laika werdykt (max 3-4 zdania), czy obraz jest wygenerowany przez AI, czy prawdziwy.
    
    Wyniki modeli (prawdopodobieństwo, że to FAKE):
    - Model CLIP (analiza semantyki i logiki): {predictions.get('clip_prob', 0) * 100:.1f}%
    - Model FFT (analiza ukrytych częstotliwości i szumu generatora): {predictions.get('fft_prob', 0) * 100:.1f}%
    - Model RGB (analiza artefaktów na pikselach): {predictions.get('rgb_prob', 0) * 100:.1f}%
    - Model Noise (anomalie w naturalnym szumie matrycy): {predictions.get('noise_prob', 0) * 100:.1f}%
    
    Napisz werdykt w języku polskim, w profesjonalnym ale przystępnym tonie. 
    Zwróć uwagę na to, które modele są najbardziej pewne (np. FFT na 100% to bardzo silny sygnał).
    """
    
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Jesteś pomocnym asystentem analitykiem."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        # Dodajemy werdykt LLM do wyników dla Reacta
        predictions['llm_verdict'] = response.choices[0].message.content
    except Exception as e:
        print(f"Błąd DeepSeek API: {e}")
        predictions['llm_verdict'] = "Nie udało się połączyć z modelem językowym, aby wygenerować werdykt."

    return {"status": "success", "predictions": predictions}