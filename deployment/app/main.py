from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deployment.app.engine import InferenceEngine
from PIL import Image
import io
import os
import json # Dodaliśmy import JSON
from dotenv import load_dotenv
from openai import AsyncOpenAI
import re

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
    
    # --- NOWOŚĆ: Wyciągamy fizyczne statystyki obrazu ---
    img_width, img_height = image.size
    img_megapixels = (img_width * img_height) / 1000000
    file_size_kb = len(contents) / 1024
    # ----------------------------------------------------

    # 1. Pobranie predykcji z modeli ML
    predictions = engine.predict(image)
    
    # Wyciągamy label z CLIPa (jeśli nie ma, dajemy domyślny)
    semantic_context = predictions.get('clip_label', 'Brak szczegółowej analizy semantycznej')

    # --- NOWOŚĆ: Funkcja do liczenia pewności modelu (Confidence) ---
    def get_confidence(prob):
        # Odległość od 0.5 (zgadywanie) pomnożona x2 daje nam pewność od 0% do 100%
        return abs(prob - 0.5) * 200

    rgb_p = predictions.get('rgb_prob', 0)
    clip_p = predictions.get('clip_prob', 0)
    fft_p = predictions.get('fft_prob', 0)
    pca_p = predictions.get('pca_prob', 0)
    noise_p = predictions.get('noise_prob', 0)

    # AKTUALIZACJA PROMPTU DLA DEEPSEEK (Wzbogacony o statystyki)
    prompt = f"""
    ANALIZA EKSPERCKA OBRAZU:
    
    PARAMETRY FIZYCZNE PLIKU (METADANE):
    - Rozdzielczość: {img_width}x{img_height} ({img_megapixels:.1f} Mpx)
    - Waga pliku: {file_size_kb:.1f} KB
    - Kontekst Semantyczny (CLIP): "{semantic_context}" 
    
    SYGNAŁY TECHNICZNE (Prawdopodobieństwo FAKE oraz Pewność Modelu):
    - RGB (Piksele): {rgb_p*100:.1f}% FAKE | Pewność diagnozy: {get_confidence(rgb_p):.1f}%
    - CLIP (Logika): {clip_p*100:.1f}% FAKE | Pewność diagnozy: {get_confidence(clip_p):.1f}%
    - FFT (Częstotliwości): {fft_p*100:.1f}% FAKE | Pewność diagnozy: {get_confidence(fft_p):.1f}%
    - PCA (Gradienty): {pca_p*100:.1f}% FAKE | Pewność diagnozy: {get_confidence(pca_p):.1f}%
    - Noise (Szum matrycy): {noise_p*100:.1f}% FAKE | Pewność diagnozy: {get_confidence(noise_p):.1f}%

    ŻELAZNE ZASADY WNIOSKOWANIA (ZASTOSUJ BEZWZGLĘDNIE):
    1. ROZDZIELCZOŚĆ A SZUM: Jeśli obraz ma poniżej 1 Mpx lub waży bardzo mało (duża kompresja), modele PCA i Noise mogą wariować. Jeśli ich pewność jest niska (poniżej 40%), uznaj to za kompresję JPG, a nie AI.
    2. TWARDE VETO (FAKE): Tylko jeśli główny model pikseli (RGB) LUB logiczny (CLIP) wskazuje na FAKE z PEWNOŚCIĄ powyżej 60%.
    3. AUTENTYCZNE ZDJĘCIE (REAL): Jeśli wskaźniki techniczne (RGB, FFT) są poniżej 40% FAKE.
    4. IGNORUJ ZGADYWANIE: Jeśli jakikolwiek model ma "Pewność diagnozy" poniżej 20% (wyniki bliskie 50%), oznacza to, że model zgaduje. Zignoruj jego wskazanie w ostatecznym werdykcie.

    WYMOGI ZWROTNE (STRICT JSON FORMAT):
    Musisz wygenerować odpowiedź WYŁĄCZNIE jako obiekt JSON. Zastosuj się do poniższej struktury:
    {{
        "global_verdict": "Ostateczny werdykt (max 3-4 zdania). Oceń wpływ kompresji/rozdzielczości na ewentualne błędy modeli. Chłodny, techniczny ton.",
        "rgb_llm": "Jedno techniczne zdanie analizy. Zwróć uwagę na pewność modelu.",
        "clip_llm": "Jedno techniczne zdanie analizy semantyki i pewności modelu.",
        "fft_llm": "Jedno techniczne zdanie analizy. Skoreluj z kompresją pliku jeśli to istotne.",
        "pca_llm": "Jedno techniczne zdanie analizy gradientów.",
        "noise_llm": "Jedno techniczne zdanie analizy szumu. Uwzględnij rozdzielczość i kompresję."
    }}
    """
    
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Jesteś surowym, analitycznym API. Twoja odpowiedź musi być zawsze poprawnym formatem JSON, bez żadnych znaczników markdown czy tekstu poza JSONem."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # Zabezpieczenie zmuszające model do zwrotu JSON
            max_tokens=1700,
            temperature=0.1
        )
        
        # Pobieramy tekst
        raw_json_text = response.choices[0].message.content
        
        # --- DEBUG: Wyświetlamy to w terminalu uvicorna ---
        print("\n--- SUROWA ODPOWIEDŹ LLM ---")
        print(raw_json_text)
        print("----------------------------\n")
        
        # --- KULOODPORNA SZCZOTKA REGEX ---
        # Szukamy wszystkiego od pierwszej klamry '{' do ostatniej klamry '}'
        match = re.search(r'\{.*\}', raw_json_text, re.DOTALL)
        
        if not match:
            raise ValueError("Nie znaleziono struktury JSON w odpowiedzi!")
            
        clean_json_text = match.group(0)
        llm_data = json.loads(clean_json_text)
        
        # Wypychamy rozpakowane dane do słownika predictions, żeby React mógł to odebrać
        predictions['llm_verdict'] = llm_data.get('global_verdict', 'Błąd generowania werdyktu globalnego.')
        predictions['rgb_llm'] = llm_data.get('rgb_llm', 'Brak szczegółowej analizy RGB.')
        predictions['clip_llm'] = llm_data.get('clip_llm', 'Brak szczegółowej analizy CLIP.')
        predictions['fft_llm'] = llm_data.get('fft_llm', 'Brak szczegółowej analizy FFT.')
        predictions['pca_llm'] = llm_data.get('pca_llm', 'Brak szczegółowej analizy PCA.')
        predictions['noise_llm'] = llm_data.get('noise_llm', 'Brak szczegółowej analizy Noise.')

    except json.JSONDecodeError as e:
        print(f"🛑 Błąd parsowania JSON! Szczegóły: {e}")
        predictions['llm_verdict'] = "Błąd systemu: Model językowy zwrócił uszkodzony JSON (np. niepoprawne znaki)."
    except Exception as e:
        print(f"🛑 Błąd DeepSeek API: {e}")
        predictions['llm_verdict'] = "Nie udało się połączyć z modelem językowym, aby wygenerować werdykt."

    return {"status": "success", "predictions": predictions}