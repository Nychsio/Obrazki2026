# Frontend - AuthentiScan Dashboard

## Przegląd

Frontend projektu AuthentiScan to nowoczesna aplikacja webowa napisana w React z Vite, która służy jako interfejs użytkownika dla systemu detekcji obrazów generowanych przez AI. Aplikacja umożliwia przesyłanie obrazów, wyświetlanie wyników analizy z pięciu różnych modeli ML oraz wizualizację XAI (Explainable AI).

## Architektura Techniczna

### Stack Technologiczny
- **Framework**: React 19.2.4 + Vite 8.0.4
- **Styling**: Tailwind CSS 4.2.2 z custom dark theme
- **HTTP Client**: Axios 1.15.0
- **Ikony**: Lucide React 1.8.0
- **Markdown**: React Markdown 10.1.0
- **Linting**: ESLint 9.39.4
- **Build Tool**: Vite z pluginem React

### Struktura Projektu
```
frontend/frontend/
├── public/                    # Statyczne zasoby
│   ├── favicon.svg
│   └── icons.svg
├── src/
│   ├── App.jsx               # Główny komponent aplikacji
│   ├── App.css               # Globalne style
│   ├── index.css             # Tailwind imports
│   ├── main.jsx              # Punkt wejścia
│   ├── assets/               # Obrazy, fonty
│   ├── components/           # Komponenty React
│   │   ├── ModelCard.jsx
│   │   ├── ModelModal.jsx
│   │   └── ModelDetailTab.jsx
│   └── data/                 # Dane statyczne
│       └── modelsInfo.js
├── package.json              # Zależności i skrypty
├── vite.config.js            # Konfiguracja Vite
├── eslint.config.js          # Konfiguracja ESLint
└── index.html                # Szablon HTML
```

## Kluczowe Funkcjonalności

### 1. Przesyłanie Obrazów
- Drag & drop interface z preview
- Obsługa formatów: JPEG, PNG, WebP
- Walidacja rozmiaru i typu pliku
- Responsywny design dla wszystkich urządzeń

### 2. Analiza Ensemble
- Integracja z backendem FastAPI (`http://localhost:8000/api/v1/analyze`)
- Wyświetlanie wyników z 5 modeli:
  - **Model RGB** (CNN - analiza pikseli)
  - **Model CLIP** (ViT - analiza semantyczna)
  - **Model FFT** (analiza częstotliwościowa)
  - **Model Noise** (analiza szumu PRNU)
  - **Model PCA** (analiza gradientów)

### 3. Wizualizacja XAI (Explainable AI)
- Dynamiczne przełączanie między modelami
- Mapy ciepła Grad-CAM dla modelu RGB
- Attention maps dla modelu CLIP
- Wizualizacje widma FFT
- Elipsy wariancji dla PCA
- Saliency maps dla modelu Noise

### 4. Werdykt LLM
- Integracja z DeepSeek API
- Automatyczne generowanie werdyktu w języku polskim
- Analiza wyników wszystkich modeli
- Profesjonalny ton dla użytkowników końcowych

### 5. Dashboard i Statystyki
- Siatka wyników z procentami prawdopodobieństwa
- Kolorystyka: zielony (prawdziwe) / czerwony (fake)
- Techniczne metryki (czas opóźnienia, VRAM)
- Responsywny układ grid

## Komponenty

### App.jsx (Główny Komponent)
- Zarządza stanem aplikacji (file, predictions, loading, error)
- Obsługa przesyłania plików i analizy
- Renderowanie interfejsu użytkownika
- Integracja z API backendu

### ModelCard.jsx
- Wyświetla pojedynczy model z ikoną i wynikiem
- Kolorystyka w zależności od wyniku
- Responsywny design

### ModelModal.jsx
- Modal z szczegółami modelu
- Opis metodologii i interpretacji XAI
- Techniczne szczegóły implementacji

### ModelDetailTab.jsx
- Dynamiczne karty szczegółów dla każdego modelu
- Przełączanie między różnymi wizualizacjami XAI

## Styling i Design

### Motyw Kolorystyczny
- **Główny background**: `#0f172a` (ciemny granat)
- **Tekst główny**: `#f8fafc` (jasny szary)
- **Akcent**: `#38bdf8` (niebieski)
- **Highlight**: `#94a3b8` (szary)
- **Borders**: `#334155` (ciemny szary)

### Responsywność
- Mobile-first design
- Grid system z Tailwind
- Breakpoints: sm (640px), md (768px), lg (1024px), xl (1280px)
- Flexbox dla układów

## Integracja z Backendem

### Endpoint API
```javascript
POST http://localhost:8000/api/v1/analyze
Content-Type: multipart/form-data
Body: { file: File }
```

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
    "llm_verdict": "Werdykt DeepSeek..."
  }
}
```

## Skrypty i Komendy

### Development
```bash
cd frontend/frontend
npm install          # Instalacja zależności
npm run dev          # Uruchomienie dev server (localhost:5173)
```

### Build
```bash
npm run build        # Build produkcyjny
npm run preview      # Preview buildu
npm run lint         # Sprawdzenie kodu ESLint
```

### Zależności
```json
{
  "dependencies": {
    "@tailwindcss/vite": "^4.2.2",
    "axios": "^1.15.0",
    "lucide-react": "^1.8.0",
    "react": "^19.2.4",
    "react-dom": "^19.2.4",
    "react-markdown": "^10.1.0"
  }
}
```

## Konfiguracja Środowiska

### Vite Config (`vite.config.js`)
```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
})
```

### Tailwind Config (wbudowany w Vite 4)
- Używa najnowszej wersji Tailwind CSS z natywną integracją Vite
- Custom theme zdefiniowany w `index.css`

## Known Issues i TODO

### ✅ Działające
- Podstawowy flow przesyłania i analizy
- Wyświetlanie wyników wszystkich modeli
- XAI visualizations dla większości modeli
- Responsywny design

### ⚠️ Wymagające Poprawek
- Brak obsługi błędów sieciowych (retry, timeout)
- Brak cache'owania wyników
- Brak TypeScript (planowana migracja)
- Brak testów jednostkowych

### 🚀 Planowane Rozszerzenia
1. **TypeScript Migration** - lepsze type safety
2. **Unit Tests** - React Testing Library
3. **E2E Tests** - Cypress
4. **PWA Support** - offline capabilities
5. **Internationalization** - wielojęzyczność
6. **Dark/Light Theme Toggle**
7. **Batch Processing** - analiza wielu obrazów
8. **History Panel** - zapis poprzednich analiz

## Best Practices

### Performance
- Lazy loading komponentów
- Memoization z React.memo i useMemo
- Image optimization z Vite
- Code splitting

### Security
- Sanitization inputów
- CORS configuration
- Environment variables dla API keys
- HTTPS w produkcji

### Accessibility
- Semantic HTML
- ARIA labels
- Keyboard navigation
- Screen reader support

## Deployment

### Build dla Produkcji
```bash
npm run build
```

### Hosting Options
1. **Vercel** (zalecane dla React)
2. **Netlify**
3. **GitHub Pages**
4. **Docker Container**

### Docker
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 5173
CMD ["npm", "run", "preview"]
```

---

*Dokumentacja zaktualizowana: 2026-04-16*  
*Autor: Zespół AuthentiScan*  
*Cel: Kompleksowy przewodnik po frontendzie dla developerów i maintainerów*