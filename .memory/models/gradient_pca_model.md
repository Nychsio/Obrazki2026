# Gradient PCA Model Documentation

## Przegląd

Model Gradient PCA to piąty agent w architekturze "Obrazy Destylacja". Jego celem jest ekstrakcja cech gradientowych z obrazów RGB poprzez analizę macierzy kowariancji gradientów, które następnie mogą być użyte do redukcji wymiarowości za pomocą PCA.

## Architektura

### GradientCovarianceExtractor

Główna klasa ekstraktora wykonuje następujące kroki:

1. **Konwersja RGB na luminancję**:
   - Wagi ITU-R BT.709: `[0.2126, 0.7152, 0.0722]`
   - Tensor wejściowy: `[B, 3, H, W]` → Tensor wyjściowy: `[B, 1, H, W]`

2. **Obliczenie gradientów**:
   - Filtry Sobela dla gradientów poziomych (G_x) i pionowych (G_y)
   - Filtry 3x3 z odpowiednimi wagami
   - Padding='same' dla zachowania wymiarów przestrzennych

3. **Obliczenie macierzy kowariancji**:
   - Dla każdego obrazu w batchu obliczana jest macierz kowariancji 2x2:
     ```
     [[var(G_x), cov(G_x, G_y)],
      [cov(G_x, G_y), var(G_y)]]
     ```
   - Użycie nieobciążonego estymatora (N-1 w mianowniku)

4. **Wyjście**:
   - Spłaszczone macierze kowariancji: `[B, 4]`
   - Kolejność: `[var_G_x, cov_Gx_Gy, cov_Gx_Gy, var_G_y]`

## Interfejs API

### Klasy

#### `GradientCovarianceExtractor`
```python
class GradientCovarianceExtractor:
    def __init__(self, device: Optional[torch.device] = None)
    def __call__(self, rgb_tensor: torch.Tensor) -> torch.Tensor
    def extract_with_intermediates(self, rgb_tensor: torch.Tensor) -> dict
```

### Funkcje

#### `extract_gradient_covariance`
```python
def extract_gradient_covariance(
    rgb_tensor: torch.Tensor, 
    device: Optional[torch.device] = None
) -> torch.Tensor
```

## Użycie

### Podstawowe użycie
```python
from src.models.gradient_pca import GradientCovarianceExtractor

# Inicjalizacja ekstraktora
extractor = GradientCovarianceExtractor()

# Tensor RGB: [batch_size, 3, height, width]
rgb_tensor = torch.randn(4, 3, 224, 224)

# Ekstrakcja cech
covariance_features = extractor(rgb_tensor)  # [4, 4]
```

### Ekstrakcja z wynikami pośrednimi
```python
results = extractor.extract_with_intermediates(rgb_tensor)
# results zawiera:
# - 'luminance': tensor luminancji [B, 1, H, W]
# - 'G_x': gradienty poziome [B, 1, H, W]
# - 'G_y': gradienty pionowe [B, 1, H, W]
# - 'covariance': macierze kowariancji [B, 4]
```

### Funkcja pomocnicza
```python
from src.models.gradient_pca import extract_gradient_covariance

features = extract_gradient_covariance(rgb_tensor)
```

## Wymagania

### Zależności
- PyTorch 2.2+
- torch.nn.functional
- Brak zewnętrznych bibliotek poza PyTorch

### Wymagania sprzętowe
- GPU (CUDA) opcjonalne, ale zalecane
- Automatyczne wykrywanie urządzenia: CUDA jeśli dostępne, w przeciwnym razie CPU

## Testowanie

### Uruchomienie testów
```bash
cd src/models/gradient_pca
python simple_test.py
```

### Testy obejmują:
1. Podstawową funkcjonalność
2. Symetrię macierzy kowariancji
3. Różne rozmiary obrazów
4. Funkcję pomocniczą
5. Wyniki pośrednie

## Integracja z Systemem Ensemble

### Dane wejściowe
- Obrazy RGB znormalizowane do zakresu [0, 1] lub [-1, 1]
- Dowolny rozmiar obrazu (H, W)

### Dane wyjściowe
- 4-wymiarowe cechy na obraz (spłaszczona macierz kowariancji 2x2)
- Możliwość użycia jako wejście do klasyfikatora lub dalszej redukcji wymiarowości PCA

## Znane Ograniczenia

1. **Wydajność**: Pętla for dla każdego obrazu w batchu może być wąskim gardłem dla dużych batchy.
2. **Pamięć**: Przechowywanie pośrednich gradientów może wymagać dodatkowej pamięci.
3. **Stabilność numeryczna**: Dla bardzo małych obrazów (< 3x3) obliczenia kowariancji mogą być niestabilne.

## Planowane Rozszerzenia

1. **Optymalizacja wydajności**: Wykorzystanie operacji wsadowych zamiast pętli for.
2. **Więcej filtrów gradientowych**: Dodanie filtrów Robertsa, Prewitta, Scharra.
3. **Analiza wyższych momentów**: Skosność i kurtoza gradientów.
4. **Integracja z PCA**: Bezpośrednia redukcja wymiarowości w module.

## Status

- ✅ Implementacja podstawowa
- ✅ Testy jednostkowe
- ✅ Dokumentacja
- 🔄 Integracja z systemem ensemble (w trakcie)
- 🔄 Optymalizacja wydajności (planowane)

## Autor

Obrazki Destylacja Project  
Data utworzenia: 2026-04-13  
Ostatnia aktualizacja: 2026-04-13