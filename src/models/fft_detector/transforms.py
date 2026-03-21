import torch
import torchvision.transforms.functional as F
from torchvision import transforms

class ComplexFourierTransform:
    """
    Konwertuje obraz do widma amplitudowego Fouriera (Magnitude Spectrum).
    Idealne do wychwytywania artefaktów wysokich częstotliwości z modeli generatywnych.
    """
    def __call__(self, img):
        # 1. Konwersja do skali szarości i na tensor
        img_gray = F.rgb_to_grayscale(img)
        tensor_img = F.to_tensor(img_gray) # Kształt: [1, H, W]
        
        # 2. 2D Fast Fourier Transform
        fft_complex = torch.fft.fft2(tensor_img)
        
        # 3. Przesunięcie zerowej częstotliwości na środek (Shift)
        fft_shifted = torch.fft.fftshift(fft_complex)
        
        # 4. Obliczenie magnitudy (wartości bezwzględnej)
        magnitude = torch.abs(fft_shifted)
        
        # 5. Skala logarytmiczna (dodajemy małą wartość epsilon, aby uniknąć log(0))
        magnitude_log = torch.log(magnitude + 1e-8)
        
        # 6. Normalizacja min-max do przedziału [0, 1] dla stabilności sieci
        min_val = magnitude_log.min()
        max_val = magnitude_log.max()
        denom = max_val - min_val
        eps = 1e-8
        if torch.is_tensor(denom):
            # denom może być tensoralną skalarną
            if denom.abs() < eps:
                normalized_magnitude = torch.zeros_like(magnitude_log)
            else:
                normalized_magnitude = (magnitude_log - min_val) / (denom + eps)
        else:
            # fallback dla nie-tensorów
            if abs(denom) < eps:
                normalized_magnitude = torch.zeros_like(magnitude_log)
            else:
                normalized_magnitude = (magnitude_log - min_val) / (denom + eps)
        
        # 7. Obliczenie fazy i normalizacja z przedziału [-pi, pi] do [0, 1]
        phase = torch.angle(fft_shifted)
        normalized_phase = (phase + torch.pi) / (2 * torch.pi)

        # 8. Połączenie amplitudy i fazy w tensor 2-kanałowy: [2, H, W]
        return torch.cat((normalized_magnitude, normalized_phase), dim=0)