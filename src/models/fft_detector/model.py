import torch
import torch.nn as nn
import torch.fft

class FFTDeepfakeDetector(nn.Module):
    def __init__(self, image_size=224):
        super(FFTDeepfakeDetector, self).__init__()
        
        # --- TOR 1: Analiza 2D (Obraz Widma dla GAN-ów) ---
        self.cnn_2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ) # Wypluwa wektor 128-wymiarowy

        # --- TOR 2: Analiza 1D (Widmo Radialne dla Modeli Dyfuzyjnych) ---
        # Zakładamy domyślnie obraz 224x224, więc promień to ok. 112
        self.max_radius = image_size // 2
        self.mlp_1d = nn.Sequential(
            nn.Linear(self.max_radius, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU()
        ) # Wypluwa wektor 64-wymiarowy

        # --- GŁÓWNA FUZJA (Połączenie 2D i 1D) ---
        # 128 z CNN + 64 z MLP = 192
        self.fusion_classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def extract_fft_features(self, x):
        """
        Główny silnik FFT. Zwraca log-widmo 2D oraz profil radialny 1D.
        """
        B, C, H, W = x.shape
        
        # 1. Konwersja do Y (Grayscale) w locie
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
        # 2. 2D FFT & Shift
        fft_result = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft_result, dim=(-2, -1))
        
        # 3. Log-Amplituda (Zgodnie z badaniami używamy log1p dla bezpieczeństwa)
        magnitude = torch.abs(fft_shift)
        log_magnitude = torch.log1p(magnitude)
        
        # Normalizacja 2D
        mean_2d = log_magnitude.mean(dim=(1, 2, 3), keepdim=True)
        std_2d = log_magnitude.std(dim=(1, 2, 3), keepdim=True) + 1e-8
        spectrum_2d = (log_magnitude - mean_2d) / std_2d
        
        # 4. Widmo Radialne (Radial Profile 1D)
        # Tworzymy siatkę współrzędnych, żeby policzyć odległość od środka
        center_y, center_x = H // 2, W // 2
        y, x_grid = torch.meshgrid(torch.arange(H, device=x.device), 
                                   torch.arange(W, device=x.device), indexing='ij')
        r = torch.sqrt((x_grid - center_x)**2 + (y - center_y)**2)
        r_int = torch.round(r).long()
        
        # Maska tylko dla ważnych promieni (od środka do krawędzi)
        valid_mask = r_int < self.max_radius
        r_int_flat = r_int[valid_mask]
        
        # Liczymy średnią moc dla każdego promienia dla całego batcha
        radial_profiles = []
        for i in range(B):
            mag_flat = log_magnitude[i, 0][valid_mask]
            # Używamy bincount do szybkiego sumowania wartości na tych samych promieniach
            bin_sum = torch.bincount(r_int_flat, weights=mag_flat, minlength=self.max_radius)
            bin_count = torch.bincount(r_int_flat, minlength=self.max_radius)
            profile = bin_sum / (bin_count + 1e-8)
            radial_profiles.append(profile)
            
        radial_1d = torch.stack(radial_profiles, dim=0) # [B, max_radius]
        
        # Normalizacja 1D (per profil)
        mean_1d = radial_1d.mean(dim=1, keepdim=True)
        std_1d = radial_1d.std(dim=1, keepdim=True) + 1e-8
        radial_1d = (radial_1d - mean_1d) / std_1d
        
        return spectrum_2d, radial_1d

    def forward(self, x):
        # Wejście to czyste zdjęcie RGB!
        with torch.no_grad():
            spectrum_2d, radial_1d = self.extract_fft_features(x)
            
        # Przepuszczamy 2D przez CNN
        feat_2d = self.cnn_2d(spectrum_2d)
        
        # Przepuszczamy 1D przez MLP
        feat_1d = self.mlp_1d(radial_1d)
        
        # Fuzja - łączymy wnioski z obu analiz
        combined = torch.cat([feat_2d, feat_1d], dim=1)
        logits = self.fusion_classifier(combined)
        
        return logits