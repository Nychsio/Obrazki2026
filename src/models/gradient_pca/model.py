import torch
import torch.nn as nn
from src.models.gradient_pca.extractor import GradientCovarianceExtractor

class GradientPCADetector(nn.Module):
    def __init__(self, device=None):
        super(GradientPCADetector, self).__init__()
        # Inicjalizacja naszego matematycznego ekstraktora
        self.extractor = GradientCovarianceExtractor(device=device)
        
        # Lekki klasyfikator (Micro-MLP)
        self.classifier = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),  # Stabilizuje trening dla małych cech
            nn.ReLU(),
            nn.Dropout(0.2),     # Zapobiega przeuczeniu (Overfitting)
            nn.Linear(16, 1)     # Zwraca 1 logit (Real vs Fake)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor obrazów RGB [B, 3, H, W]
        """
        # 1. Ekstrakcja cech matematycznych (bez śledzenia gradientów dla wag Sobela)
        with torch.no_grad():
            features = self.extractor(x)  # Zwraca [B, 4]
            
        # 2. Klasyfikacja
        logits = self.classifier(features) # Zwraca [B, 1]
        
        return logits