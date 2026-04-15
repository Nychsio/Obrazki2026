import torch
import torch.nn as nn
from src.models.gradient_pca.extractor import MultiScaleStructureTensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class GradientPCADetector(nn.Module):
    def __init__(self, device=None):
        super(GradientPCADetector, self).__init__()
        
        # Ekstraktor z wiedzy naukowej IEEE (Multi-scale YCbCr)
        self.extractor = MultiScaleStructureTensor(device=device)
        
        # Ekstrakcja głębokich cech - w pełni konwolucyjna (zachowuje przestrzeń!)
        self.features = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            # Tutaj kończymy zabawę głębokimi filtrami. Mamy tensor [Batch, 256, H', W']
        )
        
        # === MAGIA XAI (Explainable AI) ===
        # Zamiast Flatten i Linear, używamy konwolucji 1x1. 
        # Zgniata ona 256 kanałów matematycznych bezpośrednio w 1 KANAŁ ("MAPĘ FAŁSZU").
        self.xai_heatmap_conv = nn.Conv2d(256, 1, kernel_size=1)
        
        # Global Pooling używamy tylko do wyciągnięcia 1 ostatecznego werdyktu dla całego obrazu
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, return_heatmap=False):
        # 1. Obliczenie Tensorów Struktury
        with torch.no_grad():
            pca_features = self.extractor(x)
            
        # 2. Szukanie wzorców przez CNN
        deep_features = self.features(pca_features)
        
        # 3. Wygenerowanie bezpośredniej mapy cieplnej XAI! [B, 1, H', W']
        heatmap = self.xai_heatmap_conv(deep_features)
        
        # 4. Wyciągnięcie ostatecznego logitu (średnia z całej mapy cieplnej)
        logits = self.global_pool(heatmap).view(-1, 1)
        
        if return_heatmap:
            return logits, heatmap
            
        return logits
