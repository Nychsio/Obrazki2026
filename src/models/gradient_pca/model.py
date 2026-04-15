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
        
        # Shortcut na wypadek zmiany liczby kanałów
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
        
        # Nasz nowy, potężny ekstraktor matematyczny
        self.extractor = MultiScaleStructureTensor(device=device)
        
        # Classifier oparty na głębokich konwolucjach do szukania wzorców na "mapie" kowariancji
        # Wejście ma 10 kanałów matematycznych (4 z 8x8, 4 z 16x16, 2 z chrominancji)
        self.classifier = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            
            ResidualBlock(128, 256),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1)), # Niezależne od ostatecznego rozmiaru wejścia
            nn.Flatten(),
            
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ekstrakcja Tensorów Struktury bez śledzenia gradientów (czysta matematyka)
        with torch.no_grad():
            features = self.extractor(x)
            
        # Głęboka klasyfikacja anomalii przestrzennych
        logits = self.classifier(features)
        
        return logits