import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FFTResNetDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(FFTResNetDetector, self).__init__()
        
        # --- STRUMIEŃ 1: Analiza Przestrzenna (RGB) ---
        self.spatial_stream = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.spatial_stream.fc = nn.Identity() # Usuwamy głowę klasyfikacyjną (zostaje 512 cech)
        
        # --- STRUMIEŃ 2: Analiza Częstotliwości (FFT 2-kanałowe) ---
        self.frequency_stream = resnet18(weights=ResNet18_Weights.DEFAULT)
        original_conv1 = self.frequency_stream.conv1
        self.frequency_stream.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Adaptacja 3 kanałów na 2 kanały dla ResNeta
            self.frequency_stream.conv1.weight = nn.Parameter(
                torch.mean(original_conv1.weight, dim=1, keepdim=True).repeat(1, 2, 1, 1)
            )
        self.frequency_stream.fc = nn.Identity() # Zostaje 512 cech
        
        # --- BLOK FUZJI (Połączenie wniosków) ---
        # 512 cech z RGB + 512 cech z FFT = 1024 wejścia
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_rgb, x_fft):
        # Przepuszczamy dane przez oba strumienie
        feat_spatial = self.spatial_stream(x_rgb)
        feat_frequency = self.frequency_stream(x_fft)
        
        # Łączymy cechy i klasyfikujemy
        fused_features = torch.cat((feat_spatial, feat_frequency), dim=1)
        return self.classifier(fused_features)