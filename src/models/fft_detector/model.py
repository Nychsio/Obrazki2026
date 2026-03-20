import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FFTResNetDetector(nn.Module):
    def __init__(self, num_classes=1): # 1 klasa dla klasyfikacji binarnej (Fake/Real)
        super(FFTResNetDetector, self).__init__()
        
        # Ładujemy pretrenowany model (Transfer Learning przyspieszy zbieżność)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modulacja pierwszej warstwy dla 1 kanału (zamiast 3 RGB)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Używamy średniej po kanałach zamiast sumy, by zachować skalę wag pretrenowanego modelu
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                torch.mean(original_conv1.weight, dim=1, keepdim=True)
            )
            
        # Zmiana ostatniej warstwy w pełni połączonej (Linear) do klasyfikacji binarnej
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)