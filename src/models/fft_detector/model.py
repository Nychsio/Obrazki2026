import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FFTResNetDetector(nn.Module):
    def __init__(self, num_classes=1): # 1 klasa dla klasyfikacji binarnej (Fake/Real)
        super(FFTResNetDetector, self).__init__()
        
        # Ładujemy pretrenowany model (Transfer Learning przyspieszy zbieżność)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modulacja pierwszej warstwy dla 2 kanałów (Amplitude + Phase) zamiast 3 RGB
        # Uśredniamy wagi wzdłuż osi kanałów i powielamy je do 2 kanałów wejściowych
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Zachowujemy skalę wag pretrenowanego modelu i dopasowujemy kształt do [64, 2, 7, 7]
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                torch.mean(original_conv1.weight, dim=1, keepdim=True).repeat(1, 2, 1, 1)
            )
            
        # Zmiana ostatniej warstwy w pełni połączonej (Linear) do klasyfikacji binarnej
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)