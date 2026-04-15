import torch
import torch.nn as nn
from src.models.gradient_pca.extractor import MultiScaleStructureTensor

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.SiLU() # SOTA Activation
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() # SOTA Activation instead of ReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Mechanizm Uwagi (Attention) - Model sam wie na co patrzeć!
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += self.shortcut(x)
        return self.act(out)

class GradientPCADetector(nn.Module):
    def __init__(self, device=None):
        super(GradientPCADetector, self).__init__()
        
        # Ekstraktor z IEEE (Multi-scale YCbCr)
        self.extractor = MultiScaleStructureTensor(device=device)
        
        # Głębsza, bardziej zaawansowana sieć z CBAM i SiLU
        self.features = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            CBAMResidualBlock(64, 128),
            nn.MaxPool2d(2),
            
            CBAMResidualBlock(128, 256),
            nn.MaxPool2d(2),
            
            nn.Dropout2d(0.2) # Spatial Dropout (SOTA dla obrazów)
        )
        
        # XAI (Explainable AI) - Warstwa konwertująca na mapę cieplną (Heatmap)
        self.xai_heatmap_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, return_heatmap=False):
        with torch.no_grad():
            pca_features = self.extractor(x)
            
        deep_features = self.features(pca_features)
        heatmap = self.xai_heatmap_conv(deep_features)
        logits = self.global_pool(heatmap).view(-1, 1)
        
        if return_heatmap:
            return logits, heatmap
            
        return logits