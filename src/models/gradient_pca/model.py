import torch
import torch.nn as nn
from src.models.gradient_pca.extractor import MultiScaleStructureTensor

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.SiLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
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
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += self.shortcut(x)
        return self.act(out)

class GradientPCADetector(nn.Module):
    def __init__(self, device=None):
        super(GradientPCADetector, self).__init__()
        self.extractor = MultiScaleStructureTensor(device=device)
        self.features = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            CBAMResidualBlock(64, 128),
            nn.MaxPool2d(2),
            CBAMResidualBlock(128, 256),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.xai_heatmap_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, return_heatmap=False):
        # POPRAWKA CLAUDE: Usunięto no_grad() by pozwolić na głęboki przepływ wsteczny (XAI/Adversarial)
        pca_features = self.extractor(x)
        deep_features = self.features(pca_features)
        heatmap = self.xai_heatmap_conv(deep_features)
        logits = self.global_pool(heatmap).view(-1, 1)
        
        if return_heatmap:
            return logits, heatmap
        return logits