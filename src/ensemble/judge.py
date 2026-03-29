"""
MetaJudgeEnsemble - System późnej fuzji (Late Fusion) dla 4 modeli detekcji.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import warnings

# Importy modeli bazowych
try:
    from src.noise.model import NoiseBinaryClassifier
    from src.rgb.train import RGBClassifier
    from src.models.clip.semantic_judge import SemanticJudgeCLIP
    from src.models.fft_detector.model import FFTResNetDetector
except ImportError as e:
    warnings.warn(f"Cannot import base models: {e}")
    # Placeholder definitions for sanity check
    class NoiseBinaryClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.classifier = nn.Linear(32 * 224 * 224, 1)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    class RGBClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.classifier = nn.Linear(32 * 224 * 224, 1)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    class SemanticJudgeCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.classifier = nn.Linear(32 * 224 * 224, 1)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    class FFTResNetDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
            self.classifier = nn.Linear(32 * 224 * 224, 1)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)


class MetaJudgeEnsemble(nn.Module):
    """Meta-Classifier Ensemble for late fusion of 4 detection models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.noise_model = NoiseBinaryClassifier()
        self.rgb_model = RGBClassifier()
        self.clip_model = SemanticJudgeCLIP()
        self.fft_model = FFTResNetDetector()
        
        self._freeze_base_models()
        
        self.fusion_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        
        self.to(self.device)
    
    def _freeze_base_models(self):
        for model in [self.noise_model, self.rgb_model, self.clip_model, self.fft_model]:
            for param in model.parameters():
                param.requires_grad = False
    
    def load_base_weights(self, weight_paths: Dict[str, str]):
        required_keys = ['noise', 'rgb', 'clip', 'fft']
        
        for key in required_keys:
            if key not in weight_paths:
                raise KeyError(f"Missing key '{key}' in weight_paths")
        
        try:
            # Load each model
            for model_name, model in [('noise', self.noise_model),
                                     ('rgb', self.rgb_model),
                                     ('clip', self.clip_model),
                                     ('fft', self.fft_model)]:
                checkpoint = torch.load(weight_paths[model_name], map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Weight file not found: {e}")
        
        for model in [self.noise_model, self.rgb_model, self.clip_model, self.fft_model]:
            model.eval()
        
        print("Base model weights loaded successfully.")
    
    def forward(self, img_noise, img_rgb, img_clip, img_fft):
        img_noise = img_noise.to(self.device)
        img_rgb = img_rgb.to(self.device)
        img_clip = img_clip.to(self.device)
        img_fft = img_fft.to(self.device)
        
        base_logits = []
        
        with torch.no_grad():
            base_logits.append(self.noise_model(img_noise))
            base_logits.append(self.rgb_model(img_rgb))
            base_logits.append(self.clip_model(img_clip))
            base_logits.append(self.fft_model(img_fft))
        
        concatenated = torch.cat(base_logits, dim=1)
        return self.fusion_head(concatenated)
    
    def predict_proba(self, img_noise, img_rgb, img_clip, img_fft):
        with torch.no_grad():
            logits = self.forward(img_noise, img_rgb, img_clip, img_fft)
            return torch.sigmoid(logits)


if __name__ == "__main__":
    print("=== MetaJudgeEnsemble Sanity Check ===")
    
    model = MetaJudgeEnsemble()
    print(f"Device: {model.device}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n--- Checking frozen base models ---")
    trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.append(name)
    
    if trainable:
        print("Trainable parameters:")
        for name in trainable:
            print(f"  {name}")
    else:
        print("All parameters frozen (only fusion_head should be trainable)")
    
    batch_size = 2
    dummy_noise = torch.randn(batch_size, 3, 224, 224)
    dummy_rgb = torch.randn(batch_size, 3, 224, 224)
    dummy_clip = torch.randn(batch_size, 3, 224, 224)
    dummy_fft = torch.randn(batch_size, 2, 224, 224)
    
    print(f"\nDummy tensors created:")
    print(f"  noise: {dummy_noise.shape}")
    print(f"  rgb: {dummy_rgb.shape}")
    print(f"  clip: {dummy_clip.shape}")
    print(f"  fft: {dummy_fft.shape}")
    
    print("\n--- Testing forward pass ---")
    try:
        with torch.no_grad():
            output = model(dummy_noise, dummy_rgb, dummy_clip, dummy_fft)
            print(f"Success! Output shape: {output.shape}")
            print(f"Output values: {output}")
            
            probs = model.predict_proba(dummy_noise, dummy_rgb, dummy_clip, dummy_fft)
            print(f"\nProbabilities shape: {probs.shape}")
            print(f"Probabilities: {probs}")
        
        print("\n[SUCCESS] All sanity checks passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()