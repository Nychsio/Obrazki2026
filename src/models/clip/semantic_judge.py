import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class SemanticJudgeCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze_backbone=True):
        """
        Inicjalizacja Sędziego Semantycznego bazującego na modelu CLIP.
        """
        super(SemanticJudgeCLIP, self).__init__()
        
        # Inicjalizacja procesora (do transformacji obrazów) i samego modelu
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(
            model_name, 
            use_safetensors=True, 
            attn_implementation="eager"  # Wymusza zapisywanie map atencji dla XAI!
        ).vision_model
        
        # Zamrożenie wag modelu CLIP (Transfer Learning - Feature Extraction)
        if freeze_backbone:
            for param in self.clip.parameters():
                param.requires_grad = False
                
        # Głowa klasyfikująca (Binary Classification: Real vs Fake)
        # CLIP ViT-Base-Patch32 zwraca wektor o rozmiarze 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Zapobiega overfittingowi
            nn.Linear(256, 1),
            #nn.Sigmoid()     # Zwraca prawdopodobieństwo (0.0 - Real, 1.0 - Fake)
        )

    def forward(self, pixel_values):
        """
        Przejście w przód (Forward pass).
        """
        # Ekstrakcja cech przez vision model z CLIP
        outputs = self.clip(pixel_values=pixel_values)
        
        # Pobranie osadzenia z tokena [CLS] (globalna reprezentacja obrazu)
        pooled_output = outputs.pooler_output 
        
        # Klasyfikacja: zwracamy surowe logity (bez Sigmoid)
        logits = self.classifier(pooled_output)
        return logits