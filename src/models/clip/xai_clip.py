import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
import matplotlib.pyplot as plt

# Importujemy naszego sędziego
from semantic_judge import SemanticJudgeCLIP

class CLIPExplainer:
    def __init__(self, model_path="checkpoints/clip_classifier_best.pth", device=None):
        """
        Inicjalizacja modułu XAI dla Sędziego CLIP.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Ładowanie modelu i wag
        self.model = SemanticJudgeCLIP(freeze_backbone=True).to(self.device)
        try:
            # Wczytujemy wagi naszej nauczonej głowy
            self.model.classifier.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print("Wczytano wagi klasyfikatora.")
        except FileNotFoundError:
            print("Brak pliku z wagami. Model wygeneruje mapę atencji bazując na pre-trenowanym CLIPie (bez finetuningu głowy).")
        
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def generate_heatmap(self, image_path, save_path="heatmap_output.jpg"):
        """
        Generuje mapę ciepła pokazującą, na które fragmenty obrazu zwraca uwagę model.
        """
        # 1. Wczytanie i przygotowanie obrazu
        original_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=original_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 2. Wyciągamy prawdopodobieństwo (wynik modelu)
            logits = self.model(inputs.pixel_values)
            prob = torch.sigmoid(logits).item()
            is_fake = prob > 0.5
            label_str = f"FAKE ({prob*100:.1f}%)" if is_fake else f"REAL ({(1-prob)*100:.1f}%)"

            # 3. Magia XAI: Pobieramy mapy atencji bezpośrednio z CLIPa
            # Wymuszamy zwrócenie atencji (output_attentions=True)
            outputs = self.model.clip(pixel_values=inputs.pixel_values, output_attentions=True)
            
            # Pobieramy atencje z OSTATNIEJ warstwy Transformera
            # Kształt: (batch_size, num_heads, sequence_length, sequence_length)
            attentions = outputs.attentions[-1] 
            
            # Uśredniamy po wszystkich głowach (heads) atencji dla pierwszego (i jedynego) obrazka w batchu
            attn_weights = torch.mean(attentions[0], dim=0) # Kształt: (50, 50)
            
            # Interesuje nas token [CLS] (indeks 0) i jego uwaga na wszystkie inne patche obrazu (od indeksu 1 do 50)
            # W modelu ViT-Base-Patch32 obraz 224x224 jest cięty na siatkę 7x7 patchy (razem 49 patchy)
            cls_attention = attn_weights[0, 1:] # Kształt: (49,)
            
            # Przekształcamy 1D wektor z powrotem na siatkę 2D (7x7)
            spatial_attention = cls_attention.reshape(7, 7).cpu().numpy()

        # 4. Przetwarzanie obrazu za pomocą OpenCV
        # Skalujemy atencję z przedziału 0-1, żeby miała wartości od 0 do 255
        spatial_attention = (spatial_attention - spatial_attention.min()) / (spatial_attention.max() - spatial_attention.min())
        spatial_attention = np.uint8(255 * spatial_attention)

        # Wczytujemy oryginał do CV2 (konwersja z formatu PIL RGB na CV2 BGR)
        img_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        
        # Zmieniamy rozmiar mapy ciepła z 7x7 na oryginalny rozmiar obrazka za pomocą interpolacji
        heatmap_resized = cv2.resize(spatial_attention, (img_cv2.shape[1], img_cv2.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Nakładamy kolory na mapę ciepła (JET colormap - czerwony to duża uwaga, niebieski to mała)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # Mieszamy oryginalny obraz (70%) z mapą ciepła (30%)
        superimposed_img = cv2.addWeighted(img_cv2, 0.7, heatmap_color, 0.3, 0)

        # Dodajemy tekst z wynikiem klasyfikacji na obrazku
        color = (0, 0, 255) if is_fake else (0, 255, 0) # Czerwony dla Fake, Zielony dla Real
        cv2.putText(superimposed_img, label_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # 5. Zapisujemy wynik
        cv2.imwrite(save_path, superimposed_img)
        print(f"✅ Zapisano mapę ciepła w: {save_path} | Wynik: {label_str}")

# Szybki test
if __name__ == "__main__":
    print("Uruchamiam moduł XAI...")
    explainer = CLIPExplainer()
    explainer.generate_heatmap("test_image.jpg", "wynik_xai.jpg")
    print("Moduł XAI gotowy do użycia.")