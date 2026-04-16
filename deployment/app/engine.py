from pathlib import Path
import torch
import base64
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from captum.attr import LayerGradCam

# Importy klas modeli
from src.models.fft_detector.model import FFTDeepfakeDetector
from src.models.gradient_pca.model import GradientPCADetector
from src.models.fft_detector.transforms import ComplexFourierTransform
from src.models.rgb.train import RGBClassifier
from src.models.noise.model import NoiseBinaryClassifier

class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.init() # Inicjalizacja kontekstu CUDA dla FastAPI
        
        # Profesjonalne określenie ścieżki do obecnego pliku (engine.py) i folderu models/
        self.base_dir = Path(__file__).resolve().parent
        self.models_dir = self.base_dir / "models"
        self.models = {}
        
        print(f"🔧 Inicjalizacja InferenceEngine na urządzeniu: {self.device}")
        print(f"📂 Szukam wag modeli w: {self.models_dir}")
        
        # Transformacje
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.fft_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            ComplexFourierTransform()
        ])
        
        # Dedykowana transformacja dla RGB (ImageNet Normalization)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transformacja dla modelu Noise
        self.noise_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Ładowanie
        self._load_fft()
        self._load_clip()
        self._load_pca()
        self._load_rgb()
        self._load_noise()

    def _load_fft(self):
        model_path = self.models_dir / "best_fft_model.pt"
        if model_path.exists():
            print("⏳ Ładowanie modelu FFT...")
            model = FFTDeepfakeDetector()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models['fft'] = model
            print("✅ Model FFT załadowany gotowy!")
        else:
            print(f"❌ Brak pliku: {model_path}")

    def _load_clip(self):
        try:
            print("⏳ Ładowanie modelu CLIP (Zero-Shot)...")
            from transformers import CLIPProcessor, CLIPModel
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

            # MAGIA JEST TUTAJ: Zmuszamy model do trybu 'eager', żeby oddał nam macierze Atencji!
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                attn_implementation="eager"
            ).to(self.device)

            self.models['clip'] = self.clip_model
            print("✅ Model CLIP załadowany!")
        except Exception as e:
            print(f"⚠️ Nie udało się załadować CLIP: {e}")

    def _load_pca(self):
        try:
            print("⏳ Ładowanie modelu PCA SOTA (Deep Architecture)...")
            from src.models.gradient_pca.model import GradientPCADetector
            import os
            import torch
            
            # 1. Inicjalizacja architektury
            model = GradientPCADetector()
            
            # 2. Ścieżka do Twojego nowego pliku 4MB
            weights_path = "deployment/app/models/best_pca_model.pt"
            
            if os.path.exists(weights_path):
                # Ładowanie wag - map_location zapewnia, że zadziała na CPU i GPU
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✅ Sukces: Załadowano wagi SOTA ({weights_path})")
            else:
                print(f"❌ Krytyczny błąd: Brak pliku wag w {weights_path}!")

            # 3. Przeniesienie na GPU i tryb ewaluacji
            model.to(self.device).eval()
            self.models['pca'] = model
            
        except Exception as e:
            print(f"⚠️ Nie udało się zainicjować PCA: {e}")

    def _load_rgb(self):
        model_path = self.models_dir / "best_rgb_model.pt"
        if model_path.exists():
            print("⏳ Ładowanie modelu RGB...")
            model = RGBClassifier()
            # Używamy weights_only=False ze względu na zaufane źródło
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models['rgb'] = model
            print("✅ Model RGB załadowany!")
        else:
            print(f"❌ Brak pliku: {model_path}")

    def _load_noise(self):
        model_path = self.models_dir / "best_noise_model.pt"
        if model_path.exists():
            print("⏳ Ładowanie modelu Noise...")
            model = NoiseBinaryClassifier() # Upewnij się, że to poprawna nazwa Twojej klasy
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models['noise'] = model
            print("✅ Model Noise załadowany i gotowy!")
        else:
            print(f"❌ Brak pliku: {model_path}")

    def _generate_noise_xai(self, model, input_tensor, original_image):
        """Generuje mapę Saliency dla modelu Noise"""
        try:
            with torch.enable_grad():
                input_tensor = input_tensor.clone().requires_grad_()
                logits = model(input_tensor)
                logits.backward(torch.ones_like(logits))
                
                gradients = input_tensor.grad.abs().squeeze().cpu().numpy()
                saliency = np.max(gradients, axis=0)
                
                if np.max(saliency) > 0:
                    saliency = saliency / np.max(saliency)
                    
                # Używamy chłodnej palety OCEAN dla analizy szumu
                heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_OCEAN)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                orig_resized = original_image.resize((224, 224))
                orig_np = np.array(orig_resized)
                overlay = cv2.addWeighted(orig_np, 0.4, heatmap, 0.6, 0)
                
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Błąd generowania XAI Noise: {str(e)}")
            return None

    def _generate_fft_vis(self, original_image):
        """Generuje zaawansowaną wizualizację FFT akcentującą artefakty w wysokich pasmach"""
        try:
            # 1. Standardowe Widmo Amplitudowe
            img_gray = cv2.cvtColor(np.array(original_image.resize((224, 224))), cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)
            
            # Normalizacja
            mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Kolorowanie surowego widma (OCEAN dla bazy)
            heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_OCEAN)
            
            # 2. Akcentowanie stref dyfuzji (Wysokie częstotliwości na brzegach)
            h, w = mag_norm.shape
            center_y, center_x = h // 2, w // 2
            
            # Tworzymy maskę odległości od środka (radial mask)
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Wygaszamy środek (naturalne niskie częstotliwości), wzmacniamy krawędzie
            high_freq_mask = np.clip((dist_from_center / (h/2)), 0, 1)
            
            # Nakładamy filtr INFERNO tylko na anomalię w wysokich pasmach
            anomaly_colored = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
            
            # Blendowanie: Baza to OCEAN, krawędzie przechodzą w INFERNO (tam gdzie AI się myli)
            blended = np.zeros_like(heatmap)
            for i in range(3): # Dla każdego kanału BGR
                blended[:, :, i] = heatmap[:, :, i] * (1 - high_freq_mask) + anomaly_colored[:, :, i] * high_freq_mask
                
            # Dodanie subtelnych siatek (okręgów referencyjnych dla pro-wyglądu)
            cv2.circle(blended, (center_x, center_y), h//4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(blended, (center_x, center_y), h//2 - 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', blended)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Błąd generowania FFT XAI: {str(e)}")
            return None

    def _generate_clip_xai(self, model, processor, original_image):
        """Generuje mapę uwagi (Self-Attention) dla Transformera ViT."""
        try:
            with torch.no_grad():
                inputs = processor(text=["surreal AI generated image"], images=original_image, return_tensors="pt").to(self.device)

                # Odpalamy model z flagą output_attentions=True
                outputs = model(**inputs, output_attentions=True)

                # Pobieramy macierze uwagi z OSTATNIEJ warstwy modelu wizyjnego
                vision_attns = outputs.vision_model_output.attentions[-1]

                # Średnia uwaga ze wszystkich 12 "głowic" (heads)
                avg_attn = vision_attns.mean(dim=1).squeeze()

                # Interesuje nas to, jak główny token [CLS] (indeks 0) patrzy na resztę obrazu (indeks 1+)
                cls_attn = avg_attn[0, 1:].cpu().numpy()

                # ViT-Base-Patch32 dzieli obraz na siatkę 7x7 (dla obrazu 224x224)
                grid_size = int(np.sqrt(len(cls_attn)))
                spatial_attn = cls_attn.reshape(grid_size, grid_size)

                # Normalizacja i powiększenie do rozmiaru oryginalnego obrazu (CUBIC daje gładkie rozmycie)
                spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min())
                spatial_attn = cv2.resize(spatial_attn, (original_image.width, original_image.height), interpolation=cv2.INTER_CUBIC)

                # Odcinamy słabe sygnały (tło)
                spatial_attn[spatial_attn < 0.4] = 0

                heatmap = cv2.applyColorMap(np.uint8(255 * spatial_attn), cv2.COLORMAP_INFERNO)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                orig_np = np.array(original_image)
                overlay = cv2.addWeighted(orig_np, 0.5, heatmap, 0.7, 0)

                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                return base64.b64encode(buffer).decode('utf-8')

        except Exception as e:
            print(f"⚠️ Błąd generowania Attention XAI dla CLIP: {str(e)}")
            return None

    def _generate_pca_heatmap(self, heatmap_tensor, original_image):
        """Generuje gładką, profesjonalną mapę PCA tylko dla obszarów anomalii"""
        try:
            # 1. Wyciągamy surową mapę (często bardzo małą, np. 14x14)
            heatmap_np = heatmap_tensor.squeeze().cpu().detach().numpy()
            heatmap_np = np.maximum(heatmap_np, 0)
            
            if np.max(heatmap_np) > 0:
                heatmap_np = heatmap_np / np.max(heatmap_np)
                
            # 2. PŁYNNE SKALOWANIE I ROZMYCIE (Koniec z kwadratami!)
            heatmap_np = cv2.resize(heatmap_np, (original_image.width, original_image.height), interpolation=cv2.INTER_CUBIC)
            heatmap_np = cv2.GaussianBlur(heatmap_np, (11, 11), 0)
            
            # 3. Odcięcie szumu tła (pokazujemy tylko silne sygnały > 30%)
            heatmap_np[heatmap_np < 0.3] = 0
            
            # 4. Kolorowanie (MAGMA wygląda bardzo analitycznie)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_MAGMA)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # 5. Maska Alfa (Wtapianie) - heatmapa jest widoczna tam, gdzie wartość jest wysoka
            alpha = heatmap_np[..., np.newaxis] * 0.75 # Max 75% przezroczystości
            orig_np = np.array(original_image)
            
            # Mieszamy obraz oryginalny z mapą ciepła używając maski
            overlay = (orig_np * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
            
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Błąd generowania SOTA PCA Heatmap: {str(e)}")
            return None

    def _generate_gradcam(self, model, input_tensor, original_image):
        """Generuje gładką mapę ciepła Grad-CAM"""
        try:
            with torch.enable_grad(): 
                input_tensor = input_tensor.clone().requires_grad_()
                target_layer = model.backbone.conv_head
                layer_gc = LayerGradCam(model, target_layer)
                attributions = layer_gc.attribute(input_tensor, target=0)
                
                # Pobieramy numpy array z atrybucjami
                attr_np = attributions.squeeze().cpu().detach().numpy()
                attr_np = np.maximum(attr_np, 0)
                
                if np.max(attr_np) > 0:
                    attr_np = attr_np / np.max(attr_np)
                
                # NOWE: Płynne skalowanie do oryginalnego rozmiaru i mocne rozmycie Gaussa
                attr_np = cv2.resize(attr_np, (original_image.width, original_image.height), interpolation=cv2.INTER_CUBIC)
                attr_np = cv2.GaussianBlur(attr_np, (25, 25), 0)
                
                # Opcjonalnie: odcięcie najsłabszych szumów tła
                attr_np[attr_np < 0.2] = 0
                    
                heatmap = cv2.applyColorMap(np.uint8(255 * attr_np), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                orig_np = np.array(original_image)
                # Używamy maski alfa dla ładniejszego wtapiania zamiast tępego addWeighted na całym kadrze
                alpha = attr_np[..., np.newaxis] * 0.65 
                overlay = (orig_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
                
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            print(f"⚠️ Błąd generowania Grad-CAM: {str(e)}")
            return None

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        img_tensor = self.base_transform(image).unsqueeze(0).to(self.device)
        
        results = {}
        with torch.no_grad():
            if 'noise' in self.models:
                noise_tensor = self.noise_transform(image).unsqueeze(0).to(self.device)
                logits = self.models['noise'](noise_tensor)
                results['noise_prob'] = torch.sigmoid(logits).item()
                
                noise_xai = self._generate_noise_xai(self.models['noise'], noise_tensor, image)
                if noise_xai: results['noise_vis'] = noise_xai

            if 'fft' in self.models:
                try:
                    # KROK 1: Przygotowanie tensora (jeśli potrzebne)
                    import torchvision.transforms as transforms
                    fft_input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
                    
                    # KROK 2: Standardowe, czyste wywołanie PyTorch bez starych keywordów
                    with torch.no_grad():
                        # Jeśli Twój nowy model sam liczy FFT w środku, podajemy mu po prostu obraz:
                        logits = self.models['fft'](fft_input_tensor)
                        
                        # (Uwaga: Jeśli model oczekuje gotowego widma FFT policzonego tutaj, 
                        # zamień fft_input_tensor na odpowiednio wyliczony fft_tensor)
                    
                    # KROK 3: Wyciągnięcie wyniku
                    if logits.numel() == 1:
                        fake_prob = torch.sigmoid(logits).item()
                    else:
                        fake_prob = torch.softmax(logits, dim=1)[0][1].item()
                        
                    results['fft_prob'] = fake_prob
                    
                    # Generowanie XAI dla FFT 
                    fft_xai = self._generate_fft_vis(image)
                    if fft_xai: results['fft_vis'] = fft_xai

                except Exception as e:
                    import traceback
                    print(f"🛑 Błąd FFT: \n{traceback.format_exc()}")
                
            if 'clip' in self.models:
                self.clip_labels = [
                    "a natural, realistic photo of a common object",
                    "a surreal, logically impossible AI generated image",
                    "a photo with distorted textures and structural errors",
                    "a composite image of two unrelated objects merged together"
                ]

                # 1. Przygotowanie wejścia dla CLIP
                clip_inputs = self.clip_processor(text=self.clip_labels, images=image, return_tensors="pt", padding=True).to(self.device)

                # Pytamy model, do którego zdania obraz pasuje bardziej
                outputs = self.models['clip'](**clip_inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                # 2. Wybieramy najlepszą etykietę (Top-1)
                top_idx = probs.argmax().item()
                detected_label = self.clip_labels[top_idx]
                max_prob = probs[0][top_idx].item()

                # 3. PRZEKAZUJEMY TO DO WYNIKÓW
                results['clip_prob'] = max_prob
                results['clip_label'] = detected_label

                # Generujemy nowe, lepsze XAI
                clip_xai = self._generate_clip_xai(self.models['clip'], self.clip_processor, image)
                if clip_xai: results['clip_vis'] = clip_xai
                
            if 'rgb' in self.models:
                rgb_tensor = self.rgb_transform(image).unsqueeze(0).to(self.device)
                
                logits = self.models['rgb'](rgb_tensor)
                results['rgb_prob'] = torch.sigmoid(logits).item()
                
                # Generowanie XAI
                gradcam_xai = self._generate_gradcam(self.models['rgb'], rgb_tensor, image)
                if gradcam_xai:
                    results['rgb_gradcam'] = gradcam_xai

            if 'pca' in self.models:
                try:
                    import torchvision.transforms as transforms
                    import numpy as np
                    import cv2

                    # --- ZAAWANSOWANY PREPROCESSING (Jak na treningu) ---
                    pca_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    
                    # Aplikujemy pełne transformacje i wysyłamy na GPU
                    pca_tensor = pca_transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.models['pca'](pca_tensor, return_heatmap=True)
                        
                        if isinstance(output, tuple):
                            logits, heatmap = output
                        else:
                            logits = output
                            heatmap = None

                        if logits.numel() == 1:
                            fake_prob = torch.sigmoid(logits).item()
                        else:
                            fake_prob = torch.softmax(logits, dim=1)[0][1].item()
                    
                    results['pca_prob'] = fake_prob

                    # --- WIZUALIZACJA DLA UI ---
                    if heatmap is not None:
                        pca_vis = self._generate_pca_heatmap(heatmap, image)
                        if pca_vis:
                            results['pca_vis'] = pca_vis

                except Exception as e:
                    import traceback
                    print(f"🛑 Błąd PCA SOTA (Inference): \n{traceback.format_exc()}")
                
        return results