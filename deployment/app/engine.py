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
from src.models.fft_detector.model import FFTResNetDetector
from src.models.gradient_pca.model import GradientCovarianceExtractor
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
        model_path = self.models_dir / "fft_detector_best.pth"
        if model_path.exists():
            print("⏳ Ładowanie modelu FFT...")
            model = FFTResNetDetector(num_classes=1)
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
        model_path = self.models_dir / "best_gradient_pca_model.pt"
        if model_path.exists():
            print("⏳ Ładowanie modelu PCA...")
            extractor = GradientCovarianceExtractor(device=self.device)
            self.models['pca'] = extractor
            print("✅ Model PCA załadowany i gotowy!")
        else:
            print(f"❌ Brak pliku: {model_path}")

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
        """Generuje wizualizację widma FFT jako Base64"""
        try:
            # Konwersja obrazu do skali szarości i obliczenie 2D FFT
            img_gray = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2GRAY)
            f = np.fft.fft2(img_gray)
            fshift = np.fft.fftshift(f)
            
            # Obliczenie magnitudy (zabezpieczenie przed log(0))
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Normalizacja do formatu obrazu 0-255
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Kolorowanie widma na inny kolor niż RGB (np. fioletowo-żółty VIRIDIS)
            heatmap = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_VIRIDIS)
            
            # Kodowanie do Base64
            _, buffer = cv2.imencode('.jpg', heatmap)
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

    def _generate_pca_vis(self, features):
        """Generuje elegancki wykres elipsy wariancji dla PCA."""
        try:
            var_gx, cov_xy, cov_yx, var_gy = features
            fig, ax = plt.subplots(figsize=(5, 5))

            # Stylizacja pod mroczny motyw Reacta
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            ax.axhline(0, color='#334155', lw=1, ls='--')
            ax.axvline(0, color='#334155', lw=1, ls='--')

            cov_matrix = np.array([[var_gx, cov_xy], [cov_yx, var_gy]])
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))

            # Rysowanie elipsy
            ell = Ellipse(xy=(0, 0), width=np.sqrt(eigenvalues[1]) * 4, height=np.sqrt(eigenvalues[0]) * 4,
                          angle=angle, facecolor='#38bdf8', alpha=0.4, edgecolor='#0284c7', lw=2)
            ax.add_patch(ell)

            max_val = np.sqrt(max(var_gx, var_gy)) * 3
            if max_val == 0:
                max_val = 1
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
            ax.tick_params(colors='#94a3b8')

            buf = io.BytesIO()
            plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Błąd generowania wykresu PCA: {str(e)}")
            return None

    def _generate_gradcam(self, model, input_tensor, original_image):
        """Generuje mapę ciepła Grad-CAM i zwraca ją jako string Base64"""
        try:
            with torch.enable_grad(): # Wymuszamy śledzenie gradientów tylko na moment!
                input_tensor = input_tensor.clone().requires_grad_()
                # W modelu EfficientNet z timm ostatnia warstwa konwolucyjna to conv_head
                target_layer = model.backbone.conv_head
                
                # Inicjalizacja Captum LayerGradCam
                layer_gc = LayerGradCam(model, target_layer)
                
                # Obliczenie atrybucji (target=0, bo mamy klasyfikację binarną na 1 neuronie)
                attributions = layer_gc.attribute(input_tensor, target=0)
                
                # Interpolacja do rozmiaru 224x224
                attributions = LayerGradCam.interpolate(attributions, (224, 224), interpolate_mode='bilinear')
                
                # Konwersja do numpy i nałożenie ReLU (tylko pozytywne wpływy)
                attr_np = attributions.squeeze().cpu().detach().numpy()
                attr_np = np.maximum(attr_np, 0)
                
                # Normalizacja do [0, 1]
                if np.max(attr_np) > 0:
                    attr_np = attr_np / np.max(attr_np)
                    
                # Generowanie kolorowej mapy ciepła (OpenCV JET)
                heatmap = cv2.applyColorMap(np.uint8(255 * attr_np), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Przygotowanie oryginalnego obrazu jako tła
                orig_resized = original_image.resize((224, 224))
                orig_np = np.array(orig_resized)
                
                # Nałożenie mapy ciepła na obraz (50% przezroczystości)
                overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
                
                # Kodowanie obrazu do Base64, aby wysłać go przez API
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
                # Transformacja zwraca tensor 2-kanałowy [2, H, W]
                fft_tensor = self.fft_transform(image).unsqueeze(0).to(self.device)
                
                try:
                    # Próba wywołania wariantu dwustrumieniowego: forward(self, x, x_fft)
                    logits = self.models['fft'](img_tensor, x_fft=fft_tensor)
                except TypeError:
                    # Awaryjne wywołanie, gdyby model wymagał tylko argumentu x_fft
                    logits = self.models['fft'](x_fft=fft_tensor)
                    
                results['fft_prob'] = torch.sigmoid(logits).item()
                
                # Dodane XAI FFT
                fft_xai = self._generate_fft_vis(image)
                if fft_xai: results['fft_vis'] = fft_xai
                
            if 'clip' in self.models:
                # Lepszy Prompt Engineering dla CLIPa
                labels = [
                    "a real, natural, authentic photograph without any edits", 
                    "an impossible, surreal, AI-generated image with logical mistakes and strange objects"
                ]

                # Przetwarzamy obraz ORAZ tekst
                inputs = self.clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(self.device)

                # Pytamy model, do którego zdania obraz pasuje bardziej
                outputs = self.models['clip'](**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

                # Pobieramy prawdopodobieństwo dla klasy [1] czyli "AI generated"
                fake_probability = probs[0][1].item()
                results['clip_prob'] = fake_probability

                # Generujemy nowe, lepsze XAI
                clip_xai = self._generate_clip_xai(self.models['clip'], self.clip_processor, image)
                if clip_xai: results['clip_vis'] = clip_xai
                
            if 'rgb' in self.models:
                rgb_tensor = self.rgb_transform(image).unsqueeze(0).to(self.device)
                
                logits = self.models['rgb'](rgb_tensor)
                results['rgb_prob'] = torch.sigmoid(logits).item()
                
                # Generowanie XAI
                gradcam_b64 = self._generate_gradcam(self.models['rgb'], rgb_tensor, image)
                if gradcam_b64:
                    results['rgb_gradcam'] = gradcam_b64

            # PRZYWRÓCONY BLOK PCA:
            if 'pca' in self.models:
                import torchvision.transforms as transforms
                pca_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
                pca_features = self.models['pca'](pca_tensor).cpu().numpy().tolist()[0]

                results['pca_features'] = pca_features
                results['pca_prob'] = 0.5 # PCA to ekstraktor, nie klasyfikator, dajemy neutralne 50%

                pca_vis = self._generate_pca_vis(pca_features)
                if pca_vis: results['pca_vis'] = pca_vis
                
        return results