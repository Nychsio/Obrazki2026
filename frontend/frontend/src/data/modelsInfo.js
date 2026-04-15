import { Brain, Waves, Image as ImageIcon, Binary, Activity } from 'lucide-react';

export const MODEL_DETAILS = {
  clip: {
    title: "Model CLIP (Analiza Semantyczna)",
    icon: Brain,
    color: "text-emerald-400",
    description: "Ten model wykorzystuje architekturę Vision Transformer (ViT). Nie patrzy na pojedyncze piksele, ale na sens i logikę obrazu. Rozumie fizykę świata.",
    xai_desc: "Ognista mapa (paleta Inferno). Czerwone i żółte obszary pokazują obiekty lub połączenia, które według sztucznej inteligencji nie mają logicznego sensu (np. arbuz wtapiający się w krowę)."
  },
  fft: {
    title: "Model FFT (Widmo Częstotliwości)",
    icon: Waves,
    color: "text-purple-400",
    description: "Analizuje obraz w dziedzinie częstotliwości za pomocą Szybkiej Transformaty Fouriera. Generatory AI (szczególnie GANy) zostawiają w tej przestrzeni niewidoczne gołym okiem ślady (tzw. grid artifacts).",
    xai_desc: "Mapa w palecie Viridis. Jasne piki i regularne prążki z dala od centrum krzyża oznaczają nienaturalne anomalie, charakterystyczne dla syntezy cyfrowej."
  },
  rgb: {
    title: "Model RGB (Analiza Przestrzenna)",
    icon: ImageIcon,
    color: "text-red-400",
    description: "Klasyczna sieć konwolucyjna (EfficientNet-B0). Przeszukuje strukturę pikseli w poszukiwaniu klasycznych błędów generatorów: rozmytych krawędzi, asymetrii i złego oświetlenia.",
    xai_desc: "Klasyczny Grad-CAM (paleta Jet). Czerwone strefy to obszary, w których struktura pikseli zdradziła algorytm (np. dziwne przejścia między tłem a obiektem)."
  },
  noise: {
    title: "Model Noise (Analiza Szumu Matrycy)",
    icon: Binary,
    color: "text-blue-400",
    description: "Każdy fizyczny aparat zostawia unikalny szum (PRNU). Algorytmy AI próbują go naśladować, ale często robią to zbyt perfekcyjnie lub wprowadzają nienaturalne wzorce.",
    xai_desc: "Mapa Saliency (paleta Ocean). Zaznacza obszary, gdzie ziarno obrazu jest podejrzanie gładkie lub sztucznie wygenerowane."
  },
  pca: {
    title: "Model PCA (Macierz Kowariancji)",
    icon: Activity,
    color: "text-amber-400",
    description: "Bada wariancję gradientów pionowych i poziomych. Sztuczne sieci neuronowe często wprowadzają nienaturalną regularność na osiach X i Y.",
    xai_desc: "Elipsa ufności kowariancji. Idealnie okrągła lub przechylona pod nienaturalnym kątem sugeruje obecność tzw. artefaktów szachownicy (checkerboard artifacts)."
  }
};