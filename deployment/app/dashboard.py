import streamlit as st
import requests
from PIL import Image
import io
import base64

# Konfiguracja strony
st.set_page_config(
    page_title="Detektor Deepfake AI",
    page_icon="🛡️",
    layout="wide"
)

# Adres naszego lokalnego API (FastAPI)
API_URL = "http://127.0.0.1:8000/api/v1/analyze"

st.title("🛡️ Detektor Obrazów AI - Dashboard Naukowy")
st.markdown("""
Projekt *Obrazki 2026*. System fuzji wyników (Ensemble) z modeli: Noise, RGB, FFT, CLIP oraz Gradient PCA.
""")

# Interfejs wczytywania pliku
uploaded_file = st.file_uploader("Wgraj obraz do analizy", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Wyświetlanie wgranego obrazka
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Analizowany obraz", width="stretch")
        
        analyze_button = st.button("🔍 Uruchom Analizę Ensemble", width="stretch")

    # Inicjalizacja pamięci podręcznej (Session State)
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    with col2:
        if analyze_button:
            with st.spinner("Odpytywanie modeli bazowych przez API..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        data = response.json()
                        # ZAPISUJEMY WYNIKI DO PAMIĘCI
                        st.session_state.predictions = data.get("predictions", {})
                        st.success("Analiza zakończona!")
                    else:
                        st.error(f"Błąd API: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Brak połączenia z API.")

        # WYŚWIETLANIE WYNIKÓW (działa nawet przy przełączaniu opcji!)
        if st.session_state.predictions:
            predictions = st.session_state.predictions
            
            st.subheader("📊 Wyniki poszczególnych modeli")
            # Zmieniamy na 4 kolumny
            m1, m2, m3, m4 = st.columns(4)
            
            if "clip_prob" in predictions: m1.metric("Model CLIP", f"{predictions['clip_prob'] * 100:.2f}% Fake")
            if "fft_prob" in predictions: m2.metric("Model FFT", f"{predictions['fft_prob'] * 100:.2f}% Fake")
            if "rgb_prob" in predictions: m3.metric("Model RGB", f"{predictions['rgb_prob'] * 100:.2f}% Fake")
            if "noise_prob" in predictions: m4.metric("Model Noise", f"{predictions['noise_prob'] * 100:.2f}% Fake")
            
            if "pca_features" in predictions:
                # ---> TUTAJ PRZYWRACAMY PCA <---
                st.write("**Cechy kowariancji gradientów (PCA):**")
                features = predictions["pca_features"][0]
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Wariancja G_x", f"{features[0]:.4f}")
                pc2.metric("Kowariancja XY", f"{features[1]:.4f}")
                pc3.metric("Kowariancja YX", f"{features[2]:.4f}")
                pc4.metric("Wariancja G_y", f"{features[3]:.4f}")
                
                import matplotlib.pyplot as plt
                from matplotlib.patches import Ellipse
                import numpy as np
                
                # Pobranie wartości
                var_gx, cov_xy, cov_yx, var_gy = features
                
                # Tworzenie wykresu
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.axhline(0, color='grey', lw=0.5, ls='--')
                ax.axvline(0, color='grey', lw=0.5, ls='--')
                
                # Obliczenia do elipsy (Wartości własne i wektory własne macierzy kowariancji)
                cov_matrix = np.array([[var_gx, cov_xy], [cov_yx, var_gy]])
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Kąt nachylenia elipsy
                angle = np.degrees(np.arctan2(*eigenvectors[:, 1][::-1]))
                
                # Rysowanie elipsy (szerokość i wysokość oparte na pierwiastku z eigenvalues)
                ell = Ellipse(xy=(0, 0), 
                              width=np.sqrt(eigenvalues[1]) * 4, 
                              height=np.sqrt(eigenvalues[0]) * 4, 
                              angle=angle, facecolor='cyan', alpha=0.3, edgecolor='blue', lw=2)
                ax.add_patch(ell)
                
                # Ustawienia osi
                max_val = np.sqrt(max(var_gx, var_gy)) * 3
                ax.set_xlim(-max_val, max_val)
                ax.set_ylim(-max_val, max_val)
                ax.set_title("Elipsa kowariancji gradientów (PCA)", fontsize=10)
                ax.set_xlabel("Gradient X")
                ax.set_ylabel("Gradient Y")
                
                st.pyplot(fig)
                # ---------------------------------
            
            st.divider()
            
            # Interaktywna sekcja XAI
            st.subheader("🧠 Interaktywne XAI - Zobacz, co zdradziło ten obraz")
            xai_option = st.radio(
                "Wybierz perspektywę eksperta:",
                ["🔴 Model RGB (Przestrzeń)", "🟢 Model CLIP (Semantyka)", "🟣 Model FFT (Widmo)", "🔵 Model Noise (Szum)"],
                horizontal=True
            )
            
            xai_col1, xai_col2 = st.columns([1, 1])
            with xai_col1:
                if "RGB" in xai_option and "rgb_gradcam" in predictions:
                    img_bytes = base64.b64decode(predictions["rgb_gradcam"])
                    st.image(Image.open(io.BytesIO(img_bytes)), caption="Analiza przestrzenna (RGB)", width="stretch")
                elif "CLIP" in xai_option and "clip_vis" in predictions:
                    img_bytes = base64.b64decode(predictions["clip_vis"])
                    st.image(Image.open(io.BytesIO(img_bytes)), caption="Analiza semantyczna (CLIP)", width="stretch")
                elif "FFT" in xai_option and "fft_vis" in predictions:
                    img_bytes = base64.b64decode(predictions["fft_vis"])
                    st.image(Image.open(io.BytesIO(img_bytes)), caption="Analiza częstotliwościowa (FFT)", width="stretch")
                elif "Noise" in xai_option and "noise_vis" in predictions:
                    img_bytes = base64.b64decode(predictions["noise_vis"])
                    st.image(Image.open(io.BytesIO(img_bytes)), caption="Analiza anomalii szumu", width="stretch")