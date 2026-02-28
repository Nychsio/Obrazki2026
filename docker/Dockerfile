# Używamy oficjalnego obrazu PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja zależności systemowych dla OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie i instalacja bibliotek Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie kodu projektu
COPY . .

CMD ["python", "src/main.py"]