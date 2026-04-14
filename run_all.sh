#!/bin/bash

# Ustawienie zmiennych środowiskowych dla Linuxa
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export HF_TOKEN=${HF_TOKEN}

echo "========================================"
echo "🚀 1/5: Start - Model Noise"
echo "========================================"
python src/noise/train.py

echo "========================================"
echo "🚀 2/5: Start - Model RGB"
echo "========================================"
python src/rgb/train.py

echo "========================================"
echo "🚀 3/5: Start - Model Gradient PCA"
echo "========================================"
python src/models/gradient_pca/train_pca.py

echo "========================================"
echo "🚀 4/5: Start - Model CLIP"
echo "========================================"
python src/models/clip/train_clip.py

echo "========================================"
echo "🚀 5/5: Start - Model FFT"
echo "========================================"
python src/models/fft_detector/train.py

echo "========================================"
echo "✅ WSZYSTKIE MODELE ZAKOŃCZYŁY TRENING!"
echo "========================================"