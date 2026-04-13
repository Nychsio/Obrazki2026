#!/usr/bin/env python3
"""
Skrypt uruchamiający sekwencyjnie trening wszystkich 5 modeli Ensemble.
Przeznaczony do pełnego treningu produkcyjnego na RTX 3090 Ti (24GB VRAM).

Użycie:
    python train_all.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_training(script_name, description):
    """Uruchamia skrypt treningowy i loguje wynik."""
    print(f"\n{'='*60}")
    print(f"🚀 Rozpoczynam trening: {description}")
    print(f"📁 Skrypt: {script_name}")
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Uruchomienie skryptu
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Trening zakończony pomyślnie!")
        print(f"⏱️  Czas trwania: {elapsed_time:.2f} sekund ({elapsed_time/60:.2f} minut)")
        
        # Wyświetlenie ostatnich linii outputu
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            print("📋 Ostatnie linie outputu:")
            for line in lines[-5:]:
                print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Błąd podczas treningu!")
        print(f"⏱️  Czas trwania: {elapsed_time:.2f} sekund")
        print(f"📋 Kod błędu: {e.returncode}")
        
        if e.stdout:
            print("📋 Output (stdout):")
            print(e.stdout[-1000:])  # Ostatnie 1000 znaków
        
        if e.stderr:
            print("📋 Błędy (stderr):")
            print(e.stderr[-1000:])  # Ostatnie 1000 znaków
        
        return False
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku: {script_name}")
        return False

def main():
    """Główna funkcja uruchamiająca wszystkie treningi sekwencyjnie."""
    print("🤖 SYSTEM ENSEMBLE - PEŁNY TRENING PRODUKCYJNY")
    print("🖥️  Przeznaczony dla RTX 3090 Ti (24GB VRAM)")
    print(f"📅 Data rozpoczęcia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Katalog roboczy: {os.getcwd()}")
    
    # Lista skryptów do uruchomienia w kolejności
    training_scripts = [
        {
            "script": "src/noise/train.py",
            "description": "Model Noise Binary Classifier (analiza szumów)"
        },
        {
            "script": "src/rgb/train.py", 
            "description": "Model RGB Classifier (EfficientNet-B0)"
        },
        {
            "script": "src/models/fft_detector/train.py",
            "description": "Model FFT ResNet Detector (analiza częstotliwości)"
        },
        {
            "script": "src/models/clip/train_clip.py",
            "description": "Model Semantic Judge CLIP (analiza semantyczna)"
        },
        {
            "script": "src/models/gradient_pca/train_pca.py",
            "description": "Model Gradient PCA Detector (analiza gradientów)"
        }
    ]
    
    # Sprawdzenie czy wszystkie pliki istnieją
    print("\n🔍 Sprawdzanie dostępności skryptów...")
    missing_files = []
    for item in training_scripts:
        if not os.path.exists(item["script"]):
            missing_files.append(item["script"])
    
    if missing_files:
        print("❌ Brakujące pliki:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n⚠️  Nie można kontynuować. Upewnij się, że wszystkie skrypty istnieją.")
        return 1
    
    print("✅ Wszystkie skrypty są dostępne.")
    
    # Uruchomienie treningów sekwencyjnie
    successful = []
    failed = []
    
    for item in training_scripts:
        success = run_training(item["script"], item["description"])
        
        if success:
            successful.append(item["description"])
        else:
            failed.append(item["description"])
        
        # Krótka przerwa między treningami
        if item != training_scripts[-1]:  # Nie czekaj po ostatnim
            print(f"\n⏳ Oczekiwanie 10 sekund przed następnym treningiem...")
            time.sleep(10)
    
    # Podsumowanie
    print(f"\n{'='*60}")
    print("📊 PODSUMOWANIE TRENINGU ENSEMBLE")
    print(f"{'='*60}")
    print(f"✅ Zakończone pomyślnie: {len(successful)}/{len(training_scripts)}")
    
    if successful:
        print("   Pomyślne treningi:")
        for item in successful:
            print(f"   - {item}")
    
    if failed:
        print(f"\n❌ Nieudane treningi: {len(failed)}/{len(training_scripts)}")
        for item in failed:
            print(f"   - {item}")
    
    # Sprawdzenie czy powstały checkpointy
    print(f"\n🔍 Sprawdzanie wygenerowanych checkpointów...")
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if checkpoints:
            print(f"✅ Znaleziono {len(checkpoints)} checkpointów:")
            for cp in sorted(checkpoints):
                cp_path = os.path.join(checkpoint_dir, cp)
                size = os.path.getsize(cp_path) / (1024*1024)  # MB
                print(f"   - {cp} ({size:.2f} MB)")
        else:
            print("⚠️  Katalog checkpoints istnieje, ale jest pusty.")
    else:
        print("❌ Katalog checkpoints nie istnieje!")
    
    print(f"\n🏁 Zakończono wszystkie treningi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())