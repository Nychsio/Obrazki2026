import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # Do paska postępu w terminalu

# Importujemy nasze moduły
from src.data.data_loader import get_streaming_dataloader
from src.models.fft_detector.model import FFTResNetDetector

def train():
    # 1. Ustawienia urządzenia (Device Agnostic Code - duży plus dla rekruterów)
    # Automatycznie wybierze kartę graficzną (CUDA), Apple Silicon (MPS) lub ostatecznie procesor (CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Używane urządzenie: {device}")

    # 2. Inicjalizacja modelu, ładowarki danych, optymalizatora i funkcji straty
    print("Inicjalizacja modelu...")
    model = FFTResNetDetector(num_classes=1).to(device)
    
    dataloader = get_streaming_dataloader(batch_size=32)
    
    criterion = nn.BCEWithLogitsLoss()
    # Learning Rate (współczynnik uczenia) - jak duże kroki robi model podczas nauki
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    # 3. Parametry treningu
    epochs = 15 # Zaczynamy od małej liczby epok (przejść przez zbiór)
    
    print("Rozpoczęcie treningu...")
    for epoch in range(epochs):
        model.train() # Ustawiamy model w tryb treningowy
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Pętla przez pakiety danych (batches)
        # Używamy tqdm dla ładnego paska postępu w konsoli
        progress_bar = tqdm(dataloader, desc=f"Epoka {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Przenosimy dane na odpowiednie urządzenie (GPU/CPU)
            inputs = batch['image'].to(device)
            # Etykiety muszą być zmiennoprzecinkowe (float) dla funkcji BCE
            labels = batch['label'].to(device).float().unsqueeze(1) 
            
            # Zerujemy gradienty z poprzedniego kroku
            optimizer.zero_grad()
            
            # Forward pass (przepuszczenie danych przez model)
            outputs = model(inputs)
            
            # Obliczenie błędu (straty)
            loss = criterion(outputs, labels)
            
            # Backward pass (Propagacja wsteczna - obliczanie poprawek dla wag)
            loss.backward()
            
            # Aktualizacja wag modelu
            optimizer.step()
            
            # --- Zbieranie metryk do wyświetlania ---
            running_loss += loss.item()
            
            # Przekształcamy wynik z powrotem na prawdopodobieństwo (Sigmoid) 
            # i sprawdzamy, czy jest > 0.5 (Fake) czy < 0.5 (Real)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Aktualizacja paska postępu na bieżąco
            current_loss = running_loss / total_samples
            current_acc = (correct_predictions / total_samples) * 100
            progress_bar.set_postfix({'Loss': f"{current_loss:.4f}", 'Acc': f"{current_acc:.2f}%"})
            
        print(f"Podsumowanie Epoki {epoch+1}: Loss: {running_loss/len(dataloader):.4f}, Accuracy: {current_acc:.2f}%")
        
        # Zapisujemy wagi modelu po każdej epoce (tzw. Checkpointing)
        torch.save(model.state_dict(), f"fft_detector_epoch_{epoch+1}.pth")
        print(f"Zapisano model do pliku fft_detector_epoch_{epoch+1}.pth\n")

if __name__ == "__main__":
    train()