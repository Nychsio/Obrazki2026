import torch
import torch.nn as nn
import torch.optim as optim
from src.data.data_loader import get_streaming_dataloader
from src.models.fft_detector.model import FFTResNetDetector

def train_poc():
    # Wybór urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    # Inicjalizacja modelu
    model = FFTResNetDetector(num_classes=1).to(device)
    
    # Ładowanie danych (nasze 1000 próbek)
    dataloader = get_streaming_dataloader(batch_size=16)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Start treningu PoC (1000 obrazów)...")
    model.train()
    
    # Robimy tylko 2-3 epoki, żeby szybko wygenerować checkpoint
    for epoch in range(3):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            inputs = batch['image'].to(device)
            
            # Wymuszamy konwersję na tensor, nawet jeśli HF wysłało listę
            if isinstance(batch['label'], list):
                labels = torch.tensor(batch['label']).to(device).float().unsqueeze(1)
            else:
                labels = batch['label'].to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            if i % 10 == 0:
                print(f"Epoka {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    # Zapisujemy wagi w folderze modelu
    checkpoint_path = "src/models/fft_detector/poc_fft_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss': running_loss,
    }, checkpoint_path)
    
    print(f"\nTrening zakończony! Checkpoint zapisany w: {checkpoint_path}")

if __name__ == "__main__":
    train_poc()