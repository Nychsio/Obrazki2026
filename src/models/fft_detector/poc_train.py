import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import importlib.machinery
import importlib.util


def load_module_from_path(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class RandomFFTDataset(Dataset):
    def __init__(self, num_samples=64, H=224, W=224):
        self.num_samples = num_samples
        self.H = H
        self.W = W

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate normalized magnitude spectrum in [0,1], shape [1,H,W]
        img = torch.rand(1, self.H, self.W, dtype=torch.float32)
        # Random binary label
        label = torch.randint(0, 2, (1,), dtype=torch.int64).item()
        return {'image': img, 'label': label}


def main():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.py")

    model_mod = load_module_from_path("fft_model_module", model_path)
    FFTResNetDetector = getattr(model_mod, "FFTResNetDetector")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = FFTResNetDetector(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Small synthetic dataset
    dataset = RandomFFTDataset(num_samples=32)
    loader = DataLoader(dataset, batch_size=8)

    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in loader:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    print(f"PoC Training finished: loss={epoch_loss:.4f}")

    # Save checkpoint
    ckpt = {
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),  # future work
        'loss': epoch_loss
    }
    out_path = os.path.join(base, 'poc_fft_checkpoint.pth')
    torch.save(ckpt, out_path)
    print(f"Saved PoC checkpoint to {out_path}")


if __name__ == '__main__':
    main()
