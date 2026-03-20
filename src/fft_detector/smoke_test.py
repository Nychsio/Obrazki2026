import os
import sys
import importlib.machinery
import importlib.util
from types import ModuleType
import numpy as np
from PIL import Image
import torch


def load_module_from_path(name: str, path: str) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.py")
    transforms_path = os.path.join(base, "transforms.py")

    # Dynamiczne wczytanie modułów bez polegania na package importach
    model_mod = load_module_from_path("fft_model_module", model_path)
    transforms_mod = load_module_from_path("fft_transforms_module", transforms_path)

    FFTResNetDetector = getattr(model_mod, "FFTResNetDetector")
    FourierMagnitudeTransform = getattr(transforms_mod, "FourierMagnitudeTransform")

    # Parametry testu
    batch_size = 4
    img_size = 224

    # Generujemy losowe obrazy RGB jako PIL Images
    images = []
    for _ in range(batch_size):
        arr = np.random.randint(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))

    transform = FourierMagnitudeTransform()

    # Zastosuj transform na każdej próbce i utwórz batch tensor [B,1,H,W]
    transformed = [transform(img) for img in images]
    # Każdy element powinien mieć kształt [1, H, W]
    batch_tensor = torch.stack(transformed, dim=0)

    # Sprawdź kształt: [B, 1, H, W]
    assert batch_tensor.shape[0] == batch_size, f"Batch size mismatch: {batch_tensor.shape}"
    assert batch_tensor.dim() == 4 and batch_tensor.shape[1] == 1, f"Unexpected tensor shape: {batch_tensor.shape}"

    # Utwórz model i wykonaj forward
    model = FFTResNetDetector(num_classes=1)
    model.eval()

    with torch.no_grad():
        outputs = model(batch_tensor)

    # Oczekujemy kształtu [B, 1]
    assert outputs.shape[0] == batch_size, f"Output batch mismatch: {outputs.shape}"
    # Jeżeli second dim exists, dopilnuj aby była 1
    assert outputs.dim() == 2 and outputs.shape[1] == 1, f"Unexpected model output shape: {outputs.shape}"

    # Brak NaN/Inf w wyjściu
    assert not torch.isnan(outputs).any(), "Model output contains NaN"
    assert not torch.isinf(outputs).any(), "Model output contains Inf"

    print("SMOKE TEST PASSED: forward ok, shapes as expected, no NaN/Inf in outputs")


if __name__ == "__main__":
    main()
