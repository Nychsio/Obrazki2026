import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from src.models.fft_detector.transforms import ComplexFourierTransform

def get_dataloaders(batch_size=32, train_size=4000, val_size=1000):
    print(f"Ładowanie zbioru OpenFake (Train: {train_size}, Val: {val_size})...")

    # Ładujemy główny dataset w trybie streaming
    base_dataset = load_dataset("ComplexDataLab/OpenFake", split='train', streaming=True)

    # PODZIAŁ (Validation Split 80/20):
    # Bierzemy pierwsze N próbek do treningu
    train_dataset = base_dataset.take(train_size)
    # Pomijamy pierwsze N próbek i bierzemy kolejne M do walidacji (aby się nie mieszały!)
    val_dataset = base_dataset.skip(train_size).take(val_size)

    # Transformacje dla zbioru TRENINGOWEGO (Z Augmentacją)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        ComplexFourierTransform()
    ])

    # Transformacje dla zbioru WALIDACYJNEGO (Czyste, bez zmian)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        ComplexFourierTransform()
    ])

    # Dwie osobne funkcje mapujące, żeby nie pomieszać transformacji
    def apply_train_transforms(examples):
        processed_images = []
        for img in examples['image']:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_images.append(train_transforms(img))
        examples['image'] = processed_images

        if isinstance(examples['label'][0], str):
            mapping = {"fake": 1, "real": 0, "generated": 1, "authentic": 0}
            examples['label'] = [mapping.get(str(l).lower(), 1) for l in examples['label']]
        else:
            examples['label'] = [int(l) for l in examples['label']]
        return examples

    def apply_val_transforms(examples):
        processed_images = []
        for img in examples['image']:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_images.append(val_transforms(img))
        examples['image'] = processed_images

        if isinstance(examples['label'][0], str):
            mapping = {"fake": 1, "real": 0, "generated": 1, "authentic": 0}
            examples['label'] = [mapping.get(str(l).lower(), 1) for l in examples['label']]
        else:
            examples['label'] = [int(l) for l in examples['label']]
        return examples

    # Aplikujemy odpowiednie funkcje do odpowiednich zbiorów
    train_dataset = train_dataset.map(apply_train_transforms, batched=True, remove_columns=['id']).with_format("torch")
    val_dataset = val_dataset.map(apply_val_transforms, batched=True, remove_columns=['id']).with_format("torch")

    # Zwracamy DWA dataloadery
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    #test
    return train_loader, val_loader