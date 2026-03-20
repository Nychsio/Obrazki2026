import torch
from datasets import load_dataset
from transformers import CLIPProcessor
from torch.utils.data import DataLoader

class CLIPDataStreamer:
    def __init__(self, dataset_name="ComplexDataLab/OpenFake", model_name="openai/clip-vit-base-patch32", batch_size=32):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        # CLIPProcessor automatycznie robi resize (224x224) i normalizację (mean, std)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def get_stream(self, split="train"):
        """
        Ładuje zbiór danych w trybie streaming (nie pobiera całości na dysk!).
        """
        print(f"Inicjalizacja strumienia dla zbioru: {self.dataset_name} (split: {split})")
        dataset = load_dataset(self.dataset_name, split=split, streaming=True)
        
        # Dodaj shuffle tylko dla danych treningowych
        if split == "train":
            dataset = dataset.shuffle(seed=42, buffer_size=5000)
            print("✓ Dodano shuffle dla danych treningowych (buffer_size=5000)")
            
        return dataset

    def collate_fn(self, batch):
        """
        Funkcja pakująca pojedyncze próbki w gotowy batch tensorów.
        """
        images = [item["image"].convert("RGB") for item in batch]
        
        # Bezpieczna konwersja etykiet (obsługa tekstów i liczb)
        labels = []
        for item in batch:
            val = item["label"]
            if isinstance(val, str):
                # Konwertujemy string na int (1 dla fake, 0 dla real)
                labels.append(1.0 if "fake" in val.lower() or "ai" in val.lower() else 0.0)
            else:
                labels.append(float(val))

        # Processor przygotowuje piksele pod CLIP-a
        inputs = self.processor(images=images, return_tensors="pt")
        
        # Formatujemy etykiety pod Binary Cross Entropy (wymiar: [batch_size, 1])
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        return inputs.pixel_values, labels_tensor

    def create_dataloader(self, split="train"):
        """
        Zwraca gotowy DataLoader, po którym można iterować w pętli trenującej.
        """
        dataset = self.get_stream(split)
        # Przy streamingu używamy standardowego DataLoader bez funkcji shuffle (tasowania całego zbioru)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn
        )
        return dataloader

# Szybki test lokalny (wykona się tylko, gdy bezpośrednio odpalisz ten plik)
if __name__ == "__main__":
    streamer = CLIPDataStreamer(batch_size=4)
    loader = streamer.create_dataloader(split="train")
    
    print("Oczekiwanie na pobranie pierwszego batcha...")
    for batch_idx, (pixel_values, labels) in enumerate(loader):
        print(f"Batch {batch_idx+1} | Kształt pikseli: {pixel_values.shape} | Kształt etykiet: {labels.shape}")
        break # Testujemy tylko jeden batch