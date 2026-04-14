import torch
import random
import io
import PIL.Image
from datasets import load_dataset, Image
from transformers import CLIPProcessor
from torch.utils.data import DataLoader, IterableDataset
from PIL import UnidentifiedImageError


class BufferShuffledIterableDataset(IterableDataset):
    """
    Prosty shuffle buforowy dla strumieni (IterableDataset).
    """

    def __init__(self, iterable_dataset, buffer_size=10000, seed=42):
        super().__init__()
        self.iterable_dataset = iterable_dataset
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        stream_iter = iter(self.iterable_dataset)

        buffer = []
        for _ in range(self.buffer_size):
            try:
                buffer.append(next(stream_iter))
            except StopIteration:
                break

        if not buffer:
            return

        while True:
            try:
                new_item = next(stream_iter)
            except StopIteration:
                break

            random_idx = rng.randint(0, len(buffer) - 1)
            yield buffer[random_idx]
            buffer[random_idx] = new_item

        rng.shuffle(buffer)
        for item in buffer:
            yield item


class CorruptedImageSafeIterableDataset(IterableDataset):
    """
    Odporny wrapper na uszkodzone rekordy obrazów.
    Jeśli próbka jest uszkodzona, jest pomijana zamiast przerywać cały trening.
    """

    def __init__(self, iterable_dataset):
        super().__init__()
        self.iterable_dataset = iterable_dataset

    def __iter__(self):
        for item in self.iterable_dataset:
            try:
                raw_bytes = item["image"]["bytes"]
                if raw_bytes is None and "path" in item["image"]:
                    with open(item["image"]["path"], "rb") as f:
                        raw_bytes = f.read()

                img = PIL.Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                safe_item = dict(item)
                safe_item["image"] = img
                yield safe_item
            except (UnidentifiedImageError, OSError) as e:
                print(f"Skipping corrupted image: {e}")
                continue
            except Exception as e:
                print(f"Skipping corrupted image: {e}")
                continue

class CLIPDataStreamer:
    def __init__(
        self,
        dataset_name="ComplexDataLab/OpenFake",
        model_name="openai/clip-vit-base-patch32",
        batch_size=32,
        shuffle_buffer_size=10000,
        shuffle_seed=42,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        # CLIPProcessor automatycznie robi resize (224x224) i normalizację (mean, std)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _normalize_split(self, split):
        if split == "validation":
            print("⚠️ Split 'validation' nie istnieje w OpenFake, używam splitu 'test'.")
            return "test"
        return split
        
    def get_stream(self, split="train"):
        """
        Ładuje zbiór danych w trybie streaming (nie pobiera całości na dysk!).
        """
        split = self._normalize_split(split)
        print(f"Inicjalizacja strumienia dla zbioru: {self.dataset_name} (split: {split})")
        dataset = load_dataset(self.dataset_name, split=split, streaming=True)
        
        # Wyłączenie automatycznego dekodowania obrazów – zabezpieczenie przed wywalającymi się danymi
        dataset = dataset.cast_column('image', Image(decode=False))
        
        # Dodaj shuffle buforowy tylko dla danych treningowych
        if split == "train":
            dataset = BufferShuffledIterableDataset(
                dataset,
                buffer_size=self.shuffle_buffer_size,
                seed=self.shuffle_seed,
            )
            print(
                "✓ Dodano buffer shuffle dla danych treningowych "
                f"(buffer_size={self.shuffle_buffer_size})"
            )

        dataset = CorruptedImageSafeIterableDataset(dataset)
        print("✓ Włączono pomijanie uszkodzonych obrazów w strumieniu")
            
        return dataset

    def collate_fn(self, batch):
        """
        Funkcja pakująca pojedyncze próbki w gotowy batch tensorów.
        """
        images = [item["image"] for item in batch]
        
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
            collate_fn=self.collate_fn,
            shuffle=False,
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