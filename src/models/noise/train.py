import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.noise.model import NoiseBinaryClassifier
from src.models.rgb.data import OpenFakeDataset, get_transforms


def _labels_to_tensor(labels, device: torch.device) -> torch.Tensor:
	"""Convert dataset labels to float tensor in shape (N, 1)."""
	if isinstance(labels, torch.Tensor):
		labels_tensor = labels
	elif isinstance(labels, np.ndarray):
		labels_tensor = torch.from_numpy(labels)
	elif isinstance(labels, (list, tuple)):
		if len(labels) > 0 and isinstance(labels[0], str):
			labels_tensor = torch.tensor([0 if str(v).lower() == "real" else 1 for v in labels])
		else:
			labels_tensor = torch.as_tensor(labels)
	else:
		labels_tensor = torch.as_tensor(labels)

	return labels_tensor.to(device=device, dtype=torch.float32).view(-1, 1)


def safe_dataloader(dataloader):
	iterator = iter(dataloader)
	while True:
		try:
			yield next(iterator)
		except StopIteration:
			break
		except Exception as e:
			print(f"Błąd ładowania paczki danych, pomijanie: {e}")
			continue


def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	optimizer: optim.Optimizer,
	scaler: GradScaler,
	device: torch.device,
	steps_per_epoch: int,
	use_amp: bool,
) -> float:
	model.train()
	running_loss = 0.0
	step_count = 0

	pbar = tqdm(total=steps_per_epoch, desc="Training", leave=False)
	for images, labels in safe_dataloader(dataloader):
		if step_count >= steps_per_epoch:
			break

		images = images.to(device)
		labels = _labels_to_tensor(labels, device)

		optimizer.zero_grad(set_to_none=True)
		with autocast(enabled=use_amp):
			logits = model(images)
			loss = criterion(logits, labels)

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		running_loss += loss.item()
		step_count += 1
		pbar.update(1)

	pbar.close()
	if step_count == 0:
		return 0.0
	return running_loss / step_count


@torch.no_grad()
def validate(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	steps_per_val: int,
	use_amp: bool,
) -> tuple[float, float, float]:
	model.eval()
	running_loss = 0.0
	step_count = 0
	all_probs = []
	all_targets = []

	pbar = tqdm(total=steps_per_val, desc="Validation", leave=False)
	for images, labels in safe_dataloader(dataloader):
		if step_count >= steps_per_val:
			break

		images = images.to(device)
		targets = _labels_to_tensor(labels, device)

		with autocast(enabled=use_amp):
			logits = model(images)
			loss = criterion(logits, targets)

		probs = torch.sigmoid(logits)
		all_probs.append(probs.detach().cpu())
		all_targets.append(targets.detach().cpu())

		running_loss += loss.item()
		step_count += 1
		pbar.update(1)

	pbar.close()
	if step_count == 0:
		return 0.0, 0.0, 0.0

	avg_val_loss = running_loss / step_count
	y_score = torch.cat(all_probs, dim=0).numpy().reshape(-1)
	y_true = torch.cat(all_targets, dim=0).numpy().reshape(-1)

	if np.unique(y_true).size > 1:
		val_roc_auc = roc_auc_score(y_true, y_score)
	else:
		val_roc_auc = 0.5

	y_pred = (y_score >= 0.5).astype(np.int32)
	val_f1 = f1_score(y_true.astype(np.int32), y_pred, zero_division=0)

	return avg_val_loss, val_f1, val_roc_auc


def main() -> None:
	batch_size = 64  # Zwiększone dla RTX 3090 Ti (24GB VRAM)
	num_workers = 8  # Increased for better data loading performance
	epochs = 12  # Zwiększone dla pełnego treningu
	learning_rate = 1e-4
	steps_per_epoch = 1000  # Zwiększone dla pełnego treningu
	steps_per_val = 200  # Zwiększone dla pełnego treningu
	checkpoint_path = "checkpoints/best_noise_model.pt"

	# Utworzenie katalogu checkpoints jeśli nie istnieje
	import os
	os.makedirs("checkpoints", exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	use_amp = device.type == "cuda"
	print(f"Using device: {device}")

	writer = SummaryWriter("runs/noise_experiment_1")

	transforms = get_transforms()
	train_dataset = OpenFakeDataset(split="train", transform=transforms)
	val_dataset = OpenFakeDataset(split="test", transform=transforms)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

	model = NoiseBinaryClassifier().to(device)
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	scaler = GradScaler(enabled=use_amp)

	best_val_roc_auc = float("-inf")

	for epoch in range(1, epochs + 1):
		train_loss = train_one_epoch(
			model=model,
			dataloader=train_loader,
			criterion=criterion,
			optimizer=optimizer,
			scaler=scaler,
			device=device,
			steps_per_epoch=steps_per_epoch,
			use_amp=use_amp,
		)

		val_loss, val_f1, val_roc_auc = validate(
			model=model,
			dataloader=val_loader,
			criterion=criterion,
			device=device,
			steps_per_val=steps_per_val,
			use_amp=use_amp,
		)

		writer.add_scalar("Training Loss", train_loss, epoch)
		writer.add_scalar("Validation F1-score", val_f1, epoch)
		writer.add_scalar("Validation ROC-AUC", val_roc_auc, epoch)

		print(
			f"Epoch {epoch}/{epochs} | "
			f"Train Loss: {train_loss:.4f} | "
			f"Val Loss: {val_loss:.4f} | "
			f"Val F1: {val_f1:.4f} | "
			f"Val ROC-AUC: {val_roc_auc:.4f}"
		)

		# Save strictly on highest validation ROC-AUC.
		if val_roc_auc > best_val_roc_auc:
			best_val_roc_auc = val_roc_auc
			torch.save(model.state_dict(), checkpoint_path)
			print(f"Saved new best model to {checkpoint_path} (ROC-AUC: {best_val_roc_auc:.4f})")

	writer.close()
	print(f"Training finished. Best validation ROC-AUC: {best_val_roc_auc:.4f}")


if __name__ == "__main__":
	main()
