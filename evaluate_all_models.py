from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.models.clip.clip_streamer import CLIPDataStreamer
from src.models.clip.semantic_judge import SemanticJudgeCLIP
from src.models.fft_detector.model import FFTResNetDetector
from src.models.gradient_pca.model import GradientPCADetector
from src.models.noise.model import NoiseBinaryClassifier
from src.models.rgb.data import OpenFakeDataset
from src.models.rgb.train import RGBClassifier


DEFAULT_CHECKPOINTS = {
    "noise": ["best_noise_model.pt", "checkpoints/best_noise_model.pt"],
    "rgb": ["best_rgb_model.pt", "checkpoints/best_rgb_model.pt"],
    "clip": ["clip_model_best.pth", "moj_wytrenowany_clip.pth", "checkpoints/clip_model_best.pth"],
    "fft": ["fft_detector_best.pth", "checkpoints/fft_detector_best.pth"],
    "gradient_pca": ["best_gradient_pca_model.pt", "checkpoints/best_gradient_pca_model.pt"],
}


def build_eval_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def normalize_label(label: object) -> float:
    if isinstance(label, torch.Tensor):
        return float(label.item())

    if isinstance(label, str):
        lowered = label.strip().lower()
        if lowered in {"real", "authentic", "0"}:
            return 0.0
        if lowered in {"fake", "generated", "ai", "1"}:
            return 1.0
        raise ValueError(f"Unsupported string label: {label!r}")

    return float(label)


def rgb_collate_fn(batch: list[tuple[torch.Tensor, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([normalize_label(item[1]) for item in batch], dtype=torch.float32).view(-1, 1)
    return images, labels


def rgb_to_fft_two_channel(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 4:
        raise ValueError(f"Expected a 4D tensor [B, C, H, W], got {tuple(inputs.shape)}")

    channels = inputs.size(1)
    if channels == 2:
        return inputs

    if channels == 3:
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=inputs.device, dtype=inputs.dtype).view(1, 3, 1, 1)
        gray = (inputs * rgb_weights).sum(dim=1)
    elif channels == 1:
        gray = inputs.squeeze(1)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    fft = torch.fft.fft2(gray, dim=(-2, -1), norm="ortho")
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    amplitude = torch.log1p(torch.abs(fft_shifted))
    phase = torch.angle(fft_shifted)
    return torch.stack((amplitude, phase), dim=1)


def resolve_checkpoint(candidates: Iterable[str]) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "classifier_state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                try:
                    model.load_state_dict(state_dict)
                    return
                except RuntimeError:
                    continue

    if isinstance(checkpoint, dict):
        try:
            model.load_state_dict(checkpoint)
            return
        except RuntimeError:
            pass

    raise RuntimeError(f"Could not load weights from {checkpoint_path}")


@dataclass
class ModelResult:
    name: str
    checkpoint: Path
    precision: float
    recall: float
    f1: float
    samples: int


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    predict_fn: Callable[[torch.nn.Module, tuple[torch.Tensor, torch.Tensor], torch.device], torch.Tensor],
    max_samples: Optional[int] = None,
) -> tuple[float, float, float, int]:
    model.eval()

    all_targets: list[float] = []
    all_predictions: list[float] = []

    processed_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch[1].to(device=device, dtype=torch.float32).view(-1)

            if max_samples is not None:
                remaining = max_samples - processed_samples
                if remaining <= 0:
                    break
                if labels.numel() > remaining:
                    batch = (batch[0][:remaining], batch[1][:remaining])
                    labels = labels[:remaining]

            logits = predict_fn(model, batch, device).view(-1)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(dtype=torch.float32)

            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            processed_samples += labels.numel()
            if max_samples is not None and processed_samples >= max_samples:
                break

    if not all_targets:
        return 0.0, 0.0, 0.0, 0

    y_true = [int(value) for value in all_targets]
    y_pred = [int(value) for value in all_predictions]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1, len(y_true)


def build_rgb_loader(split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = OpenFakeDataset(split=split, transform=build_eval_transform())
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=rgb_collate_fn)


def build_clip_loader(split: str, batch_size: int) -> DataLoader:
    streamer = CLIPDataStreamer(batch_size=batch_size)
    return streamer.create_dataloader(split=split)


def predict_direct(model: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> torch.Tensor:
    images = batch[0].to(device)
    return model(images)


def predict_fft(model: FFTResNetDetector, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> torch.Tensor:
    images = batch[0].to(device)
    fft_inputs = rgb_to_fft_two_channel(images)
    return model(images, fft_inputs)


def predict_clip(model: SemanticJudgeCLIP, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> torch.Tensor:
    pixel_values = batch[0].to(device)
    return model(pixel_values)


def run_evaluation(args: argparse.Namespace) -> list[ModelResult]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    results: list[ModelResult] = []

    model_specs = {
        "noise": (NoiseBinaryClassifier, predict_direct, lambda: build_rgb_loader(args.split, args.batch_size, args.num_workers)),
        "rgb": (RGBClassifier, predict_direct, lambda: build_rgb_loader(args.split, args.batch_size, args.num_workers)),
        "clip": (SemanticJudgeCLIP, predict_clip, lambda: build_clip_loader(args.split, args.batch_size)),
        "fft": (FFTResNetDetector, predict_fft, lambda: build_rgb_loader(args.split, args.batch_size, args.num_workers)),
        "gradient_pca": (
            lambda: GradientPCADetector(device=device),
            predict_direct,
            lambda: build_rgb_loader(args.split, args.batch_size, args.num_workers),
        ),
    }

    for name, (model_factory, predict_fn, dataloader_factory) in model_specs.items():
        if args.model not in {"all", name}:
            continue

        checkpoint_candidates = [args.override_checkpoint] if args.override_checkpoint and args.model == name else DEFAULT_CHECKPOINTS[name]
        checkpoint = resolve_checkpoint(checkpoint_candidates)

        if checkpoint is None:
            print(f"Skipping {name}: checkpoint not found.")
            continue

        print(f"Evaluating {name} from {checkpoint} ...")
        model = model_factory()
        if hasattr(model, "to"):
            model = model.to(device)

        load_state_dict(model, checkpoint, device)
        dataloader = dataloader_factory()

        precision, recall, f1, samples = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            predict_fn=predict_fn,
            max_samples=args.max_samples,
        )

        results.append(
            ModelResult(
                name=name,
                checkpoint=checkpoint,
                precision=precision,
                recall=recall,
                f1=f1,
                samples=samples,
            )
        )

    return results


def print_results(results: list[ModelResult]) -> None:
    if not results:
        print("No models were evaluated.")
        return

    header = f"{'Model':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Samples':>10}"
    print("\n" + header)
    print("-" * len(header))
    for result in results:
        print(f"{result.name:<16} {result.precision:>10.4f} {result.recall:>10.4f} {result.f1:>10.4f} {result.samples:>10}")


def export_results(results: list[ModelResult], json_path: Optional[str], csv_path: Optional[str]) -> None:
    if json_path:
        payload = [
            {
                "model": result.name,
                "checkpoint": str(result.checkpoint),
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "samples": result.samples,
            }
            for result in results
        ]
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved JSON metrics to {json_path}")

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "checkpoint", "precision", "recall", "f1", "samples"],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "model": result.name,
                        "checkpoint": str(result.checkpoint),
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1": result.f1,
                        "samples": result.samples,
                    }
                )
        print(f"Saved CSV metrics to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all deepfake detection models on the OpenFake test split.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers for RGB-based models.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of samples per model.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, or cuda.")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "noise", "rgb", "clip", "fft", "gradient_pca"],
        help="Evaluate a single model instead of all models.",
    )
    parser.add_argument(
        "--override-checkpoint",
        default=None,
        help="Optional checkpoint path to use when --model is set to a single model.",
    )
    parser.add_argument("--json-output", default=None, help="Optional path to save metrics as JSON.")
    parser.add_argument("--csv-output", default=None, help="Optional path to save metrics as CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_evaluation(args)
    print_results(results)
    export_results(results, args.json_output, args.csv_output)


if __name__ == "__main__":
    main()