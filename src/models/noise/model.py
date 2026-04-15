import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur


def high_pass_filter(images: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
	"""Compute a simple noise residual via high-pass filtering.

	Args:
		images: Input batch tensor with shape (N, C, H, W).
		kernel_size: Gaussian blur kernel size.
		sigma: Standard deviation for Gaussian blur.

	Returns:
		Residual tensor with the same shape as input.
	"""
	blurred = gaussian_blur(images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
	residual = images - blurred
	return residual


class NoiseBinaryClassifier(nn.Module):
	"""CNN-based real/deepfake classifier operating on noise residuals."""

	def __init__(self, feature_dim: int = 128) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.projection = nn.Linear(256, feature_dim)
		self.classifier = nn.Linear(feature_dim, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = high_pass_filter(x)
		features = self.features(residual)
		features = torch.flatten(features, 1)
		features = self.projection(features)
		logits = self.classifier(features)
		return logits
