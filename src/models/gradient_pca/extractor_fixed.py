"""
Gradient Covariance Extractor for PCA-based feature extraction.

This module implements gradient covariance extraction from RGB images
using Sobel filters for gradient computation and covariance matrix
calculation for PCA dimensionality reduction.

Author: Obrazki Destylacja Project
Date: 2026-04-13
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class GradientCovarianceExtractor:
    """
    Extracts gradient covariance features from RGB images.
    
    The extractor performs the following steps:
    1. Converts RGB to luminance using ITU-R BT.709 weights
    2. Computes horizontal (G_x) and vertical (G_y) gradients using Sobel filters
    3. Calculates 2x2 covariance matrix for each image in the batch
    4. Returns flattened covariance matrices
    
    Attributes:
        device (torch.device): Device for tensor operations
        sobel_x (torch.Tensor): Sobel filter for horizontal gradients
        sobel_y (torch.Tensor): Sobel filter for vertical gradients
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the gradient covariance extractor.
        
        Args:
            device: Device for tensor operations. If None, uses CUDA if available.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Define Sobel filters for gradient computation
        # Shape: [1, 1, 3, 3] for batch and channel dimensions
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Luminance conversion weights (ITU-R BT.709)
        self.luminance_weights = torch.tensor(
            [0.2126, 0.7152, 0.0722], 
            dtype=torch.float32, 
            device=self.device
        ).view(1, 3, 1, 1)
    
    def rgb_to_luminance(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to luminance using ITU-R BT.709 weights.
        
        Args:
            rgb_tensor: Input RGB tensor of shape [B, 3, H, W]
            
        Returns:
            Luminance tensor of shape [B, 1, H, W]
        """
        # Ensure weights are on the same device as input
        weights = self.luminance_weights.to(rgb_tensor.device)
        
        # Weighted sum across color channels
        luminance = torch.sum(rgb_tensor * weights, dim=1, keepdim=True)
        
        return luminance
    
    def compute_gradients(self, luminance_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute horizontal and vertical gradients using Sobel filters.
        
        Args:
            luminance_tensor: Luminance tensor of shape [B, 1, H, W]
            
        Returns:
            Tuple of (G_x, G_y) gradient tensors, each of shape [B, 1, H, W]
        """
        batch_size = luminance_tensor.shape[0]
        
        # Ensure Sobel filters are on the same device as input
        sobel_x = self.sobel_x.to(luminance_tensor.device)
        sobel_y = self.sobel_y.to(luminance_tensor.device)
        
        # Compute gradients for each image in batch
        G_x_list = []
        G_y_list = []
        
        for i in range(batch_size):
            # Extract single image [1, 1, H, W]
            img = luminance_tensor[i:i+1]
            
            # Compute gradients for single image
            G_x_single = F.conv2d(img, sobel_x, padding=1)
            G_y_single = F.conv2d(img, sobel_y, padding=1)
            
            G_x_list.append(G_x_single)
            G_y_list.append(G_y_single)
        
        # Concatenate results
        G_x = torch.cat(G_x_list, dim=0)
        G_y = torch.cat(G_y_list, dim=0)
        
        return G_x, G_y
    
    def compute_covariance_matrix(self, G_x: torch.Tensor, G_y: torch.Tensor) -> torch.Tensor:
        """
        Compute 2x2 covariance matrix for each image in the batch.
        
        The covariance matrix is defined as:
            [[var(G_x), cov(G_x, G_y)],
             [cov(G_x, G_y), var(G_y)]]
        
        Args:
            G_x: Horizontal gradient tensor of shape [B, 1, H, W]
            G_y: Vertical gradient tensor of shape [B, 1, H, W]
            
        Returns:
            Flattened covariance matrices of shape [B, 4]
        """
        batch_size = G_x.shape[0]
        
        # Flatten spatial dimensions
        G_x_flat = G_x.view(batch_size, -1)  # [B, H*W]
        G_y_flat = G_y.view(batch_size, -1)  # [B, H*W]
        
        # Compute means
        mean_G_x = torch.mean(G_x_flat, dim=1, keepdim=True)  # [B, 1]
        mean_G_y = torch.mean(G_y_flat, dim=1, keepdim=True)  # [B, 1]
        
        # Center the gradients
        G_x_centered = G_x_flat - mean_G_x  # [B, H*W]
        G_y_centered = G_y_flat - mean_G_y  # [B, H*W]
        
        # Compute covariance matrix elements
        # Using unbiased estimator (N-1 in denominator)
        n_pixels = G_x_flat.shape[1]
        
        # var(G_x) = E[(G_x - μ_x)²]
        var_G_x = torch.sum(G_x_centered * G_x_centered, dim=1) / (n_pixels - 1)
        
        # var(G_y) = E[(G_y - μ_y)²]
        var_G_y = torch.sum(G_y_centered * G_y_centered, dim=1) / (n_pixels - 1)
        
        # cov(G_x, G_y) = E[(G_x - μ_x)(G_y - μ_y)]
        cov_Gx_Gy = torch.sum(G_x_centered * G_y_centered, dim=1) / (n_pixels - 1)
        
        # Construct covariance matrices and flatten
        # Each matrix is [[var_G_x, cov_Gx_Gy], [cov_Gx_Gy, var_G_y]]
        covariance_matrices = torch.stack([
            var_G_x, cov_Gx_Gy, cov_Gx_Gy, var_G_y
        ], dim=1)  # [B, 4]
        
        return covariance_matrices
    
    def __call__(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract gradient covariance features from RGB tensor.
        
        Args:
            rgb_tensor: Input RGB tensor of shape [B, 3, H, W]
            
        Returns:
            Flattened covariance matrices of shape [B, 4]
            
        Raises:
            ValueError: If input tensor doesn't have expected shape
        """
        # Validate input shape
        if rgb_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, 3, H, W], got {rgb_tensor.dim()}D")
        
        if rgb_tensor.shape[1] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {rgb_tensor.shape[1]} channels")
        
        # Convert to luminance
        luminance = self.rgb_to_luminance(rgb_tensor)
        
        # Compute gradients
        G_x, G_y = self.compute_gradients(luminance)
        
        # Compute covariance matrices
        covariance_matrices = self.compute_covariance_matrix(G_x, G_y)
        
        return covariance_matrices
    
    def extract_with_intermediates(self, rgb_tensor: torch.Tensor) -> dict:
        """
        Extract gradient covariance features with intermediate results.
        
        Args:
            rgb_tensor: Input RGB tensor of shape [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - 'luminance': Luminance tensor [B, 1, H, W]
                - 'G_x': Horizontal gradients [B, 1, H, W]
                - 'G_y': Vertical gradients [B, 1, H, W]
                - 'covariance': Flattened covariance matrices [B, 4]
        """
        # Convert to luminance
        luminance = self.rgb_to_luminance(rgb_tensor)
        
        # Compute gradients
        G_x, G_y = self.compute_gradients(luminance)
        
        # Compute covariance matrices
        covariance = self.compute_covariance_matrix(G_x, G_y)
        
        return {
            'luminance': luminance,
            'G_x': G_x,
            'G_y': G_y,
            'covariance': covariance
        }


# Convenience function for quick extraction
def extract_gradient_covariance(rgb_tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convenience function for extracting gradient covariance features.
    
    Args:
        rgb_tensor: Input RGB tensor of shape [B, 3, H, W]
        device: Device for computation
        
    Returns:
        Flattened covariance matrices of shape [B, 4]
    """
    extractor = GradientCovarianceExtractor(device)
    return extractor(rgb_tensor)