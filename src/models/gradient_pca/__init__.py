"""
Gradient PCA module for the Obrazki Destylacja project.

This module provides gradient covariance extraction for PCA-based
feature extraction from RGB images.
"""

from .extractor import GradientCovarianceExtractor, extract_gradient_covariance

__all__ = [
    'GradientCovarianceExtractor',
    'extract_gradient_covariance',
]

__version__ = '0.1.0'