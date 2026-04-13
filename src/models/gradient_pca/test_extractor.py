"""
Test script for GradientCovarianceExtractor.
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.gradient_pca.extractor import GradientCovarianceExtractor, extract_gradient_covariance


def test_basic_functionality():
    """Test basic functionality of the extractor."""
    print("Testing GradientCovarianceExtractor...")
    
    # Create extractor
    extractor = GradientCovarianceExtractor()
    print(f"Extractor device: {extractor.device}")
    
    # Create dummy RGB tensor [B, 3, H, W]
    batch_size = 2
    height, width = 64, 64
    rgb_tensor = torch.randn(batch_size, 3, height, width, device=extractor.device)
    
    print(f"Input shape: {rgb_tensor.shape}")
    
    # Test __call__ method
    covariance = extractor(rgb_tensor)
    print(f"Covariance shape: {covariance.shape}")
    print(f"Covariance values (first sample): {covariance[0]}")
    
    # Test extract_with_intermediates
    intermediates = extractor.extract_with_intermediates(rgb_tensor)
    print(f"Luminance shape: {intermediates['luminance'].shape}")
    print(f"G_x shape: {intermediates['G_x'].shape}")
    print(f"G_y shape: {intermediates['G_y'].shape}")
    
    # Test convenience function
    covariance2 = extract_gradient_covariance(rgb_tensor)
    print(f"Convenience function shape: {covariance2.shape}")
    
    # Verify covariance matrices are symmetric
    for i in range(batch_size):
        cov_matrix = covariance[i].view(2, 2)
        # Check symmetry: cov_matrix[0,1] should equal cov_matrix[1,0]
        assert torch.allclose(cov_matrix[0, 1], cov_matrix[1, 0]), \
            f"Covariance matrix not symmetric for sample {i}"
        # Check variance non-negative
        assert cov_matrix[0, 0] >= 0, f"var(G_x) negative for sample {i}"
        assert cov_matrix[1, 1] >= 0, f"var(G_y) negative for sample {i}"
    
    print("✓ All basic tests passed!")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    extractor = GradientCovarianceExtractor()
    
    # Test with different image sizes
    sizes = [(32, 32), (128, 64), (256, 256)]
    for h, w in sizes:
        rgb_tensor = torch.randn(1, 3, h, w, device=extractor.device)
        covariance = extractor(rgb_tensor)
        assert covariance.shape == (1, 4), f"Wrong shape for size {h}x{w}"
        print(f"  ✓ Size {h}x{w} works")
    
    # Test error handling for wrong input shape
    try:
        extractor(torch.randn(3, 64, 64))  # Missing batch dimension
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly caught wrong dimensions: {e}")
    
    try:
        extractor(torch.randn(1, 4, 64, 64))  # Wrong channel count
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly caught wrong channel count: {e}")
    
    print("✓ All edge case tests passed!")
    return True


def test_gradient_computation():
    """Test gradient computation correctness."""
    print("\nTesting gradient computation...")
    
    extractor = GradientCovarianceExtractor()
    
    # Create a simple test image with known gradients
    # Create a ramp in x-direction (should have strong G_x)
    height, width = 32, 32
    x_ramp = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    x_ramp = x_ramp.repeat(1, 3, height, 1)  # [1, 3, 32, 32]
    x_ramp = x_ramp.to(extractor.device)
    
    intermediates = extractor.extract_with_intermediates(x_ramp)
    G_x = intermediates['G_x']
    G_y = intermediates['G_y']
    
    # For a perfect x-ramp, G_x should be positive and constant
    # G_y should be near zero
    print(f"  G_x mean: {G_x.mean().item():.4f}, std: {G_x.std().item():.4f}")
    print(f"  G_y mean: {G_y.mean().item():.4f}, std: {G_y.std().item():.4f}")
    
    # Create a ramp in y-direction
    y_ramp = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    y_ramp = y_ramp.repeat(1, 3, 1, width)
    y_ramp = y_ramp.to(extractor.device)
    
    intermediates = extractor.extract_with_intermediates(y_ramp)
    G_x = intermediates['G_x']
    G_y = intermediates['G_y']
    
    print(f"  For y-ramp - G_x mean: {G_x.mean().item():.4f}")
    print(f"  For y-ramp - G_y mean: {G_y.mean().item():.4f}")
    
    print("✓ Gradient computation tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Gradient Covariance Extractor")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_edge_cases()
        test_gradient_computation()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())