"""
Simple test for GradientCovarianceExtractor.
"""

import torch

# Direct import since we're in the same directory
try:
    from extractor import GradientCovarianceExtractor, extract_gradient_covariance
    print("✓ Successfully imported extractor module")
except ImportError:
    # Try alternative import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from extractor import GradientCovarianceExtractor, extract_gradient_covariance
    print("✓ Successfully imported extractor module (with path adjustment)")

def main():
    print("Testing GradientCovarianceExtractor...")
    
    # Create extractor
    extractor = GradientCovarianceExtractor()
    print(f"Device: {extractor.device}")
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    rgb_tensor = torch.randn(2, 3, 64, 64, device=extractor.device)
    covariance = extractor(rgb_tensor)
    print(f"   Input shape: {rgb_tensor.shape}")
    print(f"   Output shape: {covariance.shape}")
    print(f"   Output[0]: {covariance[0]}")
    
    # Check shape
    assert covariance.shape == (2, 4), f"Expected shape (2, 4), got {covariance.shape}"
    
    # Test 2: Covariance matrix symmetry
    print("\n2. Testing covariance matrix symmetry...")
    for i in range(2):
        cov_matrix = covariance[i].view(2, 2)
        assert torch.allclose(cov_matrix[0, 1], cov_matrix[1, 0]), \
            f"Covariance matrix not symmetric: {cov_matrix}"
        print(f"   Sample {i}: symmetric ✓")
    
    # Test 3: Different sizes
    print("\n3. Testing different image sizes...")
    sizes = [(32, 32), (128, 64), (256, 256)]
    for h, w in sizes:
        rgb_tensor = torch.randn(1, 3, h, w, device=extractor.device)
        covariance = extractor(rgb_tensor)
        assert covariance.shape == (1, 4), f"Wrong shape for {h}x{w}: {covariance.shape}"
        print(f"   Size {h}x{w}: ✓")
    
    # Test 4: Convenience function
    print("\n4. Testing convenience function...")
    rgb_tensor = torch.randn(3, 3, 128, 128, device=extractor.device)
    covariance = extract_gradient_covariance(rgb_tensor)
    assert covariance.shape == (3, 4), f"Wrong shape: {covariance.shape}"
    print(f"   Convenience function: ✓")
    
    # Test 5: Intermediate results
    print("\n5. Testing intermediate results...")
    rgb_tensor = torch.randn(1, 3, 64, 64, device=extractor.device)
    intermediates = extractor.extract_with_intermediates(rgb_tensor)
    assert 'luminance' in intermediates
    assert 'G_x' in intermediates
    assert 'G_y' in intermediates
    assert 'covariance' in intermediates
    print(f"   Luminance shape: {intermediates['luminance'].shape}")
    print(f"   G_x shape: {intermediates['G_x'].shape}")
    print(f"   G_y shape: {intermediates['G_y'].shape}")
    print(f"   Covariance shape: {intermediates['covariance'].shape}")
    
    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)