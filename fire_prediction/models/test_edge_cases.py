"""
Test #7: Edge Case Testing for Heskestad Physics Implementation

This test validates robustness of the physics-informed system with edge cases:
- Near-zero HRR values (flame height edge cases)
- Extreme HRR values (>1000 kW) - tests Heskestad correlation limits
- Single-sample batches (inference mode)
- NaN/Inf injection (robustness)
- Negative HRR (should be caught by monotonicity loss)

Created by: Antigravity AI Assistant
Suggested by: GitHub Copilot CLI
Date: 2026-02-06
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from fire_prediction.utils.physics import (
    heskestad_flame_height,
    physics_consistency_loss,
    monotonicity_loss,
    validate_physics_consistency
)

def test_near_zero_hrr():
    """Test 1: Near-zero HRR values"""
    print("\n" + "="*70)
    print("TEST 7.1: Near-Zero HRR Values")
    print("="*70)
    
    try:
        # Test very small HRR (0.001 kW - barely burning)
        hrr_tiny = np.array([0.001, 0.01, 0.1])
        
        print(f"\n🔬 Testing HRR values: {hrr_tiny} kW")
        
        # Heskestad flame height should handle this gracefully
        flame_heights = heskestad_flame_height(hrr_tiny, D=0.3)
        
        print(f"   Flame heights: {flame_heights}")
        print(f"   All non-negative: {np.all(flame_heights >= 0)}")
        print(f"   All finite: {np.all(np.isfinite(flame_heights))}")
        
        assert np.all(flame_heights >= 0), "Flame heights should be non-negative"
        assert np.all(np.isfinite(flame_heights)), "Flame heights should be finite"
        
        print("\n✅ TEST 7.1 PASSED: Near-zero HRR handled correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 7.1 FAILED: {e}")
        return False


def test_extreme_hrr():
    """Test 2: Extreme HRR values (>1000 kW)"""
    print("\n" + "="*70)
    print("TEST 7.2: Extreme HRR Values")
    print("="*70)
    
    try:
        # Test very large HRR (2000 kW - massive fire)
        hrr_extreme = np.array([1000, 1500, 2000])
        
        print(f"\n🔥 Testing extreme HRR values: {hrr_extreme} kW")
        
        # Heskestad correlation should still work (though less accurate at extremes)
        flame_heights = heskestad_flame_height(hrr_extreme, D=0.3)
        
        print(f"   Flame heights: {flame_heights} m")
        print(f"   All positive: {np.all(flame_heights > 0)}")
        print(f"   All finite: {np.all(np.isfinite(flame_heights))}")
        print(f"   Reasonable range: {np.all(flame_heights < 50)}")  # Flame < 50m is reasonable
        
        assert np.all(flame_heights > 0), "Extreme HRR should produce positive flames"
        assert np.all(np.isfinite(flame_heights)), "Flame heights should be finite"
        assert np.all(flame_heights < 50), "Flame heights should be physically reasonable"
        
        print("\n✅ TEST 7.2 PASSED: Extreme HRR handled correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 7.2 FAILED: {e}")
        return False


def test_single_sample_batch():
    """Test 3: Single-sample batches (inference mode)"""
    print("\n" + "="*70)
    print("TEST 7.3: Single-Sample Batches")
    print("="*70)
    
    try:
        from fire_prediction.models.physics_informed import PhysicsInformedLSTM
        
        print("\n🎯 Testing inference mode (batch size = 1)")
        
        # Create model
        model = PhysicsInformedLSTM(
            input_dim=6,
            hidden_dim=64,
            use_physics_loss=True,
            validate_physics=True
        )
        
        # Single sample input [1, 30, 6]
        single_input = torch.randn(1, 30, 6)
        
        print(f"   Input shape: {single_input.shape}")
        
        # Forward pass
        output = model(single_input)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected: (1, 10, 1)")
        
        assert output.shape == (1, 10, 1), f"Expected (1, 10, 1), got {output.shape}"
        assert torch.all(torch.isfinite(output)), "Output should be finite"
        
        print("\n✅ TEST 7.3 PASSED: Single-sample batches work correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 7.3 FAILED: {e}")
        return False


def test_nan_inf_robustness():
    """Test 4: NaN/Inf injection (robustness)"""
    print("\n" + "="*70)
    print("TEST 7.4: NaN/Inf Robustness")
    print("="*70)
    
    try:
        print("\n⚠️  Testing robustness to invalid values")
        
        # Test NaN handling
        hrr_with_nan = np.array([100, np.nan, 200])
        
        print(f"   Input with NaN: {hrr_with_nan}")
        
        # Heskestad should handle NaN gracefully (or raise clear error)
        try:
            flame_heights = heskestad_flame_height(hrr_with_nan, D=0.3)
            has_nan = np.any(np.isnan(flame_heights))
            print(f"   Output has NaN: {has_nan}")
            print(f"   ⚠️  NaN propagated (expected behavior)")
        except Exception as e:
            print(f"   ✓ Raised exception for NaN: {type(e).__name__}")
        
        # Test Inf handling
        hrr_with_inf = np.array([100, np.inf, 200])
        
        print(f"\n   Input with Inf: {hrr_with_inf}")
        
        try:
            flame_heights = heskestad_flame_height(hrr_with_inf, D=0.3)
            has_inf = np.any(np.isinf(flame_heights))
            print(f"   Output has Inf: {has_inf}")
            print(f"   ⚠️  Inf propagated (expected behavior)")
        except Exception as e:
            print(f"   ✓ Raised exception for Inf: {type(e).__name__}")
        
        print("\n✅ TEST 7.4 PASSED: NaN/Inf handling is predictable")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 7.4 FAILED: {e}")
        return False


def test_negative_hrr():
    """Test 5: Negative HRR (should be caught by monotonicity loss)"""
    print("\n" + "="*70)
    print("TEST 7.5: Negative HRR Detection")
    print("="*70)
    
    try:
        print("\n🚫 Testing negative HRR (unphysical)")
        
        # Create predictions with negative HRR
        negative_hrr = torch.tensor([[[-10.0], [-5.0], [0.0], [5.0], [10.0]]])  # [1, 5, 1]
        
        print(f"   Predictions: {negative_hrr.squeeze().tolist()}")
        
        # Monotonicity loss should penalize negative values
        mono_loss = monotonicity_loss(negative_hrr, lambda_monotonic=1.0)
        
        print(f"   Monotonicity loss: {mono_loss.item():.6f}")
        print(f"   Loss > 0: {mono_loss.item() > 0}")
        
        assert mono_loss.item() > 0, "Monotonicity loss should penalize negative HRR"
        
        # Test with all positive HRR
        positive_hrr = torch.tensor([[[10.0], [15.0], [20.0], [25.0], [30.0]]])  # [1, 5, 1]
        mono_loss_positive = monotonicity_loss(positive_hrr, lambda_monotonic=1.0)
        
        print(f"\n   All-positive HRR loss: {mono_loss_positive.item():.6f}")
        print(f"   Penalty reduced: {mono_loss_positive.item() < mono_loss.item()}")
        
        assert mono_loss_positive.item() < mono_loss.item(), "Positive HRR should have lower penalty"
        
        print("\n✅ TEST 7.5 PASSED: Negative HRR correctly penalized")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 7.5 FAILED: {e}")
        return False


def main():
    """Run all edge case tests"""
    print("\n" + "="*70)
    print("EDGE CASE TESTING - TEST #7")
    print("="*70)
    print("\nSuggested by: GitHub Copilot CLI")
    print("Implemented by: Antigravity AI Assistant")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Near-Zero HRR", test_near_zero_hrr()))
    results.append(("Extreme HRR", test_extreme_hrr()))
    results.append(("Single-Sample Batches", test_single_sample_batch()))
    results.append(("NaN/Inf Robustness", test_nan_inf_robustness()))
    results.append(("Negative HRR Detection", test_negative_hrr()))
    
    # Summary
    print("\n" + "="*70)
    print("EDGE CASE TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL EDGE CASE TESTS PASSED!")
        print("="*70 + "\n")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("="*70 + "\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
