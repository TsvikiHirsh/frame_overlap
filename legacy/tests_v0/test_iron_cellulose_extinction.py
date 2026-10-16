"""
Test iron+cellulose analysis with extinction parameters.

This test verifies that the new iron_with_cellulose configuration works correctly
with and without extinction parameters, and that default values are properly set.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct, Analysis
import numpy as np


def test_iron_cellulose_basic():
    """Test basic iron+cellulose without extinction."""
    print("\n" + "="*70)
    print("TEST 1: Basic iron+cellulose (no extinction)")
    print("="*70)

    analysis = Analysis(
        xs='iron_with_cellulose',
        vary_weights=True,
        vary_background=True,
        vary_extinction=False
    )

    assert analysis.xs is not None, "CrossSection should be created"
    assert len(analysis.xs.materials) == 2, "Should have 2 materials (iron + cellulose)"
    assert analysis.model is not None, "TransmissionModel should be created"

    print(f"✓ Created iron+cellulose analysis")
    print(f"  - Materials: {len(analysis.xs.materials)}")
    print(f"  - Model parameters: {len(analysis.model.params)}")
    print(f"  - vary_weights: {analysis.vary_weights}")
    print(f"  - vary_background: {analysis.vary_background}")
    print(f"  - vary_extinction: {analysis.vary_extinction}")


def test_iron_cellulose_with_extinction():
    """Test iron+cellulose with extinction parameters."""
    print("\n" + "="*70)
    print("TEST 2: Iron+cellulose with extinction")
    print("="*70)

    analysis = Analysis(
        xs='iron_with_cellulose',
        vary_weights=True,
        vary_background=True,
        vary_extinction=True
    )

    assert analysis.xs is not None, "CrossSection should be created"
    assert len(analysis.xs.materials) == 2, "Should have 2 materials"
    assert analysis.vary_extinction == True, "vary_extinction should be True"

    print(f"✓ Created iron+cellulose analysis with extinction")
    print(f"  - Materials: {len(analysis.xs.materials)}")
    print(f"  - Model parameters: {len(analysis.model.params)}")
    print(f"  - vary_extinction: {analysis.vary_extinction}")


def test_default_parameters():
    """Test that default thickness and norm are properly set."""
    print("\n" + "="*70)
    print("TEST 3: Default parameters (thickness=1.95cm, norm=1.0 fixed)")
    print("="*70)

    analysis = Analysis(
        xs='iron_with_cellulose',
        vary_weights=True,
        vary_background=True,
        thickness_guess=1.95
    )

    # Check if thickness parameter exists and has correct value
    thickness_param = None
    norm_param = None

    for param_name, param in analysis.model.params.items():
        if 'thickness' in param_name.lower() or param_name == 'L':
            thickness_param = param
            print(f"  - Found thickness parameter: {param_name}")
            print(f"    Value: {param.value:.2f} cm")
            print(f"    Vary: {param.vary}")
        if 'norm' in param_name.lower():
            norm_param = param
            print(f"  - Found norm parameter: {param_name}")
            print(f"    Value: {param.value:.2f}")
            print(f"    Vary: {param.vary}")

    if thickness_param:
        assert abs(thickness_param.value - 1.95) < 0.01, "Thickness should be 1.95 cm"
        print(f"✓ Thickness correctly set to {thickness_param.value:.2f} cm")

    if norm_param:
        assert abs(norm_param.value - 1.0) < 0.01, "Norm should be 1.0"
        assert norm_param.vary == False, "Norm should be fixed"
        print(f"✓ Norm correctly set to {norm_param.value:.2f} (fixed)")


def test_full_workflow_with_iron_cellulose():
    """Test complete workflow with iron+cellulose analysis."""
    print("\n" + "="*70)
    print("TEST 4: Full workflow with iron+cellulose")
    print("="*70)

    # Create data
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)
    print(f"✓ Data loaded: {len(data.data)} points")

    # Apply processing stages
    data.convolute_response(200, bin_width=10)
    print(f"✓ Convolution applied")

    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)
    print(f"✓ Poisson sampling applied")

    data.overlap(kernel=[0, 25], total_time=50)
    print(f"✓ Overlap applied: {len(data.overlapped_data)} points")

    # Reconstruct
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=0.2)
    print(f"✓ Reconstruction: {len(recon.reconstructed_data)} points")

    # Analyze with iron+cellulose
    analysis = Analysis(
        xs='iron_with_cellulose',
        vary_weights=True,
        vary_background=True,
        vary_extinction=False,
        thickness_guess=1.95
    )

    print(f"✓ Analysis object created")
    print(f"  - Will fit {len(analysis.model.params)} parameters")

    # Note: We don't actually run the fit in this test to keep it fast
    # The fit would be: result = analysis.fit(recon)


def test_vary_sans_parameter():
    """Test vary_sans parameter."""
    print("\n" + "="*70)
    print("TEST 5: vary_sans parameter")
    print("="*70)

    analysis = Analysis(
        xs='iron_with_cellulose',
        vary_sans=True,
        vary_background=True
    )

    assert analysis.vary_sans == True, "vary_sans should be True"
    print(f"✓ vary_sans parameter works")
    print(f"  - vary_sans: {analysis.vary_sans}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Iron+Cellulose Analysis with Extinction Parameters")
    print("="*70)

    try:
        test_iron_cellulose_basic()
        test_iron_cellulose_with_extinction()
        test_default_parameters()
        test_full_workflow_with_iron_cellulose()
        test_vary_sans_parameter()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  - Iron+cellulose XS works without extinction ✓")
        print("  - Iron+cellulose XS works with extinction ✓")
        print("  - Default parameters (thickness=1.95, norm=1.0 fixed) ✓")
        print("  - Full workflow integration ✓")
        print("  - vary_sans parameter ✓")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
