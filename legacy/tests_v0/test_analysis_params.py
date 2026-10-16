#!/usr/bin/env python3
"""Test Analysis parameter setting"""

from frame_overlap import Data, Reconstruct, Analysis

def test_parameter_setting():
    """Test that parameters can be properly set before fitting"""

    print("\n" + "="*70)
    print("TEST: Analysis parameter setting")
    print("="*70)

    # Load and process data
    data = Data(
        signal_file='notebooks/iron_powder.csv',
        openbeam_file='notebooks/openbeam.csv',
        flux=5e6,
        duration=0.5,
        freq=20
    )

    # Apply processing
    data.convolute_response(pulse_duration=20.0)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8.0)
    data.overlap(kernel=1, total_time=50, mode='equal')

    # Reconstruct
    recon = Reconstruct(data, tmin=None, tmax=None)
    recon.filter(kind='wiener', noise_power=1.0)

    print("✓ Data loaded and reconstructed")

    # Create Analysis object
    analysis = Analysis(
        xs='iron',
        vary_background=True,
        vary_response=False,
        vary_weights=True,
        vary_sans=None,
        vary_extinction=None
    )

    print("\n" + "="*70)
    print("Method 1: Using set_params() - RECOMMENDED")
    print("="*70)

    # Set parameters using new set_params() method
    analysis.set_params(
        thickness={'value': 1.95, 'vary': False},
        norm={'value': 1.0, 'vary': True},
        temp={'vary': False}
    )

    print("\nParameters set using set_params():")
    print(f"  thickness: value={analysis.model.params['thickness'].value}, vary={analysis.model.params['thickness'].vary}")
    print(f"  norm: value={analysis.model.params['norm'].value}, vary={analysis.model.params['norm'].vary}")
    print(f"  temp: vary={analysis.model.params['temp'].vary}")

    # Fit - explicitly pass params to ensure they are used
    result = analysis.fit(recon, params=analysis.model.params, wlmin=1.0, wlmax=4)

    print(f"\nnbragg Fit Results:")
    print(f"  Reduced χ²: {result.redchi:.4f}")
    print(f"\nFitted Parameters:")
    for param_name, param in result.params.items():
        vary_str = "✓" if param.vary else "✗"
        if param.vary and param.stderr:
            print(f"  [{vary_str}] {param_name}: {param.value:.6f} ± {param.stderr:.6f}")
        else:
            print(f"  [{vary_str}] {param_name}: {param.value:.6f}")

    # Check that thickness stayed at 1.95
    thickness_value = result.params['thickness'].value
    print(f"\nThickness check:")
    if abs(thickness_value - 1.95) < 0.01:
        print(f"  ✅ Thickness fixed at 1.95 (actual: {thickness_value:.6f})")
    else:
        print(f"  ❌ Thickness not fixed (expected: 1.95, actual: {thickness_value:.6f})")

    print("\n" + "="*70)
    print("Method 2: Using model.params.set() - OLD METHOD")
    print("="*70)

    # Create new analysis for comparison
    analysis2 = Analysis(
        xs='iron',
        vary_background=True,
        vary_response=False,
        vary_weights=True,
        vary_sans=None,
        vary_extinction=None
    )

    # Try old method
    analysis2.model.params.set(norm={"vary": True})
    analysis2.model.params.set(temp={"vary": False})
    analysis2.model.params.set(thickness={"vary": False, "value": 1.95})

    print("\nParameters set using model.params.set():")
    print(f"  thickness: value={analysis2.model.params['thickness'].value}, vary={analysis2.model.params['thickness'].vary}")
    print(f"  norm: value={analysis2.model.params['norm'].value}, vary={analysis2.model.params['norm'].vary}")

    # Fit - also pass params explicitly
    result2 = analysis2.fit(recon, params=analysis2.model.params, wlmin=1.0, wlmax=4)

    print(f"\nnbragg Fit Results:")
    print(f"  Reduced χ²: {result2.redchi:.4f}")
    print(f"\nFitted Parameters:")
    for param_name, param in result2.params.items():
        vary_str = "✓" if param.vary else "✗"
        if param.vary and param.stderr:
            print(f"  [{vary_str}] {param_name}: {param.value:.6f} ± {param.stderr:.6f}")
        else:
            print(f"  [{vary_str}] {param_name}: {param.value:.6f}")

    # Check thickness
    thickness_value2 = result2.params['thickness'].value
    print(f"\nThickness check:")
    if abs(thickness_value2 - 1.95) < 0.01:
        print(f"  ✅ Thickness fixed at 1.95 (actual: {thickness_value2:.6f})")
    else:
        print(f"  ⚠️  Thickness may have changed (expected: 1.95, actual: {thickness_value2:.6f})")

    print("\n" + "="*70)
    print("✅ Test complete")
    print("="*70)

if __name__ == "__main__":
    test_parameter_setting()
