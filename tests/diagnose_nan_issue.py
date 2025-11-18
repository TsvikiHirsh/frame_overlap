"""
Diagnose NaN issues in nbragg fitting after reconstruction.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct
import numpy as np
import pandas as pd


def diagnose_data_quality(data, stage_name=""):
    """Check data quality and report issues."""
    print(f"\n{stage_name} Data Quality:")
    print("-" * 60)

    if data is None:
        print("  ❌ Data is None!")
        return False

    # Check signal
    signal = data.table['counts'].values
    errors = data.table['err'].values
    time = data.table['time'].values

    print(f"  Signal length: {len(signal)}")
    print(f"  Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"  Signal mean: {signal.mean():.3f}")

    # Check for problems
    has_nan = np.any(np.isnan(signal))
    has_inf = np.any(np.isinf(signal))
    has_zero = np.any(signal == 0)
    has_negative = np.any(signal < 0)

    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    print(f"  Contains zeros: {has_zero} ({np.sum(signal == 0)} points)")
    print(f"  Contains negatives: {has_negative} ({np.sum(signal < 0)} points)")

    # Check errors
    err_nan = np.any(np.isnan(errors))
    err_inf = np.any(np.isinf(errors))
    err_zero = np.any(errors == 0)
    err_negative = np.any(errors <= 0)

    print(f"\n  Error stats:")
    print(f"    Error range: [{errors.min():.3f}, {errors.max():.3f}]")
    print(f"    Contains NaN: {err_nan}")
    print(f"    Contains Inf: {err_inf}")
    print(f"    Contains zeros: {err_zero} ({np.sum(errors == 0)} points)")
    print(f"    Contains <= 0: {err_negative} ({np.sum(errors <= 0)} points)")

    # Check openbeam
    if data.openbeam_table is not None:
        ob_signal = data.openbeam_table['counts'].values
        ob_has_zero = np.any(ob_signal == 0)
        ob_has_negative = np.any(ob_signal < 0)

        print(f"\n  Openbeam:")
        print(f"    Range: [{ob_signal.min():.3f}, {ob_signal.max():.3f}]")
        print(f"    Contains zeros: {ob_has_zero} ({np.sum(ob_signal == 0)} points)")
        print(f"    Contains negatives: {ob_has_negative}")

        # Calculate transmission
        min_len = min(len(signal), len(ob_signal))
        transmission = signal[:min_len] / ob_signal[:min_len]

        print(f"\n  Transmission (signal/openbeam):")
        print(f"    Range: [{transmission.min():.6f}, {transmission.max():.6f}]")
        print(f"    Mean: {transmission.mean():.6f}")

        t_nan = np.any(np.isnan(transmission))
        t_inf = np.any(np.isinf(transmission))
        t_invalid = np.any((transmission <= 0) | (transmission > 1.1))

        print(f"    Contains NaN: {t_nan}")
        print(f"    Contains Inf: {t_inf}")
        print(f"    Invalid (T<=0 or T>1.1): {t_invalid}")

        if t_invalid:
            bad_mask = (transmission <= 0) | (transmission > 1.1)
            print(f"    Number of invalid T: {np.sum(bad_mask)}")
            print(f"    Invalid T values: {transmission[bad_mask][:10]}")  # Show first 10

    # Overall assessment
    all_ok = not (has_nan or has_inf or has_negative or err_nan or err_inf or err_negative)

    if all_ok:
        print(f"\n  ✅ Data looks good!")
    else:
        print(f"\n  ❌ Data has issues that will cause NaN in fitting!")

    return all_ok


def test_reconstruction_pipeline():
    """Test the full reconstruction pipeline and diagnose NaN sources."""
    print("="*70)
    print("DIAGNOSING NaN ISSUES IN RECONSTRUCTION → NBRAGG PIPELINE")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
                flux=5e6, duration=0.5, freq=20)

    diagnose_data_quality(data, "After loading")

    # Convolute
    print("\n2. Applying convolution...")
    data.convolute_response(200, bin_width=10)
    diagnose_data_quality(data, "After convolution")

    # Poisson sampling
    print("\n3. Applying Poisson sampling...")
    data.poisson_sample(flux=1e6, freq=20, measurement_time=30, seed=42)
    diagnose_data_quality(data, "After Poisson sampling")

    # Overlap
    print("\n4. Applying overlap...")
    data.overlap(kernel=[0, 25], total_time=50)
    diagnose_data_quality(data, "After overlap")

    # Reconstruction
    print("\n5. Reconstructing...")
    recon = Reconstruct(data)
    recon.filter(kind='fobi', noise_power=0.1, interpolate_kernel=True)

    # Create reconstructed data object
    print("\n6. Checking reconstructed data...")

    # The reconstructed signal
    reconstructed_signal = recon.reconstructed_data['counts'].values
    reconstructed_err = recon.reconstructed_data['err'].values

    print(f"\nReconstructed signal:")
    print(f"  Length: {len(reconstructed_signal)}")
    print(f"  Range: [{reconstructed_signal.min():.3f}, {reconstructed_signal.max():.3f}]")
    print(f"  Mean: {reconstructed_signal.mean():.3f}")
    print(f"  Contains NaN: {np.any(np.isnan(reconstructed_signal))}")
    print(f"  Contains Inf: {np.any(np.isinf(reconstructed_signal))}")
    print(f"  Contains zeros: {np.any(reconstructed_signal == 0)} ({np.sum(reconstructed_signal == 0)} points)")
    print(f"  Contains negatives: {np.any(reconstructed_signal < 0)} ({np.sum(reconstructed_signal < 0)} points)")

    print(f"\nReconstructed errors:")
    print(f"  Range: [{reconstructed_err.min():.3f}, {reconstructed_err.max():.3f}]")
    print(f"  Contains NaN: {np.any(np.isnan(reconstructed_err))}")
    print(f"  Contains zeros: {np.any(reconstructed_err == 0)}")
    print(f"  Contains <= 0: {np.any(reconstructed_err <= 0)}")

    # Check reconstructed openbeam
    if recon.reference_openbeam is not None:
        reconstructed_ob = recon.reconstructed_openbeam['counts'].values

        print(f"\nReconstructed openbeam:")
        print(f"  Length: {len(reconstructed_ob)}")
        print(f"  Range: [{reconstructed_ob.min():.3f}, {reconstructed_ob.max():.3f}]")
        print(f"  Contains zeros: {np.any(reconstructed_ob == 0)} ({np.sum(reconstructed_ob == 0)} points)")

        # Check transmission from reconstructed data
        min_len = min(len(reconstructed_signal), len(reconstructed_ob))
        reconstructed_transmission = reconstructed_signal[:min_len] / reconstructed_ob[:min_len]

        print(f"\nTransmission from reconstructed data:")
        print(f"  Range: [{reconstructed_transmission.min():.6f}, {reconstructed_transmission.max():.6f}]")
        print(f"  Mean: {reconstructed_transmission.mean():.6f}")
        print(f"  Contains NaN: {np.any(np.isnan(reconstructed_transmission))}")
        print(f"  Contains Inf: {np.any(np.isinf(reconstructed_transmission))}")

        invalid_t = (reconstructed_transmission <= 0) | (reconstructed_transmission > 1.1)
        print(f"  Invalid transmission (T<=0 or T>1.1): {np.sum(invalid_t)} points")

        if np.any(invalid_t):
            print(f"\n  ❌ FOUND THE PROBLEM!")
            print(f"     Invalid transmission values will cause NaN in nbragg fitting!")
            print(f"     This happens when:")
            print(f"       - Reconstructed signal has zeros → T=0 → log(T)=NaN")
            print(f"       - Reconstructed signal > openbeam → T>1 → invalid")
            print(f"       - Reconstructed openbeam has zeros → division by zero")

            # Show where the problems are
            problem_indices = np.where(invalid_t)[0]
            print(f"\n  Problem locations (first 10):")
            for i in problem_indices[:10]:
                print(f"    Index {i}: signal={reconstructed_signal[i]:.3f}, "
                      f"openbeam={reconstructed_ob[i]:.3f}, T={reconstructed_transmission[i]:.6f}")


if __name__ == "__main__":
    test_reconstruction_pipeline()
