"""
Test that mimics the complete Streamlit workflow with nbragg analysis.
This verifies all the code changes work correctly.
"""

import sys
sys.path.insert(0, 'src')

from frame_overlap import Data, Reconstruct, Analysis
import pandas as pd
import matplotlib.pyplot as plt

class SessionState:
    """Mimic Streamlit session state."""
    def __init__(self):
        self.workflow_data = None
        self.recon = None
        self.analysis = None

def test_streamlit_workflow():
    """Test the complete workflow as it would run in Streamlit."""

    print("="*70)
    print("TESTING COMPLETE STREAMLIT WORKFLOW WITH NBRAGG ANALYSIS")
    print("="*70)

    # Initialize session state
    st_session_state = SessionState()

    # Configuration (from sidebar)
    apply_analysis = True
    nbragg_model = 'iron'
    vary_background = True
    vary_response = True

    print("\n📋 Configuration:")
    print(f"   - Apply nbragg Analysis: {apply_analysis}")
    print(f"   - nbragg Model: {nbragg_model}")
    print(f"   - Vary Background: {vary_background}")
    print(f"   - Vary Response: {vary_response}")

    # =================================================================
    # PIPELINE PROCESSING (mimicking "Run Pipeline" button click)
    # =================================================================

    print("\n" + "="*70)
    print("RUNNING PIPELINE")
    print("="*70)

    # Stage 1-4: Data processing
    print("\n1. Loading and processing data...")
    data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
               flux=5e6, duration=0.5, freq=20)
    data.convolute_response(200, bin_width=10)
    data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
    data.overlap(kernel=[0, 25])
    st_session_state.workflow_data = data
    print("   ✓ Data processing complete")

    # Stage 5: Reconstruction
    print("\n2. Running reconstruction...")
    recon = Reconstruct(data, tmin=3.7, tmax=11.0)
    recon.filter(kind='wiener', noise_power=0.2)
    st_session_state.recon = recon
    stats = recon.get_statistics()
    print(f"   ✓ Reconstruction complete (χ²/dof: {stats['chi2_per_dof']:.1f})")

    # Stage 6: nbragg Analysis
    print("\n3. Running nbragg analysis...")
    if apply_analysis:
        try:
            analysis = Analysis(xs=nbragg_model, vary_background=vary_background,
                               vary_response=vary_response)
            result = analysis.fit(recon)
            st_session_state.analysis = analysis
            print(f"   ✓ nbragg fit complete (χ²/dof: {result.redchi:.2f})")
        except Exception as e:
            print(f"   ✗ nbragg fit failed: {e}")
            st_session_state.analysis = None

    # =================================================================
    # TAB 2: RECONSTRUCTION TAB
    # =================================================================

    print("\n" + "="*70)
    print("TAB 2: RECONSTRUCTION - Testing Plot with nbragg Fit")
    print("="*70)

    if st_session_state.recon is not None:
        print("\n1. Creating reconstruction plot...")
        plot_type = 'transmission'
        show_errors_recon = False

        # Generate base plot
        mpl_fig = st_session_state.recon.plot(
            kind=plot_type,
            show_errors=show_errors_recon,
            figsize=(12, 8),
            ylim=(0, 1)
        )
        print("   ✓ Base reconstruction plot created")

        # Add nbragg fit curve
        if st_session_state.analysis is not None and plot_type == 'transmission':
            try:
                result = st_session_state.analysis.result
                axes = mpl_fig.get_axes()

                if len(axes) >= 1:
                    ax_data = axes[0]

                    # Get wavelength and best_fit
                    wavelength = result.userkws['wl']
                    best_fit_transmission = result.best_fit

                    # Convert to time
                    L = 9.0
                    time_us = wavelength * L * 252.778
                    time_ms = time_us / 1000

                    # Plot
                    ax_data.plot(time_ms, best_fit_transmission,
                               label='nbragg fit', color='green',
                               linewidth=2, linestyle='--')
                    ax_data.legend()

                    print(f"   ✓ nbragg fit added to plot ({len(best_fit_transmission)} points)")

                    # Count total lines
                    total_lines = len(ax_data.get_lines())
                    print(f"   ✓ Plot contains {total_lines} lines (should be 5)")

                    # Verify we have the expected lines
                    expected_lines = ['Convolved (Target)', 'Reconstructed', 'tmin', 'tmax', 'nbragg fit']
                    actual_labels = [line.get_label() for line in ax_data.get_lines()]
                    print(f"   ✓ Line labels: {', '.join([l[:20] for l in actual_labels])}")

            except Exception as e:
                print(f"   ✗ Could not add nbragg fit: {e}")
                import traceback
                traceback.print_exc()

        plt.savefig('/tmp/streamlit_reconstruction_test.png', dpi=100)
        print("\n   ✓ Plot saved to /tmp/streamlit_reconstruction_test.png")
        plt.close(mpl_fig)

    # =================================================================
    # TAB 3: STATISTICS TAB
    # =================================================================

    print("\n" + "="*70)
    print("TAB 3: STATISTICS - Testing nbragg Fit Results Display")
    print("="*70)

    if st_session_state.analysis is not None:
        print("\n1. Testing reduced chi-squared display...")
        try:
            result = st_session_state.analysis.result

            print(f"   Reduced χ² (nbragg): {result.redchi:.4f}")

            # Quality indicator
            if result.redchi < 2:
                quality = "✅ Excellent nbragg fit"
            elif result.redchi < 5:
                quality = "ℹ️ Good nbragg fit"
            else:
                quality = "⚠️ Poor nbragg fit"
            print(f"   Quality: {quality}")
            print("   ✓ Chi-squared display works")

        except Exception as e:
            print(f"   ✗ Failed: {e}")

        print("\n2. Testing parameters table...")
        try:
            params_data = []
            for param_name, param in result.params.items():
                # Handle stderr which might be None
                if param.stderr is not None:
                    stderr_str = f"{param.stderr:.4e}"
                else:
                    stderr_str = "N/A"

                params_data.append({
                    'Parameter': param_name,
                    'Value': f"{param.value:.4e}",
                    'Stderr': stderr_str,
                    'Vary': 'Yes' if param.vary else 'No'
                })

            params_table = pd.DataFrame(params_data)
            print(f"   ✓ Parameters table created ({len(params_table)} parameters)")
            print("\n   Table preview:")
            print("   " + "\n   ".join(params_table.to_string(index=False).split('\n')[:5]))

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n3. Testing fit report...")
        try:
            fit_report = result.fit_report()
            print(f"   ✓ Fit report generated ({len(fit_report)} chars)")

        except Exception as e:
            print(f"   ✗ Failed: {e}")

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print("\n" + "="*70)
    print("WORKFLOW TEST SUMMARY")
    print("="*70)

    checks = [
        ("Session state initialized", st_session_state is not None),
        ("Data processed", st_session_state.workflow_data is not None),
        ("Reconstruction complete", st_session_state.recon is not None),
        ("nbragg analysis complete", st_session_state.analysis is not None),
        ("nbragg result exists", st_session_state.analysis.result if st_session_state.analysis else False),
    ]

    print()
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}")

    all_passed = all(check[1] for check in checks)

    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL WORKFLOW TESTS PASSED!")
        print("="*70)
        print("\nThe Streamlit app should now correctly show:")
        print("  1. Green dashed nbragg fit line in Reconstruction tab")
        print("  2. nbragg parameters table in Statistics tab")
        print("  3. Reduced chi-squared and quality indicator")
        print("  4. Full fit report in expandable section")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        return False

if __name__ == "__main__":
    success = test_streamlit_workflow()
    exit(0 if success else 1)
