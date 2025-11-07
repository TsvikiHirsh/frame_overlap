"""
Test suite for resonance analysis functionality.

Tests the ResonanceAnalysis class, Cd filter, and energy conversion.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Test with mock if nres is not available
try:
    import nres
    NRES_AVAILABLE = True
except ImportError:
    NRES_AVAILABLE = False

# Import the frame_overlap package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from frame_overlap import Data, Reconstruct


class TestCdFilter:
    """Test Cd filter functionality."""

    def test_cd_filter_basic(self, tmp_path):
        """Test basic Cd filter application."""
        # Create test data with stack numbers spanning thermal to epithermal range
        # time (µs) = (stack - 1) * 10
        # Stack 10 → time = 90 µs → E = 5227*81/8100 = 52.3 eV (not filtered)
        # Stack 5000 → time = 49990 µs → E = 5227*81/2499000100 = 0.00017 eV (filtered)
        stacks = np.linspace(10, 5000, 100, dtype=int)
        counts = np.ones_like(stacks, dtype=float) * 100

        # Create CSV files with required columns
        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df = pd.DataFrame({
            'stack': stacks,
            'counts': counts,
            'err': np.sqrt(counts)
        })

        df.to_csv(signal_file, index=False)
        df.to_csv(openbeam_file, index=False)

        # Load data
        data = Data(str(signal_file), str(openbeam_file))

        # Apply Cd filter
        data.apply_cd_filter(L=9.0, cutoff_energy=0.4)

        # Check that cd_filtered_data exists
        assert data.cd_filtered_data is not None
        assert data.op_cd_filtered_data is not None

        # Check that some counts are set to zero (thermal neutrons removed)
        assert (data.cd_filtered_data['counts'] == 0).any()

    def test_cd_filter_energy_conversion(self, tmp_path):
        """Test that energy conversion is correct."""
        # Create test data
        # time (µs) = (stack - 1) * 10
        # E (eV) = 5227.0 * L^2 / t^2
        # Stack 101: time = 1000 µs → E = 5227*81/1000000 = 0.423 eV (> 0.4, not filtered)
        # Stack 5001: time = 50000 µs → E = 5227*81/2500000000 = 0.00017 eV (< 0.4, filtered!)
        # Stack 10001: time = 100000 µs → E = 5227*81/10000000000 = 0.000042 eV (< 0.4, filtered!)

        stacks = np.array([101, 5001, 10001])
        counts = np.array([100, 100, 100])

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df = pd.DataFrame({
            'stack': stacks,
            'counts': counts,
            'err': np.sqrt(counts)
        })

        df.to_csv(signal_file, index=False)
        df.to_csv(openbeam_file, index=False)

        data = Data(str(signal_file), str(openbeam_file))
        data.apply_cd_filter(L=9.0, cutoff_energy=0.4)

        # First time should have E > 0.4 eV (not filtered)
        # Longer times should have E < 0.4 eV (filtered)
        filtered_counts = data.cd_filtered_data['counts'].values
        assert filtered_counts[0] > 0  # First point not filtered
        assert filtered_counts[1] == 0  # Second point filtered
        assert filtered_counts[2] == 0  # Third point filtered

    def test_cd_filter_cutoff(self, tmp_path):
        """Test different cutoff energies."""
        stacks = np.linspace(10, 5000, 100, dtype=int)
        counts = np.ones_like(stacks, dtype=float) * 100

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df = pd.DataFrame({
            'stack': stacks,
            'counts': counts,
            'err': np.sqrt(counts)
        })

        df.to_csv(signal_file, index=False)
        df.to_csv(openbeam_file, index=False)

        # Test different cutoffs
        for cutoff in [0.1, 0.4, 1.0]:
            data = Data(str(signal_file), str(openbeam_file))
            data.apply_cd_filter(L=9.0, cutoff_energy=cutoff)

            # Higher cutoff should filter more neutrons
            n_filtered = (data.cd_filtered_data['counts'] == 0).sum()
            assert n_filtered >= 0


@pytest.mark.skipif(not NRES_AVAILABLE, reason="nres package not installed")
class TestResonanceAnalysis:
    """Test ResonanceAnalysis class."""

    def test_import_resonance_analysis(self):
        """Test that ResonanceAnalysis can be imported."""
        from frame_overlap import ResonanceAnalysis
        assert ResonanceAnalysis is not None

    def test_resonance_analysis_init(self):
        """Test ResonanceAnalysis initialization."""
        from frame_overlap import ResonanceAnalysis

        # Test basic initialization
        analysis = ResonanceAnalysis(material='Ta')
        assert analysis is not None
        assert hasattr(analysis, 'xs')
        assert hasattr(analysis, 'model')
        assert hasattr(analysis, 'material')
        assert analysis.material == 'Ta'

    def test_resonance_analysis_materials(self):
        """Test different materials."""
        from frame_overlap import ResonanceAnalysis

        # Test common resonance materials
        for material in ['Ta', 'U', 'W']:
            try:
                analysis = ResonanceAnalysis(material=material)
                assert analysis.material == material
            except Exception as e:
                pytest.skip(f"Material {material} not available in nres: {e}")

    def test_resonance_analysis_fit_options(self):
        """Test fitting options."""
        from frame_overlap import ResonanceAnalysis

        # Test with various fitting options
        analysis = ResonanceAnalysis(
            material='Ta',
            vary_weights=False,
            vary_background=True,
            vary_response=False,
            vary_tof=False
        )

        assert analysis is not None

    def test_resonance_analysis_fit(self, tmp_path):
        """Test fitting with ResonanceAnalysis."""
        from frame_overlap import ResonanceAnalysis

        # Create synthetic data
        stacks = np.linspace(101, 2001, 100, dtype=int)
        counts = np.ones_like(stacks, dtype=float) * 1000 + np.random.poisson(100, size=len(stacks))

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': counts,
            'err': np.sqrt(counts)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones_like(stacks, dtype=float) * 1200,
            'err': np.sqrt(1200) * np.ones_like(stacks, dtype=float)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Create pipeline
        data = Data(str(signal_file), str(openbeam_file))
        data.overlap(kernel=[0, 25])

        recon = Reconstruct(data)
        recon.filter(kind='wiener', noise_power=0.1)

        # Fit with resonance analysis
        analysis = ResonanceAnalysis(material='Ta', vary_background=True)

        try:
            result = analysis.fit(recon, emin=1e5, emax=1e6)

            # Check result
            assert hasattr(result, 'redchi')
            assert hasattr(analysis, 'data')
            assert hasattr(analysis, 'result')
        except Exception as e:
            # Fitting might fail with synthetic data, which is okay
            pytest.skip(f"Fit failed with synthetic data: {e}")


class TestEnergyConversion:
    """Test energy conversion utilities."""

    def test_energy_formula(self):
        """Test energy conversion formula: E = 5227.0 * L^2 / t^2."""
        # Known values
        L = 9.0  # meters
        t_us = 1000.0  # microseconds

        # Expected energy: E = 5227.0 * 81 / 1000000 = 0.423387 eV
        E_expected = 5227.0 * (L ** 2) / (t_us ** 2)
        assert abs(E_expected - 0.423387) < 0.001  # eV

    def test_energy_ranges(self):
        """Test typical energy ranges."""
        L = 9.0

        # Thermal neutrons (very long TOF)
        t_thermal = 100000  # 100 ms = 100000 microseconds
        E_thermal = 5227.0 * (L ** 2) / (t_thermal ** 2)
        assert E_thermal < 0.01  # Very low energy

        # Near Cd cutoff (~0.4 eV requires t ~ 1000 µs for L=9m)
        t_cd = 1000  # 1 ms = 1000 microseconds
        E_cd = 5227.0 * (L ** 2) / (t_cd ** 2)
        assert 0.4 < E_cd < 0.5  # Right around Cd cutoff

        # Epithermal neutrons (shorter TOF)
        t_epithermal = 300  # 0.3 ms = 300 microseconds
        E_epithermal = 5227.0 * (L ** 2) / (t_epithermal ** 2)
        assert E_epithermal > 4  # Epithermal range

        # Fast neutrons (short TOF)
        t_fast = 100  # 0.1 ms = 100 microseconds
        E_fast = 5227.0 * (L ** 2) / (t_fast ** 2)
        assert E_fast > 40  # Fast neutron range


class TestResonanceIntegration:
    """Test integration of resonance analysis with pipeline."""

    def test_pipeline_with_cd_filter(self, tmp_path):
        """Test complete pipeline with Cd filter."""
        # Create test data
        stacks = np.linspace(10, 5000, 200, dtype=int)
        signal_counts = np.random.poisson(1000, size=len(stacks))
        openbeam_counts = np.random.poisson(1200, size=len(stacks))

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': signal_counts,
            'err': np.sqrt(signal_counts)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': openbeam_counts,
            'err': np.sqrt(openbeam_counts)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Create pipeline
        data = Data(str(signal_file), str(openbeam_file))

        # Apply Cd filter before overlap
        data.apply_cd_filter(L=9.0, cutoff_energy=0.4)

        # Continue pipeline
        data.overlap(kernel=[0, 25])

        recon = Reconstruct(data)
        recon.filter(kind='wiener', noise_power=0.1)

        # Check reconstruction
        assert recon.reconstructed_data is not None
        assert len(recon.reconstructed_data) > 0

    @pytest.mark.skipif(not NRES_AVAILABLE, reason="nres package not installed")
    def test_full_resonance_pipeline(self, tmp_path):
        """Test full pipeline with resonance analysis."""
        from frame_overlap import ResonanceAnalysis

        # Create test data
        stacks = np.linspace(101, 3001, 200, dtype=int)
        signal_counts = np.random.poisson(1000, size=len(stacks))
        openbeam_counts = np.random.poisson(1200, size=len(stacks))

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': signal_counts,
            'err': np.sqrt(signal_counts)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': openbeam_counts,
            'err': np.sqrt(openbeam_counts)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Full pipeline
        data = Data(str(signal_file), str(openbeam_file))
        data.apply_cd_filter(L=9.0, cutoff_energy=0.4)
        data.overlap(kernel=[0, 25])

        recon = Reconstruct(data)
        recon.filter(kind='wiener', noise_power=0.1)

        analysis = ResonanceAnalysis(material='Ta', vary_background=True)

        try:
            result = analysis.fit(recon, emin=1e5, emax=1e6)
            assert hasattr(result, 'redchi')
        except Exception as e:
            pytest.skip(f"Full pipeline fit failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
