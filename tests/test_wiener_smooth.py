"""
Test suite for smoothed Wiener filter.

Tests the wiener_smooth filter method that applies smoothing before deconvolution,
following the approach from the paper.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from frame_overlap import Data, Reconstruct


class TestWienerSmooth:
    """Test smoothed Wiener filter functionality."""

    def test_wiener_smooth_basic(self, tmp_path):
        """Test basic smoothed Wiener filter application."""
        # Create synthetic data
        n_points = 500
        stacks = np.arange(1, n_points + 1)

        # Create a signal with some structure
        times = (stacks - 1) * 10
        signal = 1000 * np.exp(-times / 10000) + 100
        signal += np.random.poisson(50, size=n_points)

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': signal,
            'err': np.sqrt(signal)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1200,
            'err': np.sqrt(1200) * np.ones(n_points)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Create pipeline
        data = Data(str(signal_file), str(openbeam_file))
        data.overlap(kernel=[0, 12.5, 12.5, 12.5])  # 4 frames

        recon = Reconstruct(data)
        recon.filter(kind='wiener_smooth', noise_power=0.1)

        # Check that reconstruction succeeded
        assert recon.reconstructed_data is not None
        assert len(recon.reconstructed_data) > 0
        assert 'counts' in recon.reconstructed_data.columns
        assert 'err' in recon.reconstructed_data.columns

    def test_wiener_smooth_vs_regular(self, tmp_path):
        """Compare smoothed Wiener with regular Wiener."""
        # Create synthetic data
        n_points = 500
        stacks = np.arange(1, n_points + 1)

        times = (stacks - 1) * 10
        signal = 1000 * np.exp(-times / 10000) + 100
        signal += np.random.poisson(50, size=n_points)

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': signal,
            'err': np.sqrt(signal)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1200,
            'err': np.sqrt(1200) * np.ones(n_points)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Test with regular Wiener
        data1 = Data(str(signal_file), str(openbeam_file))
        data1.overlap(kernel=[0, 12.5, 12.5, 12.5])
        recon1 = Reconstruct(data1)
        recon1.filter(kind='wiener', noise_power=0.1)

        # Test with smoothed Wiener
        data2 = Data(str(signal_file), str(openbeam_file))
        data2.overlap(kernel=[0, 12.5, 12.5, 12.5])
        recon2 = Reconstruct(data2)
        recon2.filter(kind='wiener_smooth', noise_power=0.1)

        # Both should produce valid reconstructions
        assert recon1.reconstructed_data is not None
        assert recon2.reconstructed_data is not None

        # They should be different (smoothing should affect the result)
        diff = np.abs(recon1.reconstructed_data['counts'].values -
                     recon2.reconstructed_data['counts'].values)
        assert np.mean(diff) > 0  # Should be different

    def test_smooth_window_parameter(self, tmp_path):
        """Test different smoothing window sizes."""
        # Create synthetic data
        n_points = 500
        stacks = np.arange(1, n_points + 1)

        times = (stacks - 1) * 10
        signal = 1000 * np.exp(-times / 10000) + 100
        signal += np.random.poisson(50, size=n_points)

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': signal,
            'err': np.sqrt(signal)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1200,
            'err': np.sqrt(1200) * np.ones(n_points)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Test with different window sizes
        results = {}
        for window in [3, 5, 7, 11]:
            data = Data(str(signal_file), str(openbeam_file))
            data.overlap(kernel=[0, 12.5, 12.5, 12.5])
            recon = Reconstruct(data)
            recon.filter(kind='wiener_smooth', noise_power=0.1, smooth_window=window)
            results[window] = recon.reconstructed_data['counts'].values.copy()

        # Different windows should produce different results
        assert not np.allclose(results[3], results[11])

    def test_many_frames(self, tmp_path):
        """Test with many overlapping frames (like the paper's 32 frames)."""
        # Create synthetic data
        n_points = 3000  # 30ms at 10µs bins = 3000 points
        stacks = np.arange(1, n_points + 1)

        times = (stacks - 1) * 10  # Time in µs
        signal = 1000 * np.exp(-times / 10000) + 100
        signal += np.random.poisson(50, size=n_points)

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': signal,
            'err': np.sqrt(signal)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1200,
            'err': np.sqrt(1200) * np.ones(n_points)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Create 32 random frames in 30ms (like the paper)
        np.random.seed(42)
        n_frames = 32
        frame_times_ms = sorted(np.random.uniform(0, 30, n_frames))

        # Convert to list format for overlap function
        kernel = [frame_times_ms[0]]  # First frame
        for i in range(1, n_frames):
            kernel.append(frame_times_ms[i] - frame_times_ms[i-1])

        data = Data(str(signal_file), str(openbeam_file))
        data.overlap(kernel=frame_times_ms)  # Use absolute times

        recon = Reconstruct(data)
        recon.filter(kind='wiener_smooth', noise_power=1.0)

        # Check reconstruction
        assert recon.reconstructed_data is not None
        assert len(recon.reconstructed_data) > 0

        # Check statistics if available
        if hasattr(recon, 'statistics') and recon.statistics:
            chi2_per_dof = recon.statistics.get('chi2_per_dof', None)
            if chi2_per_dof is not None:
                # Should have reasonable reconstruction quality
                assert chi2_per_dof > 0

    def test_wiener_smooth_reconstruction_quality(self, tmp_path):
        """Test that smoothed Wiener provides good reconstruction quality."""
        # Create synthetic data with known structure
        n_points = 1000
        stacks = np.arange(1, n_points + 1)

        times = (stacks - 1) * 10
        # Create a smooth exponential decay
        true_signal = 1000 * np.exp(-times / 10000) + 200

        # Add some noise
        noisy_signal = true_signal + np.random.normal(0, 20, size=n_points)
        noisy_signal = np.maximum(noisy_signal, 1)  # Ensure positive

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df_signal = pd.DataFrame({
            'stack': stacks,
            'counts': noisy_signal,
            'err': np.sqrt(noisy_signal)
        })

        df_openbeam = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1200,
            'err': np.sqrt(1200) * np.ones(n_points)
        })

        df_signal.to_csv(signal_file, index=False)
        df_openbeam.to_csv(openbeam_file, index=False)

        # Apply overlap
        data = Data(str(signal_file), str(openbeam_file))
        data.overlap(kernel=[0, 10, 10, 10, 10])  # 5 frames

        # Reconstruct with smoothed Wiener
        recon = Reconstruct(data)
        recon.filter(kind='wiener_smooth', noise_power=0.1)

        # Check that reconstruction exists
        assert recon.reconstructed_data is not None

        # The reconstructed signal should have similar magnitude to reference
        if recon.reference_data is not None:
            ref_mean = recon.reference_data['counts'].mean()
            rec_mean = recon.reconstructed_data['counts'].mean()
            # Should be within same order of magnitude
            assert rec_mean / ref_mean > 0.1
            assert rec_mean / ref_mean < 10


class TestFilterOptions:
    """Test filter method options and error handling."""

    def test_invalid_filter_kind(self, tmp_path):
        """Test that invalid filter kind raises error."""
        # Create minimal data
        n_points = 100
        stacks = np.arange(1, n_points + 1)

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1000,
            'err': np.sqrt(1000) * np.ones(n_points)
        })

        df.to_csv(signal_file, index=False)
        df.to_csv(openbeam_file, index=False)

        data = Data(str(signal_file), str(openbeam_file))
        data.overlap(kernel=[0, 25])

        recon = Reconstruct(data)

        with pytest.raises(ValueError, match="Unknown filter kind"):
            recon.filter(kind='invalid_filter')

    def test_all_filter_kinds(self, tmp_path):
        """Test that all filter kinds work."""
        # Create minimal data
        n_points = 100
        stacks = np.arange(1, n_points + 1)

        signal_file = tmp_path / "signal.csv"
        openbeam_file = tmp_path / "openbeam.csv"

        df = pd.DataFrame({
            'stack': stacks,
            'counts': np.ones(n_points) * 1000 + np.random.poisson(50, n_points),
            'err': np.sqrt(1000) * np.ones(n_points)
        })

        df.to_csv(signal_file, index=False)
        df.to_csv(openbeam_file, index=False)

        # Test all filter kinds
        for kind in ['wiener', 'wiener_smooth', 'wiener_adaptive', 'lucy', 'tikhonov']:
            data = Data(str(signal_file), str(openbeam_file))
            data.overlap(kernel=[0, 25])
            recon = Reconstruct(data)

            if kind == 'lucy':
                recon.filter(kind=kind, iterations=10)
            else:
                recon.filter(kind=kind, noise_power=0.1)

            assert recon.reconstructed_data is not None
            assert len(recon.reconstructed_data) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
