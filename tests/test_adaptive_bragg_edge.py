"""
Tests for Adaptive Bragg Edge Measurement System
"""

import pytest
import numpy as np
from frame_overlap.bragg_edge_model import (
    BraggEdge,
    BraggEdgeSample,
    IncidentSpectrum,
    TOFCalibration,
    MeasurementSimulator,
    estimate_edge_position,
    calculate_edge_gradient
)
from frame_overlap.chopper_patterns import (
    PatternLibrary,
    ForwardModel,
    calculate_pattern_efficiency,
    validate_pattern,
    ChopperConstraints
)
from frame_overlap.adaptive_measurement import (
    BayesianEdgeOptimizer,
    GradientFocusedMeasurement,
    MultiResolutionEdgeSearch,
    RealTimeAdaptiveSystem,
    EdgePosterior
)
from frame_overlap.bragg_edge_optimizer import (
    BraggEdgeMeasurementSystem,
    AdaptiveEdgeOptimizer,
    MeasurementTarget,
    optimize_measurement_strategy
)
from frame_overlap.performance_metrics import (
    PerformanceEvaluator,
    PerformanceMetrics,
    calculate_mutual_information
)


class TestBraggEdgeModel:
    """Tests for Bragg edge physical model"""

    def test_tof_calibration(self):
        """Test TOF calibration"""
        calib = TOFCalibration(flight_path=10.0)

        # Test wavelength to TOF conversion
        wavelength = 4.0  # Angstrom
        tof = calib.wavelength_to_tof(wavelength)
        assert tof > 0
        assert isinstance(tof, (float, np.floating))

        # Test round-trip conversion
        wl_back = calib.tof_to_wavelength(tof)
        assert np.isclose(wl_back, wavelength, rtol=1e-6)

    def test_bragg_edge_creation(self):
        """Test Bragg edge creation"""
        edge = BraggEdge(position=4.05, height=0.6, width=0.1)

        assert edge.position == 4.05
        assert edge.height == 0.6
        assert edge.width == 0.1

        # Test edge function
        wavelengths = np.linspace(3.0, 5.0, 100)
        edge_values = edge.edge_function(wavelengths)

        assert len(edge_values) == len(wavelengths)
        assert np.all(edge_values >= 0)
        assert np.all(edge_values <= 1)

        # Edge should be near 0.5 at edge position
        idx = np.argmin(np.abs(wavelengths - 4.05))
        assert np.isclose(edge_values[idx], 0.5, atol=0.1)

    def test_iron_sample_creation(self):
        """Test iron sample creation"""
        sample = BraggEdgeSample.create_iron_sample(strain=0.001)

        assert sample.material == 'Fe'
        assert len(sample.edges) == 3

        # Check strain is applied
        expected_position = 4.05 * 1.001
        assert np.isclose(sample.edges[0].position, expected_position, rtol=1e-6)

        # Test transmission calculation
        wavelengths = np.linspace(1.0, 10.0, 1000)
        transmission = sample.transmission(wavelengths)

        assert len(transmission) == len(wavelengths)
        assert np.all(transmission >= 0)
        assert np.all(transmission <= 1)

    def test_incident_spectrum(self):
        """Test incident spectrum models"""
        wavelengths = np.linspace(1.0, 10.0, 100)

        # Test Maxwellian spectrum
        maxwellian = IncidentSpectrum(spectrum_type='maxwellian', temperature=300.0)
        intensity = maxwellian.intensity(wavelengths)

        assert len(intensity) == len(wavelengths)
        assert np.all(intensity >= 0)
        assert np.max(intensity) == 1.0  # Should be normalized

        # Test flat spectrum
        flat = IncidentSpectrum(spectrum_type='flat')
        intensity_flat = flat.intensity(wavelengths)

        assert np.allclose(intensity_flat, 1.0)

    def test_measurement_simulator(self):
        """Test measurement simulator"""
        sample = BraggEdgeSample.create_iron_sample()
        spectrum = IncidentSpectrum(spectrum_type='maxwellian')
        calib = TOFCalibration(flight_path=10.0)

        simulator = MeasurementSimulator(
            sample, spectrum, calib,
            wavelength_range=(1.0, 10.0),
            n_wavelength_bins=1000
        )

        # Create simple chopper pattern
        pattern = PatternLibrary.uniform_sparse(1000, duty_cycle=0.1, seed=42)

        # Simulate measurement
        times, counts = simulator.simulate_tof_measurement(
            pattern, n_time_bins=1000, flux=1e6,
            measurement_time=1.0, add_noise=True
        )

        assert len(times) == 1000
        assert len(counts) == 1000
        assert np.sum(counts) > 0


class TestChopperPatterns:
    """Tests for chopper pattern generation"""

    def test_uniform_sparse_pattern(self):
        """Test uniform sparse pattern"""
        pattern = PatternLibrary.uniform_sparse(1000, duty_cycle=0.1, seed=42)

        assert len(pattern) == 1000
        assert np.all((pattern == 0) | (pattern == 1))

        duty = np.sum(pattern) / len(pattern)
        assert 0.05 < duty < 0.15  # Allow some randomness

    def test_focused_window_pattern(self):
        """Test focused window pattern"""
        pattern = PatternLibrary.focused_window(
            1000, center=500, width=200, density=0.5, seed=42
        )

        assert len(pattern) == 1000

        # Most pulses should be in central region
        central = pattern[400:600]
        edges = np.concatenate([pattern[:400], pattern[600:]])

        assert np.sum(central) > np.sum(edges)

    def test_gradient_weighted_pattern(self):
        """Test gradient-weighted pattern"""
        gradient = np.exp(-((np.arange(1000) - 500) ** 2) / (2 * 50 ** 2))

        pattern = PatternLibrary.gradient_weighted(
            1000, gradient, target_duty_cycle=0.1, power=2.0, seed=42
        )

        assert len(pattern) == 1000

        # Most pulses should be near peak gradient
        central = pattern[450:550]
        edges = np.concatenate([pattern[:450], pattern[550:]])

        assert np.sum(central) > np.sum(edges) / 2

    def test_pattern_validation(self):
        """Test pattern validation"""
        constraints = ChopperConstraints(
            max_duty_cycle=0.5,
            min_pulse_width=10e-6
        )

        # Valid pattern
        pattern = PatternLibrary.uniform_sparse(1000, duty_cycle=0.1)
        is_valid, msg = validate_pattern(pattern, constraints, time_resolution=1e-6)
        assert is_valid

        # Invalid pattern (duty cycle too high)
        pattern_invalid = np.ones(1000)
        is_valid, msg = validate_pattern(pattern_invalid, constraints, time_resolution=1e-6)
        assert not is_valid

    def test_forward_model(self):
        """Test forward model"""
        wavelength_grid = np.linspace(1.0, 10.0, 100)
        time_grid = np.linspace(0, 0.01, 1000)

        calib = TOFCalibration(flight_path=10.0)
        spectrum = IncidentSpectrum(spectrum_type='flat')
        incident_intensity = spectrum.intensity(wavelength_grid)

        forward_model = ForwardModel(
            wavelength_grid,
            time_grid,
            calib.wavelength_to_tof,
            incident_intensity
        )

        # Create simple transmission and pattern
        transmission = np.ones(len(wavelength_grid)) * 0.8
        pattern = PatternLibrary.uniform_sparse(len(time_grid), duty_cycle=0.1, seed=42)

        # Predict measurement
        measurement = forward_model.predict_measurement(
            transmission, pattern, flux=1e6, measurement_time=1.0
        )

        assert len(measurement) == len(time_grid)
        assert np.sum(measurement) > 0


class TestAdaptiveMeasurement:
    """Tests for adaptive measurement strategies"""

    def test_edge_posterior(self):
        """Test edge posterior"""
        posterior = EdgePosterior(
            position_mean=4.05,
            position_std=0.1
        )

        # Test sampling
        samples = posterior.sample(n_samples=100, seed=42)

        assert samples.shape == (100, 3)
        assert np.abs(np.mean(samples[:, 0]) - 4.05) < 0.05
        assert np.all(samples[:, 1] > 0)  # Heights positive
        assert np.all(samples[:, 2] > 0)  # Widths positive

    def test_bayesian_optimizer_initialization(self):
        """Test Bayesian optimizer initialization"""
        wavelength_grid = np.linspace(1.0, 10.0, 100)
        time_grid = np.linspace(0, 0.01, 1000)

        calib = TOFCalibration(flight_path=10.0)
        spectrum = IncidentSpectrum(spectrum_type='flat')
        incident_intensity = spectrum.intensity(wavelength_grid)

        forward_model = ForwardModel(
            wavelength_grid,
            time_grid,
            calib.wavelength_to_tof,
            incident_intensity
        )

        optimizer = BayesianEdgeOptimizer(
            prior_position=4.0,
            prior_position_uncertainty=0.5,
            wavelength_grid=wavelength_grid,
            time_grid=time_grid,
            forward_model=forward_model
        )

        assert optimizer.posterior.position_mean == 4.0
        assert optimizer.posterior.position_std == 0.5

    def test_pattern_design(self):
        """Test adaptive pattern design"""
        wavelength_grid = np.linspace(1.0, 10.0, 100)
        time_grid = np.linspace(0, 0.01, 1000)

        calib = TOFCalibration(flight_path=10.0)
        spectrum = IncidentSpectrum(spectrum_type='flat')
        incident_intensity = spectrum.intensity(wavelength_grid)

        forward_model = ForwardModel(
            wavelength_grid,
            time_grid,
            calib.wavelength_to_tof,
            incident_intensity
        )

        optimizer = BayesianEdgeOptimizer(
            prior_position=4.0,
            prior_position_uncertainty=0.5,
            wavelength_grid=wavelength_grid,
            time_grid=time_grid,
            forward_model=forward_model
        )

        # Design pattern
        pattern = optimizer.design_next_pattern(
            n_time_bins=1000,
            target_duty_cycle=0.1,
            strategy='gradient_focused'
        )

        assert len(pattern) == 1000
        assert np.all((pattern == 0) | (pattern == 1))

    def test_gradient_focused_measurement(self):
        """Test gradient-focused measurement"""
        wavelength_grid = np.linspace(1.0, 10.0, 100)
        time_grid = np.linspace(0, 0.01, 1000)

        calib = TOFCalibration(flight_path=10.0)
        spectrum = IncidentSpectrum(spectrum_type='flat')
        incident_intensity = spectrum.intensity(wavelength_grid)

        forward_model = ForwardModel(
            wavelength_grid,
            time_grid,
            calib.wavelength_to_tof,
            incident_intensity
        )

        gf_measure = GradientFocusedMeasurement(
            wavelength_grid,
            time_grid,
            forward_model
        )

        # Generate pattern
        pattern = gf_measure.generate_pattern(1000, duty_cycle=0.1)

        assert len(pattern) == 1000
        assert np.all((pattern == 0) | (pattern == 1))


class TestBraggEdgeOptimizer:
    """Tests for main optimization interface"""

    def test_measurement_system_creation(self):
        """Test measurement system creation"""
        system = BraggEdgeMeasurementSystem(
            flight_path=10.0,
            wavelength_range=(3.0, 5.0)
        )

        assert system.flight_path == 10.0
        assert system.wavelength_range == (3.0, 5.0)
        assert len(system.wavelength_grid) > 0
        assert len(system.time_grid) > 0

    def test_measurement_target_creation(self):
        """Test measurement target creation"""
        target = MeasurementTarget(
            material='Fe',
            expected_edge=4.05,
            precision_required=0.01,
            max_measurement_time=300.0
        )

        assert target.material == 'Fe'
        assert target.expected_edge == 4.05
        assert target.precision_required == 0.01

    def test_adaptive_optimizer_creation(self):
        """Test adaptive optimizer creation"""
        system = BraggEdgeMeasurementSystem(flight_path=10.0)
        target = MeasurementTarget(
            material='Fe',
            expected_edge=4.05,
            precision_required=0.01
        )

        optimizer = AdaptiveEdgeOptimizer(
            system,
            target,
            strategy='bayesian'
        )

        assert optimizer.strategy == 'bayesian'
        assert optimizer.system == system
        assert optimizer.target == target

    def test_optimize_measurement_strategy(self):
        """Test high-level optimization function"""
        target = MeasurementTarget(
            material='Fe',
            expected_edge=4.05,
            precision_required=0.01,
            max_measurement_time=50.0
        )

        result = optimize_measurement_strategy(
            target,
            flight_path=10.0,
            flux=1e6,
            measurement_time_per_pattern=10.0,
            strategy='bayesian'
        )

        assert isinstance(result.edge_position, float)
        assert isinstance(result.edge_uncertainty, float)
        assert result.measurement_time > 0
        assert result.n_patterns > 0


class TestPerformanceMetrics:
    """Tests for performance evaluation"""

    def test_mutual_information(self):
        """Test mutual information calculation"""
        signal1 = np.random.randn(1000)
        signal2 = signal1 + np.random.randn(1000) * 0.1

        mi = calculate_mutual_information(signal1, signal2)

        assert mi >= 0
        assert isinstance(mi, float)

    def test_performance_metrics_creation(self):
        """Test performance metrics creation"""
        metrics = PerformanceMetrics(
            edge_position_error=0.01,
            edge_position_precision=0.005,
            measurement_time=100.0,
            total_counts=1e6,
            efficiency=0.8
        )

        assert metrics.edge_position_error == 0.01
        assert metrics.measurement_time == 100.0

        # Test conversion to dict
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert 'edge_position_error' in d

    def test_edge_position_precision(self):
        """Test edge position precision calculation"""
        wavelength = np.linspace(3.0, 5.0, 1000)
        edge = BraggEdge(position=4.05, height=0.6, width=0.1)

        # Create multiple measurements
        measurements = []
        for i in range(5):
            trans = edge.transmission_contribution(wavelength)
            trans += np.random.randn(len(trans)) * 0.01
            measurements.append(trans)

        wavelength_grids = [wavelength] * 5

        mean_error, precision = PerformanceEvaluator.edge_position_precision(
            measurements, wavelength_grids, true_edge=4.05
        )

        assert isinstance(mean_error, float)
        assert isinstance(precision, float)
        assert precision > 0

    def test_calculate_snr(self):
        """Test SNR calculation"""
        signal = np.ones(1000) * 10 + np.random.randn(1000)

        snr = PerformanceEvaluator.calculate_snr(signal)

        assert snr > 0
        assert isinstance(snr, float)


class TestEdgeEstimation:
    """Tests for edge estimation functions"""

    def test_calculate_edge_gradient(self):
        """Test gradient calculation"""
        wavelength = np.linspace(3.0, 5.0, 1000)
        edge = BraggEdge(position=4.05, height=0.6, width=0.1)
        transmission = edge.transmission_contribution(wavelength)

        gradient = calculate_edge_gradient(wavelength, transmission)

        assert len(gradient) == len(wavelength)
        assert np.max(gradient) > 0

        # Maximum gradient should be near edge position
        max_idx = np.argmax(gradient)
        max_wavelength = wavelength[max_idx]
        assert np.abs(max_wavelength - 4.05) < 0.2

    def test_estimate_edge_position(self):
        """Test edge position estimation"""
        wavelength = np.linspace(3.0, 5.0, 1000)
        edge = BraggEdge(position=4.05, height=0.6, width=0.1)
        transmission = edge.transmission_contribution(wavelength)

        pos, unc = estimate_edge_position(wavelength, transmission)

        assert isinstance(pos, float)
        assert isinstance(unc, float)
        assert unc > 0

        # Should be reasonably close to true position
        assert np.abs(pos - 4.05) < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
