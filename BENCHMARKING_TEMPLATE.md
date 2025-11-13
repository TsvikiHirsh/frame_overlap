# Benchmarking Template for Adaptive TOF Reconstruction

This document provides templates and guidelines for comprehensive benchmarking of reconstruction algorithms.

## Benchmark Suite Structure

```python
# File: src/frame_overlap/adaptive/evaluation.py

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt


class Benchmark:
    """
    Comprehensive benchmarking suite for reconstruction algorithms.

    This class provides tools to compare different reconstructors across
    multiple scenarios and metrics.
    """

    def __init__(self, reconstructors: List, scenarios: List[Dict]):
        """
        Initialize benchmark.

        Parameters
        ----------
        reconstructors : List[BaseReconstructor]
            List of reconstructors to compare
        scenarios : List[Dict]
            List of scenario configurations, each containing:
            - name: str
            - material: str
            - kernel: np.ndarray
            - n_events: int
            - flux: float
            - noise_level: float
        """
        self.reconstructors = reconstructors
        self.scenarios = scenarios
        self.results = []

    def run(self, verbose=True):
        """Run all benchmarks."""
        from tqdm import tqdm

        total = len(self.reconstructors) * len(self.scenarios)
        pbar = tqdm(total=total, desc="Benchmarking") if verbose else None

        for scenario in self.scenarios:
            # Generate synthetic data
            events = self._generate_scenario_data(scenario)
            true_spectrum = scenario['true_spectrum']

            for reconstructor in self.reconstructors:
                result = self._run_single_benchmark(
                    reconstructor,
                    events,
                    true_spectrum,
                    scenario
                )
                self.results.append(result)

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return self.get_results_dataframe()

    def _run_single_benchmark(
        self,
        reconstructor,
        events,
        true_spectrum,
        scenario
    ) -> Dict[str, Any]:
        """Run single benchmark and collect metrics."""

        # Reset reconstructor
        reconstructor.reset()

        # Time reconstruction
        start_time = time.time()
        result = reconstructor.reconstruct(events)
        elapsed_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(
            result.spectrum,
            true_spectrum,
            result.uncertainty
        )

        return {
            'reconstructor': reconstructor.__class__.__name__,
            'scenario': scenario['name'],
            'material': scenario['material'],
            'kernel': str(scenario['kernel']),
            'n_events': events.n_events,
            'computation_time': elapsed_time,
            'iterations': result.iterations,
            **metrics
        }

    def _calculate_metrics(
        self,
        recon_spectrum,
        true_spectrum,
        uncertainty
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""

        # Chi-squared
        chi2 = np.sum(((recon_spectrum - true_spectrum) / uncertainty) ** 2)
        chi2_per_dof = chi2 / len(true_spectrum)

        # RMSE
        rmse = np.sqrt(np.mean((recon_spectrum - true_spectrum) ** 2))

        # Normalized RMSE
        spectrum_range = true_spectrum.max() - true_spectrum.min()
        nrmse = rmse / spectrum_range if spectrum_range > 0 else np.inf

        # R-squared
        ss_res = np.sum((true_spectrum - recon_spectrum) ** 2)
        ss_tot = np.sum((true_spectrum - true_spectrum.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Mean absolute error
        mae = np.mean(np.abs(recon_spectrum - true_spectrum))

        # Peak position error (for Bragg edges)
        true_peak = np.argmax(np.abs(np.gradient(true_spectrum)))
        recon_peak = np.argmax(np.abs(np.gradient(recon_spectrum)))
        peak_error = np.abs(true_peak - recon_peak)

        return {
            'chi2': chi2,
            'chi2_per_dof': chi2_per_dof,
            'rmse': rmse,
            'nrmse': nrmse,
            'r_squared': r_squared,
            'mae': mae,
            'peak_error': peak_error
        }

    def _generate_scenario_data(self, scenario):
        """Generate synthetic data for scenario."""
        from .simulation import generate_synthetic_events

        events = generate_synthetic_events(
            material=scenario['material'],
            kernel=scenario['kernel'],
            n_events=scenario['n_events'],
            flux=scenario['flux'],
            noise_level=scenario.get('noise_level', 0.1)
        )

        return events

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        return pd.DataFrame(self.results)

    def plot_comparison(self, metric='chi2_per_dof', figsize=(12, 6)):
        """
        Plot comparison of reconstructors across scenarios.

        Parameters
        ----------
        metric : str
            Metric to plot (chi2_per_dof, rmse, nrmse, etc.)
        figsize : tuple
            Figure size
        """
        df = self.get_results_dataframe()

        fig, ax = plt.subplots(figsize=figsize)

        # Pivot for plotting
        pivot = df.pivot(
            index='scenario',
            columns='reconstructor',
            values=metric
        )

        pivot.plot(kind='bar', ax=ax)
        ax.set_ylabel(metric)
        ax.set_xlabel('Scenario')
        ax.set_title(f'Reconstructor Comparison: {metric}')
        ax.legend(title='Reconstructor')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_time_accuracy_tradeoff(self, figsize=(10, 6)):
        """Plot computation time vs reconstruction accuracy."""
        df = self.get_results_dataframe()

        fig, ax = plt.subplots(figsize=figsize)

        for reconstructor in df['reconstructor'].unique():
            df_recon = df[df['reconstructor'] == reconstructor]
            ax.scatter(
                df_recon['computation_time'],
                df_recon['chi2_per_dof'],
                label=reconstructor,
                s=100,
                alpha=0.7
            )

        ax.set_xlabel('Computation Time (s)')
        ax.set_ylabel('χ²/dof')
        ax.set_title('Time-Accuracy Tradeoff')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save_report(self, filepath: str):
        """Save benchmark report to CSV."""
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)

    def print_summary(self):
        """Print summary statistics."""
        df = self.get_results_dataframe()

        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80 + "\n")

        # Group by reconstructor
        grouped = df.groupby('reconstructor')

        metrics = ['chi2_per_dof', 'rmse', 'nrmse', 'r_squared', 'computation_time']

        for metric in metrics:
            if metric in df.columns:
                print(f"\n{metric.upper()}:")
                print("-" * 40)
                summary = grouped[metric].agg(['mean', 'std', 'min', 'max'])
                print(summary.to_string())

        print("\n" + "="*80)


def compare_reconstructors(
    reconstructors: List,
    event_dataset,
    true_spectrum,
    metric='chi2_per_dof'
) -> pd.DataFrame:
    """
    Quick comparison of multiple reconstructors on single dataset.

    Parameters
    ----------
    reconstructors : List[BaseReconstructor]
        Reconstructors to compare
    event_dataset : EventDataset
        Test data
    true_spectrum : np.ndarray
        Ground truth spectrum
    metric : str
        Primary metric for ranking

    Returns
    -------
    pd.DataFrame
        Comparison results, sorted by metric
    """
    results = []

    for recon in reconstructors:
        recon.reset()

        start = time.time()
        result = recon.reconstruct(event_dataset)
        elapsed = time.time() - start

        chi2 = np.sum(((result.spectrum - true_spectrum) / result.uncertainty) ** 2)
        chi2_dof = chi2 / len(true_spectrum)

        rmse = np.sqrt(np.mean((result.spectrum - true_spectrum) ** 2))

        results.append({
            'Reconstructor': recon.__class__.__name__,
            'Time (s)': elapsed,
            'χ²/dof': chi2_dof,
            'RMSE': rmse,
            'Iterations': result.iterations,
            'Converged': result.convergence
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by=metric if metric in df.columns else 'χ²/dof')

    return df
```

## Standard Test Scenarios

```python
# Standard scenarios for benchmarking

STANDARD_SCENARIOS = [
    {
        'name': 'Iron_NoOverlap',
        'material': 'iron',
        'kernel': np.array([0]),  # Baseline
        'n_events': 50000,
        'flux': 5e6,
        'noise_level': 0.1,
        'description': 'Baseline: no frame overlap'
    },
    {
        'name': 'Iron_TwoFrame_25ms',
        'material': 'iron',
        'kernel': np.array([0, 25]),
        'n_events': 50000,
        'flux': 5e6,
        'noise_level': 0.1,
        'description': '2-frame overlap, 25 ms spacing'
    },
    {
        'name': 'Iron_TwoFrame_12ms',
        'material': 'iron',
        'kernel': np.array([0, 12]),
        'n_events': 50000,
        'flux': 5e6,
        'noise_level': 0.1,
        'description': '2-frame overlap, 12 ms spacing (high overlap)'
    },
    {
        'name': 'Iron_ThreeFrame',
        'material': 'iron',
        'kernel': np.array([0, 16, 16, 16]),
        'n_events': 50000,
        'flux': 5e6,
        'noise_level': 0.1,
        'description': '3-frame overlap, uniform spacing'
    },
    {
        'name': 'Iron_HighFlux',
        'material': 'iron',
        'kernel': np.array([0, 5, 5, 5, 5]),
        'n_events': 100000,
        'flux': 1e7,
        'noise_level': 0.05,
        'description': 'High flux, tight frame spacing'
    },
    {
        'name': 'Tantalum_Resonances',
        'material': 'tantalum',
        'kernel': np.array([0, 8, 8, 8]),
        'n_events': 100000,
        'flux': 5e6,
        'noise_level': 0.1,
        'description': 'Resonance spectroscopy (narrow features)'
    },
    {
        'name': 'Iron_LowStatistics',
        'material': 'iron',
        'kernel': np.array([0, 25]),
        'n_events': 5000,
        'flux': 5e5,
        'noise_level': 0.3,
        'description': 'Low statistics scenario'
    },
]
```

## Example Usage

```python
from frame_overlap.adaptive import (
    BaselineReconstructor,
    WienerEventReconstructor,
    EMReconstructor,
    Benchmark
)

# Define reconstructors
reconstructors = [
    BaselineReconstructor(tof_range=(1000, 20000), n_bins=1000),
    WienerEventReconstructor(tof_range=(1000, 20000), n_bins=1000),
    EMReconstructor(tof_range=(1000, 20000), n_bins=1000),
]

# Run benchmark
benchmark = Benchmark(reconstructors, STANDARD_SCENARIOS)
results_df = benchmark.run(verbose=True)

# Print summary
benchmark.print_summary()

# Plot comparisons
benchmark.plot_comparison(metric='chi2_per_dof')
benchmark.plot_comparison(metric='computation_time')
benchmark.plot_time_accuracy_tradeoff()

# Save results
benchmark.save_report('benchmark_results.csv')
```

## Time-to-Target Analysis

```python
def time_to_target_analysis(
    reconstructors,
    event_stream,
    true_spectrum,
    target_chi2=100,
    update_interval=1000
):
    """
    Analyze how quickly each reconstructor reaches target accuracy.

    Parameters
    ----------
    reconstructors : List[BaseReconstructor]
        Reconstructors with online update capability
    event_stream : EventDataset
        Stream of events to process incrementally
    true_spectrum : np.ndarray
        Ground truth
    target_chi2 : float
        Target chi-squared threshold
    update_interval : int
        Number of events between updates

    Returns
    -------
    Dict[str, Dict]
        Time-to-target results for each reconstructor
    """
    results = {}

    for recon in reconstructors:
        recon.reset()

        chi2_history = []
        time_history = []
        n_events_history = []

        # Process events in batches
        events_list = event_stream.events
        n_batches = len(events_list) // update_interval

        for i in range(n_batches):
            batch = events_list[i*update_interval : (i+1)*update_interval]

            start = time.time()
            recon.update(batch)
            elapsed = time.time() - start

            # Calculate current chi2
            current_result = recon.get_current_result()
            chi2 = np.sum(
                ((current_result.spectrum - true_spectrum) / current_result.uncertainty) ** 2
            )

            chi2_history.append(chi2)
            time_history.append(elapsed)
            n_events_history.append((i+1) * update_interval)

            # Check if target reached
            if chi2 < target_chi2 and 'time_to_target' not in results.get(recon.__class__.__name__, {}):
                results[recon.__class__.__name__] = {
                    'time_to_target': sum(time_history),
                    'events_to_target': n_events_history[-1],
                    'final_chi2': chi2
                }

        results[recon.__class__.__name__].update({
            'chi2_history': chi2_history,
            'time_history': time_history,
            'n_events_history': n_events_history
        })

    return results


def plot_convergence(time_to_target_results, figsize=(12, 5)):
    """Plot convergence curves for all reconstructors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for recon_name, data in time_to_target_results.items():
        # Plot vs events
        ax1.plot(
            data['n_events_history'],
            data['chi2_history'],
            label=recon_name,
            linewidth=2
        )

        # Plot vs time
        cumulative_time = np.cumsum(data['time_history'])
        ax2.plot(
            cumulative_time,
            data['chi2_history'],
            label=recon_name,
            linewidth=2
        )

    ax1.set_xlabel('Number of Events')
    ax1.set_ylabel('χ²')
    ax1.set_title('Convergence vs Events')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Computation Time (s)')
    ax2.set_ylabel('χ²')
    ax2.set_title('Convergence vs Time')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

## Kernel Efficiency Analysis

```python
def analyze_kernel_efficiency(material='iron', n_events=50000):
    """
    Compare different kernels for a given material.

    Returns
    -------
    pd.DataFrame
        Comparison of different kernels
    """
    kernels = {
        'Separated': [0],
        '2-frame (40ms)': [0, 40],
        '2-frame (25ms)': [0, 25],
        '2-frame (12ms)': [0, 12],
        '3-frame uniform': [0, 16, 16, 16],
        '4-frame fibonacci': [0, 5, 8, 13, 21],
        'High flux': [0, 5, 5, 5, 5, 5],
    }

    results = []

    for kernel_name, kernel_values in kernels.items():
        # Generate events
        events = generate_synthetic_events(
            material=material,
            kernel=np.array(kernel_values),
            n_events=n_events
        )

        # Reconstruct with EM
        recon = EMReconstructor(tof_range=(1000, 20000), n_bins=1000)
        result = recon.reconstruct(events, max_iterations=50)

        # Calculate flux efficiency
        n_frames = len(kernel_values)
        total_period = sum(kernel_values) if len(kernel_values) > 1 else 50
        flux_efficiency = n_frames / (total_period / 50)  # relative to 50 ms period

        results.append({
            'Kernel': kernel_name,
            'N Frames': n_frames,
            'Total Period (ms)': total_period,
            'Flux Efficiency': flux_efficiency,
            'χ²/dof': result.chi2 / len(result.spectrum),
            'Computation Time (s)': result.computation_time,
            'Iterations': result.iterations
        })

    return pd.DataFrame(results)
```

## Success Criteria Checklist

### Performance Targets

- [ ] **Wiener reconstruction**: 2x better χ² than baseline for 2-frame overlap
- [ ] **EM reconstruction**: 3x better χ² than baseline, < 5x computation time
- [ ] **Adaptive kernel**: 2x faster time-to-target than best fixed kernel
- [ ] **Scalability**: Linear scaling with n_events up to 1M events
- [ ] **Memory**: < 1 GB for 100k events, 1000 bins

### Validation Targets

- [ ] Match existing `Reconstruct` class within 5% for identical inputs
- [ ] Synthetic data: χ²/dof < 1.5 for well-conditioned problems
- [ ] Real data: Visual agreement with traditional analysis
- [ ] Edge detection: < 2 bins error for Bragg edge positions
- [ ] Resonance fitting: < 5% error in peak positions

## Report Template

```markdown
# Reconstruction Algorithm Benchmark Report

**Date**: YYYY-MM-DD
**Version**: X.Y.Z
**Test Environment**: [CPU, RAM, Python version]

## Executive Summary

- **Best Overall**: [Algorithm name] (χ²/dof = X.XX)
- **Fastest**: [Algorithm name] (X.XX seconds)
- **Best for High Overlap**: [Algorithm name]
- **Best for Low Statistics**: [Algorithm name]

## Detailed Results

### Scenario: Iron, 2-Frame Overlap (25 ms)

| Reconstructor | χ²/dof | RMSE | Time (s) | Iterations |
|---------------|--------|------|----------|------------|
| Baseline      | X.XX   | X.XX | X.XXX    | 1          |
| Wiener        | X.XX   | X.XX | X.XXX    | 1          |
| EM            | X.XX   | X.XX | X.XXX    | XX         |

[Include plots]

### Time-to-Target Analysis

- **Target**: χ² < 100
- **Baseline**: XXX events, X.X seconds
- **Wiener**: XXX events, X.X seconds (XXx speedup)
- **EM**: XXX events, X.X seconds (XXx speedup)

## Recommendations

1. **For production use**: [Algorithm] provides best balance of accuracy and speed
2. **For high-overlap scenarios**: Use [Algorithm]
3. **For real-time applications**: [Algorithm] meets latency requirements
4. **For offline analysis**: [Algorithm] provides highest accuracy

## Known Limitations

- [List any failure modes or scenarios where algorithms struggle]
- [Performance bottlenecks identified]
- [Edge cases to avoid]
```

---

This benchmarking framework will allow rigorous comparison of all reconstruction approaches and guide development priorities based on quantitative performance metrics.
