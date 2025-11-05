"""Test wavelength to time conversion for plotting."""

import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib.pyplot as plt

from frame_overlap import Data, Reconstruct, Analysis

# Quick setup
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)
data.convolute_response(200, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25])

recon = Reconstruct(data, tmin=3.7, tmax=11.0)
recon.filter(kind='wiener', noise_power=0.2)

print("Getting reconstruction plot data...")
# Get reconstruction data
min_len = min(len(recon.reference_data), len(recon.reconstructed_data))
recon_time_ms = recon.reference_data['time'].values[:min_len] / 1000
recon_transmission = recon.reconstructed_data['counts'].values[:min_len] / np.maximum(
    recon.data.op_overlapped_data['counts'].values[:min_len], 1)

print(f"Reconstruction time range: [{recon_time_ms.min():.2f}, {recon_time_ms.max():.2f}] ms")
print(f"Reconstruction transmission range: [{recon_transmission.min():.4f}, {recon_transmission.max():.4f}]")

# Run nbragg analysis
print("\nRunning nbragg analysis...")
analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
result = analysis.fit(recon)

# Extract nbragg data
wavelength = analysis.data.table['wavelength'].values
best_fit = result.best_fit

print(f"\nNbragg wavelength range: [{wavelength.min():.6f}, {wavelength.max():.6f}] Å")
print(f"Nbragg best_fit range: [{best_fit.min():.4f}, {best_fit.max():.4f}]")

# Convert wavelength to time
L = 9.0  # meters
time_us = wavelength * L * 252.778
time_ms = time_us / 1000

print(f"\nConverted time range: [{time_ms.min():.2f}, {time_ms.max():.2f}] ms")

# Check if ranges overlap
print(f"\nDo ranges overlap? {time_ms.min() <= recon_time_ms.max() and time_ms.max() >= recon_time_ms.min()}")

# Plot to visualize
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(recon_time_ms, recon_transmission, 'b-', label='Reconstructed', alpha=0.7)
ax.plot(time_ms, best_fit, 'g--', label='nbragg fit', linewidth=2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Transmission')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/nbragg_fit_overlay.png', dpi=100)
print("\nPlot saved to /tmp/nbragg_fit_overlay.png")
plt.close()

print("\n✅ Test completed successfully!")
