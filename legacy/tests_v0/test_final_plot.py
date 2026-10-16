"""Test the final plotting with nbragg fit overlay."""

import sys
sys.path.insert(0, 'src')
import numpy as np
import matplotlib.pyplot as plt

from frame_overlap import Data, Reconstruct, Analysis

# Setup
data = Data('notebooks/iron_powder.csv', 'notebooks/openbeam.csv',
           flux=5e6, duration=0.5, freq=20)
data.convolute_response(200, bin_width=10)
data.poisson_sample(flux=1e6, freq=20, measurement_time=8*60, seed=42)
data.overlap(kernel=[0, 25])

recon = Reconstruct(data, tmin=3.7, tmax=11.0)
recon.filter(kind='wiener', noise_power=0.2)

# Run nbragg analysis
analysis = Analysis(xs='iron', vary_background=True, vary_response=True)
result = analysis.fit(recon)

print("="*60)
print("Testing nbragg fit overlay on reconstruction plot")
print("="*60)

# Generate reconstruction plot (mimicking streamlit code)
print("\n1. Creating reconstruction plot...")
mpl_fig = recon.plot(kind='transmission', show_errors=False, figsize=(12, 8), ylim=(0, 1))
print("   ✓ Reconstruction plot created")

# Add nbragg fit curve (mimicking streamlit code)
print("\n2. Adding nbragg fit curve...")
axes = mpl_fig.get_axes()
ax_data = axes[0]

# Get wavelength and best_fit
wavelength = result.userkws['wl']  # in Angstroms
best_fit_transmission = result.best_fit

print(f"   - wavelength shape: {wavelength.shape}")
print(f"   - best_fit shape: {best_fit_transmission.shape}")
print(f"   - wavelength range: [{wavelength.min():.4f}, {wavelength.max():.4f}] Å")

# Convert wavelength to time in ms
L = 9.0  # meters
time_us = wavelength * L * 252.778  # time in microseconds
time_ms = time_us / 1000  # time in milliseconds

print(f"   - time_ms range: [{time_ms.min():.2f}, {time_ms.max():.2f}] ms")

# Plot nbragg fit on the same axes
ax_data.plot(time_ms, best_fit_transmission,
           label='nbragg fit', color='green', linewidth=2, linestyle='--')
ax_data.legend()

print("   ✓ nbragg fit added to plot")

# Save the figure
output_path = '/tmp/reconstruction_with_nbragg_fit.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n3. Plot saved to {output_path}")

# Display plot info
print(f"\nPlot contains {len(ax_data.get_lines())} lines:")
for i, line in enumerate(ax_data.get_lines()):
    label = line.get_label()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    print(f"   Line {i}: '{label}' - {len(xdata)} points")

plt.close()

print("\n" + "="*60)
print("✅ Test completed successfully!")
print("="*60)
print("\nThe plot should show:")
print("  - Blue line: Original (convolved)")
print("  - Orange line: Reconstructed")
print("  - Green dashed line: nbragg fit")
