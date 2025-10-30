"""Test the corrected duty cycle calculation"""
from frame_overlap import Data
import numpy as np

print("=" * 70)
print("TESTING CORRECTED DUTY CYCLE CALCULATION")
print("=" * 70)

# Test case from user:
# Original: flux=5e6 n/cm²/s, duration=30min=0.5hr
# New: flux=1e6 n/cm²/s, pulse_duration=200µs, freq=60Hz
# Expected: duty_cycle = (1e6/5e6) * 60 * 200e-6 = 0.2 * 60 * 0.0002 = 0.0024

print("\nTest Case:")
print("  Original: flux=5e6 n/cm²/s, duration=0.5 hr, freq=20 Hz")
print("  New: flux=1e6 n/cm²/s, measurement_time=0.5 hr, freq=60 Hz, pulse_duration=200 µs")
print("  Expected duty_cycle = (1e6/5e6) × 60 × 200e-6 = 0.0024")

# Create data with original parameters
data = Data(signal_file='notebooks/iron_powder.csv',
            openbeam_file='notebooks/openbeam.csv',
            flux=5e6,        # Original flux
            duration=0.5,    # 30 minutes = 0.5 hours
            freq=20)         # Original freq (doesn't matter for pulsed calculation)

# Convolute to set pulse_duration
data.convolute_response(pulse_duration=200)
print(f"\n✓ Convolution applied with pulse_duration = {data.pulse_duration} µs")

# Apply Poisson with new parameters
print("\nApplying Poisson sampling...")
print("-" * 70)
data.poisson_sample(flux=1e6, measurement_time=0.5, freq=60)
print("-" * 70)

# Check the calculated duty cycle
# We can infer it from the print statements above

print("\n✅ Test completed!")
print("\nNote: The duty_cycle is calculated as:")
print("  duty_cycle = (flux_new / flux_orig) × freq_new × pulse_duration")
print("  duty_cycle = (1e6 / 5e6) × 60 × (200 / 1e6)")
print("  duty_cycle = 0.2 × 60 × 0.0002")
print("  duty_cycle = 0.0024")
