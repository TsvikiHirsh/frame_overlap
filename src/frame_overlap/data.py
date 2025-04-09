import numpy as np
import pandas as pd

def read_tof_data(file_path="tof_data.csv", threshold=None):
    """Read ToF data from CSV and apply threshold if specified."""
    df = pd.read_csv(file_path)
    if threshold is not None:
        df = df.loc[df['stack'] >= threshold]
    t_signal = (df['stack'] - 1) * 10  # Time in µs
    signal = df['counts'].values
    errors = df['err'].values
    stacks = df['stack'].values
    return t_signal, signal, errors, stacks

def prepare_full_frame(t_signal, signal, errors, stacks, max_stack=2400):
    """Prepare a full frame from 1 to max_stack with zeros for missing bins."""
    all_stacks = np.arange(1, max_stack + 1)
    full_signal = np.zeros(len(all_stacks))
    full_errors = np.zeros(len(all_stacks))
    stack_indices = [int(s - 1) for s in stacks]
    full_signal[stack_indices] = signal
    full_errors[stack_indices] = errors
    return all_stacks, full_signal, full_errors