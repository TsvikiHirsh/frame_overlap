import numpy as np
import pandas as pd

def read_tof_data(file_path="tof_data.csv", threshold=None):
    """
    Read Time-of-Flight (ToF) data from a CSV file and apply an optional threshold.

    Parameters
    ----------
    file_path : str, optional
        Path to the CSV file containing ToF data (default: "tof_data.csv").
    threshold : float, optional
        Minimum value for 'stack' column to filter data (default: None).

    Returns
    -------
    tuple
        A tuple containing:
        - t_signal : array-like, time array in microseconds
        - signal : array-like, signal counts
        - errors : array-like, uncertainties in counts
        - stacks : array-like, stack numbers

    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    KeyError
        If required columns ('stack', 'counts', 'err') are missing in the CSV.
    ValueError
        If threshold is negative or data contains invalid values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'stack': [1, 2, 3], 'counts': [100, 200, 300], 'err': [10, 20, 30]})
    >>> data.to_csv('test.csv', index=False)
    >>> t, s, e, st = read_tof_data('test.csv', threshold=2)
    >>> len(t)
    2
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found")
    
    required_columns = ['stack', 'counts', 'err']
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"CSV must contain columns: {required_columns}")
    
    if threshold is not None:
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        df = df.loc[df['stack'] >= threshold]
    
    t_signal = (df['stack'] - 1) * 10  # Convert stack to time in µs
    signal = df['counts'].values
    errors = df['err'].values
    stacks = df['stack'].values
    
    if np.any(errors <= 0):
        raise ValueError("Errors must be positive")
    if np.any(np.isnan(signal)) or np.any(np.isnan(errors)) or np.any(np.isnan(stacks)):
        raise ValueError("Data must not contain NaN values")
    
    return t_signal, signal, errors, stacks

def prepare_full_frame(t_signal, signal, errors, stacks, max_stack=2400):
    """
    Prepare a full frame from 1 to max_stack, filling missing bins with zeros.

    Parameters
    ----------
    t_signal : array-like
        Time array in microseconds.
    signal : array-like
        Signal counts.
    errors : array-like
        Uncertainties in counts.
    stacks : array-like
        Stack numbers.
    max_stack : int, optional
        Maximum stack number (default: 2400).

    Returns
    -------
    tuple
        A tuple containing:
        - all_stacks : array-like, complete stack numbers from 1 to max_stack
        - full_signal : array-like, signal with zeros for missing bins
        - full_errors : array-like, errors with zeros for missing bins

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes or invalid values.
    TypeError
        If inputs are not array-like or contain non-numeric values.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([10, 20])
    >>> s = np.array([100, 200])
    >>> e = np.array([10, 20])
    >>> st = np.array([1, 2])
    >>> stacks, sig, err = prepare_full_frame(t, s, e, st, max_stack=3)
    >>> len(stacks)
    3
    >>> sig[2]
    0.0
    """
    t_signal = np.asarray(t_signal, dtype=float)
    signal = np.asarray(signal, dtype=float)
    errors = np.asarray(errors, dtype=float)
    stacks = np.asarray(stacks, dtype=float)
    
    if not (len(t_signal) == len(signal) == len(errors) == len(stacks)):
        raise ValueError("All input arrays must have the same length")
    if max_stack < 1:
        raise ValueError("max_stack must be positive")
    if np.any(errors < 0) or np.any(stacks < 1) or np.any(stacks > max_stack):
        raise ValueError("Errors must be non-negative, stacks must be between 1 and max_stack")
    if np.any(np.isnan(t_signal)) or np.any(np.isnan(signal)) or np.any(np.isnan(errors)) or np.any(np.isnan(stacks)):
        raise ValueError("Input arrays must not contain NaN values")

    all_stacks = np.arange(1, max_stack + 1)
    full_signal = np.zeros(len(all_stacks))
    full_errors = np.zeros(len(all_stacks))
    stack_indices = [int(s - 1) for s in stacks]
    
    full_signal[stack_indices] = signal
    full_errors[stack_indices] = errors
    return all_stacks, full_signal, full_errors