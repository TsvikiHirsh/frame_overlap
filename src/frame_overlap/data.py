import pandas as pd
import numpy as np

def read_tof_data(file_path, threshold=0):
    """
    Read neutron Time-of-Flight (ToF) data from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing ToF data with columns 'stack', 'counts', 'err'.
    threshold : float, optional
        Minimum counts threshold to filter data (default: 0).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'time', 'counts', 'errors', 'stack' containing filtered ToF data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If threshold is negative or CSV file lacks required columns.
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found")
    
    required_columns = ['stack', 'counts', 'err']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    # Filter by threshold
    df = df[df['counts'] >= threshold].copy()
    
    # Calculate time (assuming stack numbers are 1-based and correspond to 10-unit time intervals)
    df['time'] = (df['stack'] - 1) * 10
    
    # Rename 'err' to 'errors' for consistency
    df = df.rename(columns={'err': 'errors'})
    
    # Ensure correct column order
    return df[['time', 'counts', 'errors', 'stack']]

def prepare_full_frame(signal_df, max_stack):
    """
    Prepare a full frame DataFrame with all stack numbers up to max_stack.

    Parameters
    ----------
    signal_df : pandas.DataFrame
        DataFrame with columns 'time', 'counts', 'errors', 'stack'.
    max_stack : int
        Maximum stack number to include in the full frame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'stack', 'counts', 'errors' containing all stacks from 1 to max_stack.

    Raises
    ------
    ValueError
        If max_stack is not positive or signal_df lacks required columns.
    """
    if max_stack <= 0:
        raise ValueError("max_stack must be positive")
    
    required_columns = ['time', 'counts', 'errors', 'stack']
    if not all(col in signal_df.columns for col in required_columns):
        raise ValueError(f"signal_df must contain columns: {required_columns}")
    
    # Create full stack range
    all_stacks = np.arange(1, max_stack + 1)
    
    # Initialize full DataFrame
    full_df = pd.DataFrame({
        'stack': all_stacks,
        'counts': 0.0,
        'errors': 0.0
    })
    
    # Merge signal data into full frame
    full_df = full_df.set_index('stack').combine_first(
        signal_df[['stack', 'counts', 'errors']].set_index('stack')
    ).reset_index()
    
    # Fill NaN values with 0
    full_df = full_df.fillna({'counts': 0.0, 'errors': 0.0})
    
    return full_df[['stack', 'counts', 'errors']]