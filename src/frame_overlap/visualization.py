import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_analysis(t_signal_df, signal_df, scaled_df, kernel_df, observed_df, reconstructed_df, residuals_df, chi2_per_dof):
    """
    Plot the analysis results including original signal, scaled signal, kernel, observed signal,
    reconstructed signal, and residuals.

    Parameters
    ----------
    t_signal_df : pandas.DataFrame
        DataFrame with column 'time' containing the time array.
    signal_df : pandas.DataFrame
        DataFrame with column 'counts' containing the original signal.
    scaled_df : pandas.DataFrame
        DataFrame with column 'counts' containing the scaled signal.
    kernel_df : pandas.DataFrame
        DataFrame with columns 'kernel_time' and 'kernel_value' containing the kernel.
    observed_df : pandas.DataFrame
        DataFrame with column 'counts' containing the observed signal.
    reconstructed_df : pandas.DataFrame
        DataFrame with column 'reconstructed' containing the reconstructed signal.
    residuals_df : pandas.DataFrame
        DataFrame with column 'residuals' containing the residuals.
    chi2_per_dof : float
        Reduced chi-squared value for the fit.

    Returns
    -------
    None
        Displays a matplotlib plot with six subplots.

    Raises
    ------
    ValueError
        If input DataFrames lack required columns or have inconsistent lengths.
    """
    required_columns = {
        't_signal_df': 'time',
        'signal_df': 'counts',
        'scaled_df': 'counts',
        'kernel_df': ['kernel_time', 'kernel_value'],
        'observed_df': 'counts',
        'reconstructed_df': 'reconstructed',
        'residuals_df': 'residuals'
    }
    
    for df_name, cols in required_columns.items():
        df = locals()[df_name]
        if isinstance(cols, list):
            if not all(col in df.columns for col in cols):
                raise ValueError(f"{df_name} must have columns: {cols}")
        else:
            if cols not in df.columns:
                raise ValueError(f"{df_name} must have column: {cols}")
    
    signal_length = len(t_signal_df)
    if not all(len(df) == signal_length for df in [signal_df, scaled_df, observed_df, reconstructed_df, residuals_df]):
        raise ValueError("All signal-related DataFrames must have the same length")
    if len(kernel_df) > signal_length:
        raise ValueError("Kernel DataFrame length must not exceed signal length")
    
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    
    # Plot original signal
    axes[0].plot(t_signal_df['time'], signal_df['counts'], label='Original Signal')
    axes[0].set_ylabel('Counts')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot scaled signal
    axes[1].plot(t_signal_df['time'], scaled_df['counts'], label='Scaled Signal', color='orange')
    axes[1].set_ylabel('Counts')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot kernel
    axes[2].plot(kernel_df['kernel_time'], kernel_df['kernel_value'], label='Kernel', color='green')
    axes[2].set_ylabel('Kernel Value')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot observed signal
    axes[3].plot(t_signal_df['time'], observed_df['counts'], label='Observed Signal', color='red')
    axes[3].set_ylabel('Counts')
    axes[3].legend()
    axes[3].grid(True)
    
    # Plot reconstructed signal
    axes[4].plot(t_signal_df['time'], reconstructed_df['reconstructed'], label='Reconstructed Signal', color='purple')
    axes[4].set_ylabel('Counts')
    axes[4].legend()
    axes[4].grid(True)
    
    # Plot residuals
    axes[5].plot(t_signal_df['time'], residuals_df['residuals'], label=f'Residuals (χ²/dof = {chi2_per_dof:.2f})', color='black')
    axes[5].set_xlabel('Time')
    axes[5].set_ylabel('Residuals')
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()