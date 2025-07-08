import pandas as pd
import matplotlib.pyplot as plt

def plot_analysis(t_signal_df, signal_df, scaled_df, kernel_df, observed_df, reconstructed_df, residuals_df, chi2_per_dof):
    """
    Plot the analysis results including original signal, kernel, observed, reconstructed, and residuals.

    Parameters
    ----------
    t_signal_df : pandas.DataFrame
        DataFrame with column 'time' for the x-axis.
    signal_df : pandas.DataFrame
        DataFrame with column 'counts' for the original signal.
    scaled_df : pandas.DataFrame
        DataFrame with column 'counts' for the scaled original signal.
    kernel_df : pandas.DataFrame
        DataFrame with columns 'kernel_time' and 'kernel_value' for the kernel.
    observed_df : pandas.DataFrame
        DataFrame with column 'observed' for the observed signal with noise.
    reconstructed_df : pandas.DataFrame
        DataFrame with column 'reconstructed' for the reconstructed signal.
    residuals_df : pandas.DataFrame
        DataFrame with column 'residuals' for the residuals (scaled - reconstructed).
    chi2_per_dof : float
        Reduced chi-squared statistic for the fit.

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
        'observed_df': 'observed',
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

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Plot original and scaled signal
    signal_df.plot(x=t_signal_df['time'], y='counts', ax=ax1, label='Original Signal', color='blue')
    scaled_df.plot(x=t_signal_df['time'], y='counts', ax=ax1, label='Scaled Signal', color='green')
    ax1.set_ylabel('Counts')
    ax1.legend()
    ax1.set_title('Original and Scaled Signal')
    
    # Plot kernel
    kernel_df.plot(x='kernel_time', y='kernel_value', ax=ax2, label='Kernel', color='red')
    ax2.set_ylabel('Kernel Value')
    ax2.legend()
    ax2.set_title('Kernel')
    
    # Plot observed and reconstructed signals
    observed_df.plot(x=t_signal_df['time'], y='observed', ax=ax3, label='Observed (Poisson)', color='orange')
    reconstructed_df.plot(x=t_signal_df['time'], y='reconstructed', ax=ax3, label='Reconstructed', color='purple')
    ax3.set_ylabel('Counts')
    ax3.legend()
    ax3.set_title(f'Reconstructed Signal (Reduced Chi2: {chi2_per_dof:.2f})')
    
    # Plot residuals
    residuals_df.plot(x=t_signal_df['time'], y='residuals', ax=ax4, label='Residuals', color='black')
    ax4.set_ylabel('Residuals')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.set_title('Residuals')
    
    plt.tight_layout()
    plt.show()