"""
Frame Overlap Interactive Explorer - Streamlit App

Explore neutron Time-of-Flight frame overlap analysis interactively!
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import io

# Import frame_overlap
import sys
sys.path.insert(0, 'src')
from frame_overlap import Data, Reconstruct, Workflow

# Use Agg backend for matplotlib (non-interactive, for conversion to images)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mpl_to_plotly(fig):
    """
    Convert matplotlib figure to plotly figure for interactivity.
    Extracts data from matplotlib axes and recreates in plotly.
    """
    # Get the main axis (or first axis if multiple)
    axes = fig.get_axes()

    # Create plotly figure
    if len(axes) == 1:
        plotly_fig = go.Figure()
        ax = axes[0]

        # Extract lines from matplotlib
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            plotly_fig.add_trace(go.Scatter(
                x=xdata, y=ydata,
                mode='lines',
                name=line.get_label(),
                line=dict(color=matplotlib.colors.rgb2hex(line.get_color())),
                showlegend=(line.get_label() and not line.get_label().startswith('_'))
            ))

        # Set layout and preserve axis limits
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        plotly_fig.update_layout(
            xaxis_title=ax.get_xlabel(),
            yaxis_title=ax.get_ylabel(),
            title=ax.get_title(),
            hovermode='x unified',
            template='plotly_white',
            xaxis=dict(range=xlim),
            yaxis=dict(range=ylim)
        )

    elif len(axes) == 2:
        # Two subplots (data + residuals)
        plotly_fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )

        # Top plot (data)
        ax_top = axes[0]
        for line in ax_top.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            plotly_fig.add_trace(go.Scatter(
                x=xdata, y=ydata,
                mode='lines',
                name=line.get_label(),
                line=dict(color=matplotlib.colors.rgb2hex(line.get_color())),
                showlegend=(line.get_label() and not line.get_label().startswith('_'))
            ), row=1, col=1)

        # Bottom plot (residuals)
        ax_bottom = axes[1]
        for line in ax_bottom.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            plotly_fig.add_trace(go.Scatter(
                x=xdata, y=ydata,
                mode='lines',
                name=line.get_label(),
                line=dict(color=matplotlib.colors.rgb2hex(line.get_color())),
                showlegend=False
            ), row=2, col=1)

        # Set layout and preserve axis limits
        top_ylim = ax_top.get_ylim()
        bottom_ylim = ax_bottom.get_ylim()
        xlim = ax_top.get_xlim()

        plotly_fig.update_xaxes(title_text=ax_bottom.get_xlabel(), range=xlim, row=2, col=1)
        plotly_fig.update_xaxes(range=xlim, row=1, col=1)
        plotly_fig.update_yaxes(title_text=ax_top.get_ylabel(), range=top_ylim, row=1, col=1)
        plotly_fig.update_yaxes(title_text=ax_bottom.get_ylabel(), range=bottom_ylim, row=2, col=1)
        plotly_fig.update_layout(
            title=ax_top.get_title(),
            hovermode='x unified',
            template='plotly_white',
            height=600
        )

    else:
        # Fallback for other cases
        plotly_fig = go.Figure()

    return plotly_fig

# Page config
st.set_page_config(
    page_title="Frame Overlap Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.stPlotly {
    background-color: white;
}
h1 {
    color: #1f77b4;
}
h2 {
    color: #ff7f0e;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔬 Frame Overlap Interactive Explorer")
st.markdown("Explore neutron Time-of-Flight frame overlap analysis step by step!")

# Cache data loading
@st.cache_data
def load_data():
    """Load the iron powder and openbeam data."""
    signal_path = 'notebooks/iron_powder.csv'
    openbeam_path = 'notebooks/openbeam.csv'

    # Check if files exist
    if not Path(signal_path).exists() or not Path(openbeam_path).exists():
        st.error("Data files not found! Please ensure iron_powder.csv and openbeam.csv are in the notebooks/ directory.")
        return None

    return signal_path, openbeam_path

# Load data
data_paths = load_data()
if data_paths is None:
    st.stop()

signal_path, openbeam_path = data_paths

# Sidebar
st.sidebar.header("⚙️ Processing Pipeline")
st.sidebar.markdown("Configure each stage of the analysis pipeline:")

# Initialize session state for workflow
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = None
if 'recon' not in st.session_state:
    st.session_state.recon = None

# Stage 1: Data Loading
with st.sidebar.expander("📁 1. Data Loading", expanded=False):
    st.markdown("**Original Measurement Parameters**")
    flux_orig = st.number_input(
        "Original Flux (n/cm²/s)",
        min_value=1e5,
        max_value=1e7,
        value=5e6,
        step=1e5,
        format="%.2e",
        help="Original neutron flux in n/cm²/s"
    )

    duration_orig = st.slider(
        "Original Duration (hours)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Original measurement duration in hours"
    )

    freq_orig = st.slider(
        "Original Frequency (Hz)",
        min_value=10,
        max_value=100,
        value=20,
        step=5,
        help="Original pulse frequency in Hz"
    )

# Stage 2: Convolution
with st.sidebar.expander("🔊 2. Instrument Response", expanded=False):
    apply_convolution = st.checkbox("Apply Convolution", value=True)

    if apply_convolution:
        pulse_duration = st.slider(
            "Pulse Duration (µs)",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            help="Instrument pulse duration in microseconds"
        )

        bin_width = st.slider(
            "Bin Width (µs)",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Time bin width"
        )
    else:
        pulse_duration = None
        bin_width = 10

# Stage 3: Poisson Sampling
with st.sidebar.expander("🎲 3. Poisson Sampling", expanded=False):
    apply_poisson = st.checkbox("Apply Poisson", value=True)

    if apply_poisson:
        st.markdown("**New Measurement Conditions**")
        flux_new = st.number_input(
            "New Flux (n/cm²/s)",
            min_value=1e5,
            max_value=5e6,
            value=1e6,
            step=1e5,
            format="%.2e",
            help="Scaled flux condition"
        )

        freq_new = st.slider(
            "New Frequency (Hz)",
            min_value=10,
            max_value=100,
            value=60,
            step=5,
            help="New pulse frequency"
        )

        measurement_time = st.slider(
            "Measurement Time (min)",
            min_value=1,
            max_value=60,
            value=30,
            step=1,
            help="New measurement duration"
        )

        seed_poisson = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )
    else:
        flux_new = None
        freq_new = None
        measurement_time = None
        seed_poisson = None

# Stage 4: Frame Overlap
with st.sidebar.expander("🔄 4. Frame Overlap", expanded=False):
    apply_overlap = st.checkbox("Apply Overlap", value=True)

    if apply_overlap:
        n_frames = st.slider(
            "Number of Frames",
            min_value=2,
            max_value=4,
            value=2,
            help="Number of overlapping frames"
        )

        if n_frames == 2:
            frame_spacing = st.slider(
                "Frame Spacing (ms)",
                min_value=10,
                max_value=40,
                value=25,
                step=1,
                help="Time between frame starts"
            )
            kernel = [0, frame_spacing]
            total_time = frame_spacing * 2
        elif n_frames == 3:
            spacing_1 = st.slider("Frame 1→2 (ms)", 10, 30, 15, 1)
            spacing_2 = st.slider("Frame 2→3 (ms)", 10, 30, 15, 1)
            kernel = [0, spacing_1, spacing_1 + spacing_2]
            total_time = spacing_1 + spacing_2 + 20
        else:  # 4 frames
            spacing = st.slider("Frame Spacing (ms)", 8, 20, 12, 1)
            kernel = [0, spacing, spacing*2, spacing*3]
            total_time = spacing * 4

        st.info(f"Kernel: {kernel} ms")
    else:
        kernel = None
        total_time = None

# Stage 5: Reconstruction
with st.sidebar.expander("🔧 5. Reconstruction", expanded=False):
    apply_reconstruction = st.checkbox("Apply Reconstruction", value=True)

    if apply_reconstruction:
        recon_method = st.selectbox(
            "Method",
            ["wiener", "lucy", "tikhonov"],
            help="Deconvolution algorithm"
        )

        if recon_method in ["wiener", "tikhonov"]:
            noise_power = st.slider(
                "Noise Power",
                min_value=0.001,
                max_value=0.2,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Regularization parameter"
            )
            recon_params = {"noise_power": noise_power}
        else:  # lucy
            iterations = st.slider(
                "Iterations",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of iterations"
            )
            recon_params = {"iterations": iterations}

        # Time range filtering
        use_time_filter = st.checkbox("Filter Time Range")
        if use_time_filter:
            tmin = st.number_input("Min Time (ms)", 0.0, 50.0, 10.0, 0.5)
            tmax = st.number_input("Max Time (ms)", 0.0, 50.0, 40.0, 0.5)
        else:
            tmin, tmax = None, None
    else:
        apply_reconstruction = False
        recon_method = None
        recon_params = {}
        tmin, tmax = None, None

# Process button
process_button = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# Main content area
if process_button:
    with st.spinner("Processing pipeline..."):
        try:
            # Create workflow
            data = Data(signal_path, openbeam_path,
                       flux=flux_orig, duration=duration_orig, freq=freq_orig)

            # Apply stages
            if apply_convolution:
                data.convolute_response(pulse_duration, bin_width=bin_width)
                st.sidebar.success(f"✓ Convolved (pulse: {pulse_duration} µs)")

            if apply_poisson:
                data.poisson_sample(flux=flux_new, freq=freq_new,
                                   measurement_time=measurement_time, seed=seed_poisson)
                st.sidebar.success(f"✓ Poisson (flux: {flux_new:.1e})")

            if apply_overlap:
                data.overlap(kernel=kernel, total_time=total_time)
                st.sidebar.success(f"✓ Overlap ({n_frames} frames)")

            st.session_state.workflow_data = data

            # Reconstruction
            if apply_reconstruction and apply_overlap:
                recon = Reconstruct(data, tmin=tmin, tmax=tmax)
                recon.filter(kind=recon_method, **recon_params)
                st.session_state.recon = recon

                stats = recon.get_statistics()
                st.sidebar.success(f"✓ Reconstructed (χ²/dof: {stats['chi2_per_dof']:.1f})")

            st.sidebar.success("✅ Pipeline complete!")

        except Exception as e:
            st.error(f"Error processing pipeline: {e}")
            st.exception(e)

# Display results
if st.session_state.workflow_data is not None:
    data = st.session_state.workflow_data

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signal / Transmission", "🔍 Reconstruction", "📉 Statistics", "🔁 GroupBy"])

    with tab1:
        st.header("Signal / Transmission")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Controls row 1: Plot type and stages
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

            with ctrl_col1:
                plot_kind = st.radio(
                    "View",
                    ["signal", "transmission"],
                    format_func=lambda x: "Signal" if x == "signal" else "Transmission",
                    horizontal=True
                )

            with ctrl_col2:
                show_stages = st.checkbox("Show All Stages", value=True)

            with ctrl_col3:
                show_errors = st.checkbox("Show Error Bars", value=False)

            with ctrl_col4:
                log_scale = st.checkbox("Log Scale (Y)", value=False)

            # Generate plot
            if plot_kind == "transmission" and data.op_data is None:
                st.warning("No openbeam data available for transmission calculation.")
            else:
                # Use Data.plot() method
                if plot_kind == 'transmission':
                    mpl_fig = data.plot(kind='transmission', show_stages=show_stages,
                                       show_errors=show_errors, figsize=(12, 6), ylim=(0, 1))
                else:
                    mpl_fig = data.plot(kind='signal', show_stages=show_stages,
                                       show_errors=show_errors, figsize=(12, 6))

                # Convert to Plotly for interactivity
                plotly_fig = mpl_to_plotly(mpl_fig)

                # Apply log scale if requested
                if log_scale:
                    plotly_fig.update_yaxes(type="log")

                st.plotly_chart(plotly_fig, use_container_width=True)
                plt.close(mpl_fig)

        with col2:
            st.markdown("**Current Stage Info**")
            if data.overlapped_data is not None:
                st.metric("Stage", "Overlapped")
                st.metric("Frames", len(kernel) if kernel else "N/A")
                st.metric("Data Points", len(data.overlapped_data))
            elif data.poissoned_data is not None:
                st.metric("Stage", "Poissoned")
                st.metric("Data Points", len(data.poissoned_data))
            elif data.convolved_data is not None:
                st.metric("Stage", "Convolved")
                st.metric("Pulse (µs)", pulse_duration)
            else:
                st.metric("Stage", "Original")
                st.metric("Data Points", len(data.data))

    with tab2:
        st.header("Reconstruction Results")

        if st.session_state.recon is not None:
            recon = st.session_state.recon

            # Controls
            ctrl_col1, ctrl_col2 = st.columns(2)

            with ctrl_col1:
                plot_type = st.radio(
                    "Plot Type",
                    ["transmission", "signal"],
                    format_func=lambda x: "Transmission" if x == "transmission" else "Signal",
                    horizontal=True
                )

            with ctrl_col2:
                recon_log_scale = st.checkbox("Log Scale (Y)", value=False, key="recon_log")

            # Use Reconstruct.plot() method with ylim for transmission
            if plot_type == 'transmission':
                mpl_fig = recon.plot(kind=plot_type, show_errors=show_errors,
                                    figsize=(12, 8), ylim=(0, 1))
            else:
                mpl_fig = recon.plot(kind=plot_type, show_errors=show_errors, figsize=(12, 8))

            # Convert to Plotly for interactivity
            plotly_fig = mpl_to_plotly(mpl_fig)

            # Apply log scale if requested (only to top plot)
            if recon_log_scale:
                plotly_fig.update_yaxes(type="log", row=1, col=1)

            st.plotly_chart(plotly_fig, use_container_width=True)
            plt.close(mpl_fig)
        else:
            st.info("Run reconstruction to see results here.")

    with tab3:
        st.header("Statistics & Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Processing Parameters")
            params_df = pd.DataFrame({
                'Parameter': ['Original Flux', 'Original Duration', 'Original Frequency',
                             'Pulse Duration', 'New Flux', 'New Frequency', 'Frames'],
                'Value': [
                    f"{flux_orig:.2e} n/cm²/s",
                    f"{duration_orig:.1f} hours",
                    f"{freq_orig} Hz",
                    f"{pulse_duration} µs" if pulse_duration else "N/A",
                    f"{flux_new:.2e} n/cm²/s" if flux_new else "N/A",
                    f"{freq_new} Hz" if freq_new else "N/A",
                    f"{len(kernel)}" if kernel else "1"
                ]
            })
            st.dataframe(params_df, hide_index=True, use_container_width=True)

        with col2:
            if st.session_state.recon is not None:
                st.subheader("Reconstruction Quality")
                stats = recon.get_statistics()

                metrics_df = pd.DataFrame({
                    'Metric': ['χ²/dof'],
                    'Value': [
                        f"{stats['chi2_per_dof']:.2f}",
                        # f"{stats['rmse']:.2e}",
                        # f"{stats['mae']:.2e}",
                        # f"{stats['max_abs_error']:.2e}",
                        # f"{stats['r_squared']:.4f}"
                    ]
                })
                st.dataframe(metrics_df, hide_index=True, use_container_width=True)

                # Quality indicator
                chi2 = stats['chi2_per_dof']
                if chi2 < 2:
                    st.success(f"✅ Excellent fit (χ²/dof = {chi2:.2f})")
                elif chi2 < 5:
                    st.info(f"ℹ️ Good fit (χ²/dof = {chi2:.2f})")
                else:
                    st.warning(f"⚠️ Poor fit (χ²/dof = {chi2:.2f})")
            else:
                st.info("No reconstruction statistics available yet.")

    with tab4:
        st.header("GroupBy - Parameter Sweep")

        if st.session_state.recon is not None:
            st.markdown("""
            Run a parameter sweep to explore how different parameter values affect reconstruction quality.
            This uses the Workflow's `groupby()` method to automatically sweep through parameter ranges.
            """)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Sweep Configuration")

                # Parameter selection
                sweep_params = {
                    'pulse_duration': 'Pulse Duration (µs)',
                    'noise_power': 'Noise Power',
                    'iterations': 'Lucy-Richardson Iterations',
                    'flux': 'Flux (n/cm²/s)',
                }

                param_to_sweep = st.selectbox(
                    "Parameter to Sweep",
                    options=list(sweep_params.keys()),
                    format_func=lambda x: sweep_params[x],
                    help="Select which parameter to vary"
                )

                # Parameter range inputs
                st.markdown("**Range Settings**")

                # Default ranges for different parameters
                default_ranges = {
                    'pulse_duration': (50, 500, 50),
                    'noise_power': (0.001, 0.1, 0.01),
                    'iterations': (5, 50, 5),
                    'flux': (1e5, 5e6, 5e5),
                }

                low_default, high_default, step_default = default_ranges.get(
                    param_to_sweep, (0.001, 0.1, 0.01)
                )

                use_num_points = st.checkbox("Use number of points instead of step", value=False)

                if use_num_points:
                    low_val = st.number_input(
                        "Start Value",
                        value=float(low_default),
                        format="%.4f" if param_to_sweep == 'noise_power' else "%.2e" if param_to_sweep == 'flux' else "%.1f"
                    )
                    high_val = st.number_input(
                        "End Value",
                        value=float(high_default),
                        format="%.4f" if param_to_sweep == 'noise_power' else "%.2e" if param_to_sweep == 'flux' else "%.1f"
                    )
                    num_points = st.slider("Number of Points", min_value=5, max_value=50, value=10)
                    step_val = None
                else:
                    low_val = st.number_input(
                        "Start Value",
                        value=float(low_default),
                        format="%.4f" if param_to_sweep == 'noise_power' else "%.2e" if param_to_sweep == 'flux' else "%.1f"
                    )
                    high_val = st.number_input(
                        "End Value",
                        value=float(high_default),
                        format="%.4f" if param_to_sweep == 'noise_power' else "%.2e" if param_to_sweep == 'flux' else "%.1f"
                    )
                    step_val = st.number_input(
                        "Step Size",
                        value=float(step_default),
                        format="%.4f" if param_to_sweep == 'noise_power' else "%.2e" if param_to_sweep == 'flux' else "%.1f"
                    )
                    num_points = None

                # Y-axis selection for plot
                st.markdown("**Plot Configuration**")
                y_param_options = {
                    'chi2': 'χ² (Chi-squared)',
                    'redchi2': 'χ²/dof (Reduced Chi-squared)',
                    'aic': 'AIC (Akaike Information Criterion)',
                    'bic': 'BIC (Bayesian Information Criterion)',
                    'param_thickness': 'Fitted Thickness',
                    'param_N0': 'Fitted N0',
                }

                y_param = st.selectbox(
                    "Y-axis Parameter",
                    options=list(y_param_options.keys()),
                    format_func=lambda x: y_param_options[x],
                    help="Select which metric to plot"
                )

                # Run sweep button
                run_sweep = st.button("🚀 Run Parameter Sweep", type="primary", use_container_width=True)

            with col2:
                st.subheader("Results")

                if run_sweep:
                    # Initialize session state for sweep results
                    if 'sweep_results' not in st.session_state:
                        st.session_state.sweep_results = None

                    with st.spinner(f"Running parameter sweep for {sweep_params[param_to_sweep]}..."):
                        try:
                            # Create a fresh workflow from the current configuration
                            wf = Workflow(signal_path, openbeam_path,
                                        flux=flux_orig, duration=duration_orig, freq=freq_orig)

                            # Apply all the stages as configured
                            # When sweeping a parameter, call the method but don't pass that parameter
                            if apply_convolution:
                                if param_to_sweep == 'pulse_duration':
                                    wf.convolute(bin_width=bin_width)  # pulse_duration from sweep
                                else:
                                    wf.convolute(pulse_duration, bin_width=bin_width)

                            if apply_poisson:
                                if param_to_sweep == 'flux':
                                    wf.poisson(freq=freq_new, measurement_time=measurement_time,
                                             seed=seed_poisson)  # flux from sweep
                                else:
                                    wf.poisson(flux=flux_new, freq=freq_new,
                                             measurement_time=measurement_time,
                                             seed=seed_poisson)

                            if apply_overlap:
                                wf.overlap(kernel=kernel)

                            # Set up groupby
                            if use_num_points:
                                wf.groupby(param_to_sweep, low=low_val, high=high_val, num=num_points)
                            else:
                                wf.groupby(param_to_sweep, low=low_val, high=high_val, step=step_val)

                            # Reconstruct - don't pass the swept parameter here, it will be applied in the sweep
                            if param_to_sweep == 'noise_power':
                                # Sweeping noise_power, so use wiener but don't pass noise_power
                                wf.reconstruct(kind='wiener', tmin=tmin, tmax=tmax)
                            elif param_to_sweep == 'iterations':
                                # Sweeping iterations, so use lucy but don't pass iterations
                                wf.reconstruct(kind='lucy', tmin=tmin, tmax=tmax)
                            else:
                                # Not sweeping a reconstruction parameter, so pass them normally
                                wf.reconstruct(kind=recon_method, tmin=tmin, tmax=tmax, **recon_params)

                            # Analyze (requires xs parameter)
                            wf.analyze(xs='iron')

                            # Progress bar placeholder
                            progress_placeholder = st.empty()
                            progress_bar = progress_placeholder.progress(0.0)

                            # Run the sweep
                            results_df = wf.run(progress_bar=False)

                            # Store results
                            st.session_state.sweep_results = results_df

                            progress_bar.progress(1.0)
                            st.success(f"✅ Sweep completed! Processed {len(results_df)} configurations.")

                        except Exception as e:
                            st.error(f"Error during sweep: {e}")
                            st.exception(e)

                # Display results if available
                if 'sweep_results' in st.session_state and st.session_state.sweep_results is not None:
                    results_df = st.session_state.sweep_results

                    # Show summary statistics
                    st.markdown("**Summary Statistics**")
                    summary_col1, summary_col2, summary_col3 = st.columns(3)

                    with summary_col1:
                        if param_to_sweep in results_df.columns and y_param in results_df.columns:
                            # Drop NaN values before finding best
                            valid_df = results_df.dropna(subset=[y_param])
                            if len(valid_df) > 0:
                                best_idx = valid_df[y_param].idxmin() if 'chi2' in y_param or 'aic' in y_param or 'bic' in y_param else valid_df[y_param].idxmax()
                                best_value = results_df.loc[best_idx, param_to_sweep]
                                st.metric("Best Value", f"{best_value:.4g}")
                            else:
                                st.metric("Best Value", "N/A")

                    with summary_col2:
                        if y_param in results_df.columns:
                            valid_df = results_df.dropna(subset=[y_param])
                            if len(valid_df) > 0:
                                best_metric = valid_df[y_param].min() if 'chi2' in y_param or 'aic' in y_param or 'bic' in y_param else valid_df[y_param].max()
                                st.metric(f"Best {y_param_options[y_param]}", f"{best_metric:.4g}")
                            else:
                                st.metric(f"Best {y_param_options[y_param]}", "N/A")

                    with summary_col3:
                        st.metric("Total Runs", len(results_df))

                    # Plot results
                    if param_to_sweep in results_df.columns and y_param in results_df.columns:
                        fig = go.Figure()

                        # Plot all points (including NaN which will be skipped by plotly)
                        fig.add_trace(go.Scatter(
                            x=results_df[param_to_sweep],
                            y=results_df[y_param],
                            mode='lines+markers',
                            name=y_param_options[y_param],
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=8)
                        ))

                        # Highlight best point (if valid data exists)
                        valid_df = results_df.dropna(subset=[y_param])
                        if len(valid_df) > 0:
                            best_idx = valid_df[y_param].idxmin() if 'chi2' in y_param or 'aic' in y_param or 'bic' in y_param else valid_df[y_param].idxmax()
                            fig.add_trace(go.Scatter(
                                x=[results_df.loc[best_idx, param_to_sweep]],
                                y=[results_df.loc[best_idx, y_param]],
                                mode='markers',
                                name='Best',
                                marker=dict(size=15, color='red', symbol='star')
                            ))

                        fig.update_layout(
                            title=f"{y_param_options[y_param]} vs {sweep_params[param_to_sweep]}",
                            xaxis_title=sweep_params[param_to_sweep],
                            yaxis_title=y_param_options[y_param],
                            hovermode='x unified',
                            template='plotly_white',
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Show data table
                    with st.expander("📊 View Full Results Table"):
                        st.dataframe(results_df, use_container_width=True)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"sweep_{param_to_sweep}_{y_param}.csv",
                            mime="text/csv"
                        )

        else:
            st.info("⚠️ Please run a reconstruction first (see Reconstruction tab) before using GroupBy.")

else:
    # Landing page
    st.info("👈 Configure the pipeline in the sidebar and click **Run Pipeline** to start!")

    st.markdown("""
    ## How to Use

    1. **Configure Parameters**: Use the sidebar to set parameters for each processing stage
    2. **Run Pipeline**: Click the "Run Pipeline" button to process the data
    3. **Explore Results**: Use the tabs above to view different aspects of the analysis

    ## Processing Stages

    - 📁 **Data Loading**: Load iron powder and openbeam data with original measurement parameters
    - 🔊 **Instrument Response**: Convolve with instrument pulse shape
    - 🎲 **Poisson Sampling**: Apply counting statistics with new flux conditions
    - 🔄 **Frame Overlap**: Create overlapping frames (2-4 frames supported)
    - 🔧 **Reconstruction**: Recover original signal using deconvolution
    - 🔁 **GroupBy**: Parameter sweep for optimization and sensitivity analysis

    ## Available Methods

    - **Wiener Filter**: Fast, works well with known noise
    - **Lucy-Richardson**: Iterative, good for positive-valued data
    - **Tikhonov**: Regularization-based, smooth results

    ## Features

    - **Interactive Plots**: Zoom, pan, and hover to explore data
    - **Parameter Sweeps**: Automatically explore parameter space with GroupBy
    - **Real-time Processing**: See results instantly as you adjust parameters
    - **Export Results**: Download sweep results as CSV
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Frame Overlap v0.2.0**
[Documentation](https://tsvikihirsh.github.io/frame_overlap/) |
[GitHub](https://github.com/TsvikiHirsh/frame_overlap)
""")
