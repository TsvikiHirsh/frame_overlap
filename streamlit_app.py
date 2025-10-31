"""
Frame Overlap Interactive Explorer - Streamlit App

Explore neutron Time-of-Flight frame overlap analysis interactively!
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import frame_overlap
import sys
sys.path.insert(0, 'src')
from frame_overlap import Data, Reconstruct, Workflow

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
with st.sidebar.expander("📁 1. Data Loading", expanded=True):
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
with st.sidebar.expander("🔊 2. Instrument Response", expanded=True):
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
with st.sidebar.expander("🎲 3. Poisson Sampling", expanded=True):
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
with st.sidebar.expander("🔄 4. Frame Overlap", expanded=True):
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Signal", "📈 Transmission", "🔍 Reconstruction", "📉 Statistics"])

    with tab1:
        st.header("Signal Processing Stages")

        col1, col2 = st.columns([3, 1])

        with col1:
            show_stages = st.checkbox("Show All Stages", value=True)
            show_errors = st.checkbox("Show Error Bars", value=False)

            fig, ax = plt.subplots(figsize=(12, 6))

            if show_stages:
                # Plot all stages
                stages = [
                    (data.data, 'Original', 'blue'),
                ]
                if data.convolved_data is not None:
                    stages.append((data.convolved_data, 'Convolved', 'orange'))
                if data.poissoned_data is not None:
                    stages.append((data.poissoned_data, 'Poissoned', 'green'))
                if data.overlapped_data is not None:
                    stages.append((data.overlapped_data, 'Overlapped', 'red'))

                for df, label, color in stages:
                    time_ms = df['time'].values / 1000
                    counts = df['counts'].values
                    ax.plot(time_ms, counts, label=label, alpha=0.7, color=color, drawstyle='steps-mid')

                    if show_errors:
                        err = df['err'].values
                        ax.fill_between(time_ms, counts - err, counts + err, alpha=0.2, color=color)
            else:
                # Plot current stage
                if data.overlapped_data is not None:
                    df = data.overlapped_data
                    label = 'Overlapped'
                elif data.poissoned_data is not None:
                    df = data.poissoned_data
                    label = 'Poissoned'
                elif data.convolved_data is not None:
                    df = data.convolved_data
                    label = 'Convolved'
                else:
                    df = data.data
                    label = 'Original'

                time_ms = df['time'].values / 1000
                counts = df['counts'].values
                ax.plot(time_ms, counts, label=label, color='blue', drawstyle='steps-mid')

                if show_errors:
                    err = df['err'].values
                    ax.fill_between(time_ms, counts - err, counts + err, alpha=0.3)

            ax.set_xlabel('Time (ms)', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title('Signal Processing', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

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
        st.header("Transmission")

        if data.op_data is not None:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Calculate transmission
            if data.overlapped_data is not None and data.op_overlapped_data is not None:
                sig = data.overlapped_data
                op = data.op_overlapped_data
                stage_name = "Overlapped"
            elif data.poissoned_data is not None and data.op_poissoned_data is not None:
                sig = data.poissoned_data
                op = data.op_poissoned_data
                stage_name = "Poissoned"
            elif data.convolved_data is not None and data.op_convolved_data is not None:
                sig = data.convolved_data
                op = data.op_convolved_data
                stage_name = "Convolved"
            else:
                sig = data.data
                op = data.op_data
                stage_name = "Original"

            # Calculate transmission
            min_len = min(len(sig), len(op))
            time_ms = sig['time'].values[:min_len] / 1000
            transmission = sig['counts'].values[:min_len] / np.maximum(op['counts'].values[:min_len], 1)

            ax.plot(time_ms, transmission, drawstyle='steps-mid', color='purple', linewidth=1.5)
            ax.set_xlabel('Time (ms)', fontsize=12)
            ax.set_ylabel('Transmission', fontsize=12)
            ax.set_title(f'Transmission ({stage_name})', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No openbeam data available for transmission calculation.")

    with tab3:
        st.header("Reconstruction Results")

        if st.session_state.recon is not None:
            recon = st.session_state.recon

            # Plot reconstruction
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

            time_ms = recon.reference_data['time'].values / 1000
            reference = recon.reference_data['counts'].values
            reconstructed = recon.reconstructed_data['counts'].values
            errors = recon.reference_data['err'].values

            # Top plot: Data comparison
            ax1.plot(time_ms, reference, 'k-', label='Target (Poissoned+Convolved)', linewidth=2, alpha=0.7)
            ax1.plot(time_ms, reconstructed, 'r--', label='Reconstructed', linewidth=1.5)
            ax1.fill_between(time_ms, reference - errors, reference + errors, alpha=0.2, color='gray')

            # Add tmin/tmax indicators
            if tmin is not None:
                ax1.axvline(tmin, color='green', linestyle=':', alpha=0.6, linewidth=2, label=f'tmin={tmin}')
            if tmax is not None:
                ax1.axvline(tmax, color='orange', linestyle=':', alpha=0.6, linewidth=2, label=f'tmax={tmax}')

            ax1.set_ylabel('Counts', fontsize=12)
            ax1.set_title(f'Reconstruction: {recon_method.title()} Filter', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Bottom plot: Residuals
            residuals = (reference - reconstructed) / np.maximum(errors, 1e-10)
            ax2.plot(time_ms, residuals, 'b-', alpha=0.7)
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax2.axhline(2, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(-2, color='gray', linestyle='--', alpha=0.5)
            ax2.fill_between(time_ms, -2, 2, alpha=0.1, color='green')

            if tmin is not None:
                ax2.axvline(tmin, color='green', linestyle=':', alpha=0.6, linewidth=2)
            if tmax is not None:
                ax2.axvline(tmax, color='orange', linestyle=':', alpha=0.6, linewidth=2)

            ax2.set_xlabel('Time (ms)', fontsize=12)
            ax2.set_ylabel('Residuals (σ)', fontsize=12)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run reconstruction to see results here.")

    with tab4:
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
                    'Metric': ['χ²/dof', 'RMSE', 'MAE', 'Max Error', 'R²'],
                    'Value': [
                        f"{stats['chi2_per_dof']:.2f}",
                        f"{stats['rmse']:.2e}",
                        f"{stats['mae']:.2e}",
                        f"{stats['max_abs_error']:.2e}",
                        f"{stats['r_squared']:.4f}"
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

    ## Available Methods

    - **Wiener Filter**: Fast, works well with known noise
    - **Lucy-Richardson**: Iterative, good for positive-valued data
    - **Tikhonov**: Regularization-based, smooth results
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Frame Overlap v0.2.0**
[Documentation](https://tsvikihirsh.github.io/frame_overlap/) |
[GitHub](https://github.com/TsvikiHirsh/frame_overlap)
""")
