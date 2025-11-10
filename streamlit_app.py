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
import sys
import traceback
import subprocess
import platform

# Import frame_overlap
sys.path.insert(0, 'src')
from frame_overlap import Data, Reconstruct, Workflow

# Try to import Analysis with detailed error reporting
ANALYSIS_AVAILABLE = False
ANALYSIS_ERROR = None
ANALYSIS_ERROR_DETAILS = {}

try:
    from frame_overlap import Analysis
    ANALYSIS_AVAILABLE = True
except Exception as e:
    ANALYSIS_AVAILABLE = False
    ANALYSIS_ERROR = str(e)
    ANALYSIS_ERROR_DETAILS = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'traceback': traceback.format_exc()
    }

    # Try to diagnose the issue
    try:
        # Check if nbragg package exists
        import importlib.util
        nbragg_spec = importlib.util.find_spec("nbragg")
        ANALYSIS_ERROR_DETAILS['nbragg_found'] = nbragg_spec is not None

        if nbragg_spec:
            ANALYSIS_ERROR_DETAILS['nbragg_location'] = nbragg_spec.origin if nbragg_spec.origin else "unknown"

        # Try importing nbragg directly
        try:
            import nbragg
            ANALYSIS_ERROR_DETAILS['nbragg_imports'] = True
            ANALYSIS_ERROR_DETAILS['nbragg_version'] = getattr(nbragg, '__version__', 'unknown')
        except Exception as nbragg_err:
            ANALYSIS_ERROR_DETAILS['nbragg_imports'] = False
            ANALYSIS_ERROR_DETAILS['nbragg_import_error'] = str(nbragg_err)
            ANALYSIS_ERROR_DETAILS['nbragg_traceback'] = traceback.format_exc()

        # Check if NCrystal is available
        try:
            import NCrystal
            ANALYSIS_ERROR_DETAILS['ncrystal_available'] = True
            ANALYSIS_ERROR_DETAILS['ncrystal_version'] = NCrystal.__version__
        except Exception as nc_err:
            ANALYSIS_ERROR_DETAILS['ncrystal_available'] = False
            ANALYSIS_ERROR_DETAILS['ncrystal_error'] = str(nc_err)

    except Exception as diag_err:
        ANALYSIS_ERROR_DETAILS['diagnostic_error'] = str(diag_err)

# Use Agg backend for matplotlib (non-interactive, for conversion to images)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants for neutron time-of-flight conversions
SPEED_OF_LIGHT = 299792458  # m/s
MASS_OF_NEUTRON = 939.56542052 * 1e6 / (SPEED_OF_LIGHT ** 2)  # [eV s²/m²]
PLANCK_CONSTANT = 6.62607015e-34  # J·s
EV_TO_JOULE = 1.602176634e-19  # eV to Joules
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

def wavelength_to_tof(wavelength_angstrom, flight_path_length_m):
    """
    Convert neutron wavelength to time-of-flight.

    Parameters
    ----------
    wavelength_angstrom : float or array
        Wavelength in Angstroms
    flight_path_length_m : float
        Flight path length in meters

    Returns
    -------
    float or array
        Time-of-flight in microseconds
    """
    # Convert wavelength from Angstrom to meters
    wavelength_m = wavelength_angstrom * 1e-10

    # de Broglie relation: λ = h / (m * v)
    # v = h / (m * λ)
    velocity = PLANCK_CONSTANT / (NEUTRON_MASS_KG * wavelength_m)

    # t = L / v
    tof_seconds = flight_path_length_m / velocity

    # Convert to microseconds
    return tof_seconds * 1e6

def mpl_to_plotly(fig, show_errors=True):
    """
    Convert matplotlib figure to plotly figure for interactivity.
    Extracts data from matplotlib axes and recreates in plotly.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert
    show_errors : bool
        Whether to show error bars (default: True)
    """
    # Get the main axis (or first axis if multiple)
    axes = fig.get_axes()

    # Create plotly figure
    if len(axes) == 1:
        plotly_fig = go.Figure()
        ax = axes[0]

        # Extract lines and error bars from matplotlib
        # Process error bars first (lower zorder)
        if show_errors:
            for collection in ax.collections:
                # Error bars are typically PolyCollection objects
                if hasattr(collection, 'get_paths') and len(collection.get_paths()) > 0:
                    # This is likely an error bar fill
                    facecolors = collection.get_facecolor()
                    if len(facecolors) > 0:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) > 0:
                                plotly_fig.add_trace(go.Scatter(
                                    x=vertices[:, 0],
                                    y=vertices[:, 1],
                                    mode='lines',
                                    fill='toself',
                                    fillcolor=matplotlib.colors.to_hex(facecolors[0], keep_alpha=True),
                                    line=dict(width=0),
                                    opacity=0.2,
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))

        # Extract lines from matplotlib (higher zorder)
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

        # Process error bars first (lower zorder) for top plot
        if show_errors:
            for collection in ax_top.collections:
                if hasattr(collection, 'get_paths') and len(collection.get_paths()) > 0:
                    facecolors = collection.get_facecolor()
                    if len(facecolors) > 0:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            if len(vertices) > 0:
                                plotly_fig.add_trace(go.Scatter(
                                    x=vertices[:, 0],
                                    y=vertices[:, 1],
                                    mode='lines',
                                    fill='toself',
                                    fillcolor=matplotlib.colors.to_hex(facecolors[0], keep_alpha=True),
                                    line=dict(width=0),
                                    opacity=0.2,
                                    showlegend=False,
                                    hoverinfo='skip'
                                ), row=1, col=1)

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

# Show diagnostic information if Analysis failed to import
if not ANALYSIS_AVAILABLE:
    st.error("⚠️ **nbragg Analysis module failed to import**")

    with st.expander("🔍 Click here to see diagnostic information", expanded=True):
        st.markdown("### Import Error Details")
        st.code(ANALYSIS_ERROR, language="text")

        st.markdown("### System Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Platform:** {platform.system()} {platform.release()}")
            st.write(f"**Python Version:** {sys.version}")
        with col2:
            st.write(f"**Architecture:** {platform.machine()}")
            st.write(f"**Processor:** {platform.processor()}")

        st.markdown("### Dependency Diagnostics")

        if 'nbragg_found' in ANALYSIS_ERROR_DETAILS:
            if ANALYSIS_ERROR_DETAILS['nbragg_found']:
                st.success(f"✓ nbragg package found at: {ANALYSIS_ERROR_DETAILS.get('nbragg_location', 'unknown')}")
            else:
                st.error("✗ nbragg package not found in Python path")

        if 'nbragg_imports' in ANALYSIS_ERROR_DETAILS:
            if ANALYSIS_ERROR_DETAILS['nbragg_imports']:
                st.success(f"✓ nbragg imports successfully (version: {ANALYSIS_ERROR_DETAILS.get('nbragg_version', 'unknown')})")
            else:
                st.error(f"✗ nbragg import failed: {ANALYSIS_ERROR_DETAILS.get('nbragg_import_error', 'unknown error')}")
                if 'nbragg_traceback' in ANALYSIS_ERROR_DETAILS:
                    st.code(ANALYSIS_ERROR_DETAILS['nbragg_traceback'], language="text")

        if 'ncrystal_available' in ANALYSIS_ERROR_DETAILS:
            if ANALYSIS_ERROR_DETAILS['ncrystal_available']:
                st.success(f"✓ NCrystal available (version: {ANALYSIS_ERROR_DETAILS.get('ncrystal_version', 'unknown')})")
            else:
                st.error(f"✗ NCrystal not available: {ANALYSIS_ERROR_DETAILS.get('ncrystal_error', 'unknown error')}")

        st.markdown("### Installed Packages")
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Filter for relevant packages
                packages = result.stdout
                relevant = []
                for line in packages.split('\n'):
                    if any(pkg in line.lower() for pkg in ['nbragg', 'ncrystal', 'lmfit', 'numpy', 'scipy']):
                        relevant.append(line)
                if relevant:
                    st.code('\n'.join(relevant), language="text")
            else:
                st.warning("Could not list packages")
        except Exception as e:
            st.warning(f"Could not list packages: {e}")

        st.markdown("### Build Tools Check")
        build_tools = {
            'gcc': ['gcc', '--version'],
            'g++': ['g++', '--version'],
            'cmake': ['cmake', '--version'],
            'python3-dev': ['python3-config', '--includes']
        }

        for tool, cmd in build_tools.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0] if result.stdout else "installed"
                    st.success(f"✓ {tool}: {version_line}")
                else:
                    st.error(f"✗ {tool}: not available")
            except FileNotFoundError:
                st.error(f"✗ {tool}: not found")
            except Exception as e:
                st.warning(f"? {tool}: {e}")

        st.markdown("### Full Traceback")
        if 'traceback' in ANALYSIS_ERROR_DETAILS:
            st.code(ANALYSIS_ERROR_DETAILS['traceback'], language="text")

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

# Process button at the top
process_button = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Configure each stage of the analysis pipeline:")

# Initialize session state for workflow
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = None
if 'recon' not in st.session_state:
    st.session_state.recon = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

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

    st.markdown("**Wavelength Range**")
    col1, col2 = st.columns(2)
    with col1:
        lambda_min = st.number_input(
            "Min λ (Å)",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Minimum wavelength in Angstroms"
        )
    with col2:
        lambda_max = st.number_input(
            "Max λ (Å)",
            min_value=0.1,
            max_value=20.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Maximum wavelength in Angstroms"
        )

    # Convert wavelength to time range (using L=9m flight path)
    flight_path_m = 9.0
    tof_min_us = wavelength_to_tof(lambda_min, flight_path_m)
    tof_max_us = wavelength_to_tof(lambda_max, flight_path_m)

    # Show the converted time range
    st.caption(f"Time range: {tof_min_us/1000:.2f} - {tof_max_us/1000:.2f} ms")

# Stage 2: Convolution
with st.sidebar.expander("🔊 2. Instrument Response", expanded=False):
    apply_convolution = st.checkbox("Apply Convolution", value=True)

    if apply_convolution:
        pulse_duration = st.number_input(
            "Pulse Duration (µs)",
            min_value=0.0,
            max_value=5000.0,
            value=200.0,
            step=10.0,
            format="%.1f",
            help="Instrument pulse duration in microseconds (0-5000 µs)"
        )
        bin_width = 10  # Fixed at 10 µs
        st.caption("Bin Width: 10 µs (fixed)")
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

        # Frame time definition
        st.markdown("**Frame Time**")
        frame_time_mode = st.radio(
            "Define by",
            ["Frequency (Hz)", "Time (ms)"],
            help="Define frame time by frequency or directly in milliseconds"
        )

        if frame_time_mode == "Frequency (Hz)":
            freq_new = st.slider(
                "Frequency (Hz)",
                min_value=10,
                max_value=100,
                value=20,
                step=1,
                help="Pulse frequency (10-100 Hz)"
            )
            # Calculate frame time in ms from frequency
            frame_time_ms = 1000.0 / freq_new
            st.caption(f"Frame time: {frame_time_ms:.1f} ms")
        else:  # Time (ms)
            frame_time_ms = st.number_input(
                "Frame Time (ms)",
                min_value=10.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                format="%.1f",
                help="Frame time in milliseconds (10-100 ms)"
            )
            # Calculate frequency from frame time
            freq_new = int(1000.0 / frame_time_ms)
            st.caption(f"Frequency: {freq_new} Hz")

        measurement_time_hours = st.number_input(
            "Measurement Time (hours)",
            min_value=0.5,
            max_value=240.0,
            value=8.0,
            step=0.5,
            format="%.1f",
            help="New measurement duration in hours (0.5-240 hours)"
        )
        # Convert hours to minutes for internal use
        measurement_time = measurement_time_hours * 60

        seed_poisson = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )

        # Calculate and display duty cycle
        if apply_convolution and pulse_duration is not None:
            # Pulsed source duty cycle
            flux_ratio = flux_new / flux_orig
            time_ratio = (measurement_time_hours) / duration_orig
            duty_cycle_pulsed = flux_ratio * time_ratio * freq_new * (pulse_duration / 1e6)
            duty_cycle_percent = duty_cycle_pulsed * 100
            st.info(f"💡 Duty Cycle: {duty_cycle_percent:.4f}%")
        else:
            # Continuous source duty cycle
            flux_ratio = flux_new / flux_orig
            time_ratio = measurement_time_hours / duration_orig
            duty_cycle_continuous = flux_ratio * time_ratio
            duty_cycle_percent = duty_cycle_continuous * 100
            st.info(f"💡 Duty Cycle: {duty_cycle_percent:.4f}% (continuous source)")
    else:
        flux_new = None
        freq_new = None
        measurement_time = None
        measurement_time_hours = None
        seed_poisson = None
        frame_time_ms = None

# Stage 4: Frame Overlap
with st.sidebar.expander("🔄 4. Frame Overlap", expanded=False):
    apply_overlap = st.checkbox("Apply Overlap", value=True)

    if apply_overlap:
        # Get frame time from Poisson settings
        if apply_poisson and 'frame_time_ms' in locals():
            max_frame_time = frame_time_ms
        else:
            max_frame_time = 50.0  # Default if Poisson not applied

        st.caption(f"Frame time: {max_frame_time:.1f} ms (from Poisson settings)")

        # Kernel input mode
        kernel_mode = st.radio(
            "Kernel Input Mode",
            ["Auto-generate", "Manual"],
            help="Choose whether to auto-generate kernel or input manually"
        )

        if kernel_mode == "Manual":
            # Manual kernel input (time differences)
            kernel_str = st.text_input(
                "Kernel Time Differences (comma-separated, ms)",
                value="0,25",
                help="Enter time differences between frames in ms, e.g., '0,12.5,12.5,12.5' for 4 equally spaced frames in 50ms"
            )
            try:
                kernel = [float(x.strip()) for x in kernel_str.split(',')]
                n_frames = len(kernel)
                # Convert differences to absolute times for overlap() function
                kernel_absolute = [sum(kernel[:i+1]) for i in range(len(kernel))]
                total_time = kernel_absolute[-1] + 20 if kernel_absolute else 50
                st.success(f"✓ {n_frames} frames at: {[round(t, 1) for t in kernel_absolute]} ms")
            except ValueError:
                st.error("Invalid kernel format. Use comma-separated numbers.")
                kernel = [0, 25]
                kernel_absolute = [0, 25]
                n_frames = 2
                total_time = 50

        else:  # Auto-generate mode
            # Spacing type
            spacing_type = st.radio(
                "Spacing Type",
                ["Equal", "Random"],
                help="Equal spacing or random spacing between frames"
            )

            # Number of frames
            n_frames = st.slider(
                "Number of Frames",
                min_value=1,
                max_value=20,
                value=2,
                help="Number of overlapping frames"
            )

            # Generate kernel based on spacing type
            if spacing_type == "Equal":
                # Equally spaced frames - time differences
                if n_frames > 1:
                    spacing = max_frame_time / n_frames
                    kernel = [0.0] + [round(spacing, 2)] * (n_frames - 1)
                else:
                    kernel = [0.0]
            else:  # Random
                # Randomly spaced frames with seed for reproducibility
                seed_kernel = seed_poisson if seed_poisson is not None else 42
                np.random.seed(seed_kernel)
                if n_frames > 1:
                    # Generate random time differences that sum to max_frame_time
                    # Use Dirichlet distribution for random partitioning
                    random_fractions = np.random.dirichlet([1] * n_frames)
                    time_differences = random_fractions * max_frame_time
                    # First frame always at 0
                    kernel = [0.0] + [round(diff, 1) for diff in time_differences[1:]]
                else:
                    kernel = [0.0]

            # Display generated kernel in editable field
            kernel_str_generated = ','.join([str(k) for k in kernel])
            kernel_str_edited = st.text_input(
                "Generated Kernel Time Differences (editable, ms)",
                value=kernel_str_generated,
                help="Auto-generated time differences - you can edit if needed"
            )

            # Parse edited kernel
            try:
                kernel = [float(x.strip()) for x in kernel_str_edited.split(',')]
                n_frames = len(kernel)
            except ValueError:
                st.error("Invalid kernel format. Using auto-generated values.")

            # Convert differences to absolute times for overlap() function
            kernel_absolute = [sum(kernel[:i+1]) for i in range(len(kernel))]
            total_time = kernel_absolute[-1] + 20 if kernel_absolute else 50

            st.info(f"📊 Time differences: {kernel} ms → Frames at: {[round(t, 1) for t in kernel_absolute]} ms")
    else:
        kernel = None
        kernel_absolute = None
        total_time = None
        n_frames = 1

# Stage 5: Reconstruction
with st.sidebar.expander("🔧 5. Reconstruction", expanded=False):
    apply_reconstruction = st.checkbox("Apply Reconstruction", value=True)

    if apply_reconstruction:
        recon_method = st.selectbox(
            "Method",
            ["wiener", "wiener_smooth", "wiener_adaptive", "lucy", "tikhonov"],
            help="Deconvolution algorithm:\n"
                 "- wiener: Standard Wiener deconvolution\n"
                 "- wiener_smooth: Wiener with pre-smoothing (paper method)\n"
                 "- wiener_adaptive: Scipy adaptive Wiener + deconvolution\n"
                 "- lucy: Richardson-Lucy iterative\n"
                 "- tikhonov: Tikhonov regularization"
        )

        if recon_method in ["wiener", "wiener_smooth", "wiener_adaptive", "tikhonov"]:
            noise_power = st.slider(
                "Noise Power",
                min_value=0.001,
                max_value=1.0,
                value=0.2,
                step=0.001,
                format="%.3f",
                help="Regularization parameter"
            )
            recon_params = {"noise_power": noise_power}

            # Add smoothing window option for wiener_smooth
            if recon_method == "wiener_smooth":
                smooth_window = st.slider(
                    "Smooth Window",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2,
                    help="Window size for moving average smoothing (paper uses 5)"
                )
                recon_params["smooth_window"] = smooth_window

            # Add mysize option for wiener_adaptive
            elif recon_method == "wiener_adaptive":
                mysize = st.slider(
                    "Adaptive Window",
                    min_value=3,
                    max_value=21,
                    value=5,
                    step=2,
                    help="Window size for adaptive noise estimation"
                )
                recon_params["mysize"] = mysize

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
            tmin = st.number_input("Min Time (ms)", 0.0, 50.0, 3.7, 0.5)
            tmax = st.number_input("Max Time (ms)", 0.0, 50.0, 11.0, 0.5)
        else:
            tmin, tmax = None, None
    else:
        apply_reconstruction = False
        recon_method = None
        recon_params = {}
        tmin, tmax = None, None

# Stage 6: Analysis (nbragg)
with st.sidebar.expander("🔬 6. Analysis (nbragg)", expanded=False):
    if not ANALYSIS_AVAILABLE:
        st.error("⚠️ nbragg not available")
        st.caption("See diagnostic info in main page")
        apply_analysis = False
    else:
        apply_analysis = st.checkbox("Apply nbragg Analysis", value=False,
                                     help="Fit reconstructed data with nbragg material models")

    if apply_analysis and ANALYSIS_AVAILABLE:
        st.markdown("**nbragg Model Selection**")
        nbragg_model = st.selectbox(
            "Material Model",
            ["iron", "iron_with_cellulose", "iron_square_response"],
            index=0,  # Default to "iron"
            help="Select nbragg cross-section model:\n"
                 "- iron: Fe_sg229_Iron-alpha (recommended)\n"
                 "- iron_with_cellulose: Fe_sg225_Iron-gamma + cellulose\n"
                 "- iron_square_response: Fe_sg225_Iron-gamma with square response"
        )

        st.markdown("**Fitting Options**")
        vary_background = st.checkbox("Vary Background", value=True,
                                     help="Allow background to vary during fitting")
        vary_response = st.checkbox("Vary Response", value=True,
                                   help="Allow response function to vary during fitting")
    else:
        nbragg_model = "iron"
        vary_background = True
        vary_response = True

# Process button at the bottom (duplicate for convenience)
st.sidebar.markdown("---")
process_button_bottom = st.sidebar.button("🚀 Run Pipeline", type="primary", use_container_width=True, key="run_bottom")

# Main content area
if process_button or process_button_bottom:
    with st.spinner("Processing pipeline..."):
        try:
            # Create workflow
            data = Data(signal_path, openbeam_path,
                       flux=flux_orig, duration=duration_orig, freq=freq_orig)

            # Apply wavelength filtering (by filtering time range)
            if data.data is not None and data.op_data is not None:
                # Filter signal data
                mask_signal = (data.data['time'] >= tof_min_us) & (data.data['time'] <= tof_max_us)
                data.data = data.data[mask_signal].copy()
                data.table = data.data  # Update legacy reference

                # Filter openbeam data
                mask_openbeam = (data.op_data['time'] >= tof_min_us) & (data.op_data['time'] <= tof_max_us)
                data.op_data = data.op_data[mask_openbeam].copy()
                data.openbeam_table = data.op_data  # Update legacy reference

                st.sidebar.success(f"✓ Wavelength filtered: {lambda_min:.1f}-{lambda_max:.1f} Å")

            # Apply stages
            if apply_convolution:
                data.convolute_response(pulse_duration, bin_width=bin_width)
                st.sidebar.success(f"✓ Convolved (pulse: {pulse_duration} µs)")

            if apply_poisson:
                data.poisson_sample(flux=flux_new, freq=freq_new,
                                   measurement_time=measurement_time, seed=seed_poisson)
                st.sidebar.success(f"✓ Poisson (flux: {flux_new:.1e})")

            if apply_overlap:
                data.overlap(kernel=kernel_absolute, total_time=total_time)
                st.sidebar.success(f"✓ Overlap ({n_frames} frames)")

            st.session_state.workflow_data = data

            # Reconstruction
            if apply_reconstruction and apply_overlap:
                recon = Reconstruct(data, tmin=tmin, tmax=tmax)
                recon.filter(kind=recon_method, **recon_params)
                st.session_state.recon = recon

                stats = recon.get_statistics()
                st.sidebar.success(f"✓ Reconstructed (χ²/dof: {stats['chi2_per_dof']:.1f})")

            # Analysis (nbragg fitting)
            if apply_analysis and apply_reconstruction and apply_overlap and ANALYSIS_AVAILABLE:
                try:
                    analysis = Analysis(xs=nbragg_model, vary_background=vary_background,
                                       vary_response=vary_response)

                    # Prepare nbragg data and clean NaN/Inf values (critical for fitting!)
                    nbragg_data = recon.to_nbragg(L=9.0, tstep=10e-6)

                    # Remove NaN values
                    nbragg_data.table = nbragg_data.table.dropna()

                    # Remove inf values (can occur from division by zero in transmission calculation)
                    nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['trans'])]
                    nbragg_data.table = nbragg_data.table[~np.isinf(nbragg_data.table['err'])]

                    # Remove zero or negative errors
                    nbragg_data.table = nbragg_data.table[nbragg_data.table['err'] > 0]

                    # Fit using the cleaned data directly with the model
                    result = analysis.model.fit(nbragg_data)
                    analysis.result = result
                    analysis.data = nbragg_data

                    # Check if result is valid (has redchi attribute and it's not NaN)
                    if hasattr(result, 'redchi') and not pd.isna(result.redchi):
                        st.session_state.analysis = analysis
                        st.sidebar.success(f"✓ nbragg fit (χ²/dof: {result.redchi:.2f})")
                    else:
                        st.sidebar.warning(f"⚠️ nbragg fit produced invalid results")
                        st.session_state.analysis = None

                except Exception as e:
                    st.sidebar.error(f"⚠️ nbragg fit failed: {str(e)}")
                    st.session_state.analysis = None

                    # Show detailed error information
                    with st.expander("🔍 nbragg Error Details", expanded=True):
                        st.error(f"**Error Type:** {type(e).__name__}")
                        st.error(f"**Error Message:** {str(e)}")

                        st.markdown("### Full Traceback")
                        st.code(traceback.format_exc(), language="text")

                        # Try to diagnose the specific issue
                        st.markdown("### Diagnostic Checks")
                        try:
                            import sys
                            import importlib.util

                            # Check if nbragg can be found
                            nbragg_spec = importlib.util.find_spec("nbragg")
                            if nbragg_spec:
                                st.success(f"✓ nbragg package found at: {nbragg_spec.origin}")
                            else:
                                st.error("✗ nbragg package not found in sys.path")
                                st.write("sys.path:", sys.path)

                            # Try importing nbragg directly
                            try:
                                import nbragg as test_nbragg
                                st.success(f"✓ nbragg imports successfully (version: {getattr(test_nbragg, '__version__', 'unknown')})")

                                # Check materials
                                try:
                                    materials = test_nbragg.materials
                                    st.success(f"✓ nbragg.materials accessible")
                                    st.write(f"Available materials: {len(materials)} items")
                                except Exception as mat_err:
                                    st.error(f"✗ nbragg.materials failed: {mat_err}")

                            except ImportError as imp_err:
                                st.error(f"✗ nbragg import failed: {imp_err}")
                                st.code(traceback.format_exc(), language="text")

                        except Exception as diag_err:
                            st.error(f"Diagnostic check failed: {diag_err}")
            else:
                st.session_state.analysis = None

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
                plotly_fig = mpl_to_plotly(mpl_fig, show_errors=show_errors)

                # Apply log scale if requested
                if log_scale:
                    plotly_fig.update_yaxes(type="log")

                st.plotly_chart(plotly_fig, use_container_width=True)
                plt.close(mpl_fig)

        with col2:
            st.markdown("**Current Stage Info**")
            if data.overlapped_data is not None:
                st.metric("Stage", "Overlapped")
                st.metric("Frames", len(kernel_absolute) if kernel_absolute else "N/A")
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
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

            with ctrl_col1:
                plot_type = st.radio(
                    "Plot Type",
                    ["transmission", "signal"],
                    format_func=lambda x: "Transmission" if x == "transmission" else "Signal",
                    horizontal=True
                )

            with ctrl_col2:
                show_errors_recon = st.checkbox("Show Error Bars", value=False, key="recon_errors")

            with ctrl_col3:
                recon_log_scale = st.checkbox("Log Scale (Y)", value=False, key="recon_log")

            # Use Reconstruct.plot() method with ylim for transmission
            if plot_type == 'transmission':
                mpl_fig = recon.plot(kind=plot_type, show_errors=show_errors_recon,
                                    figsize=(12, 8), ylim=(0, 1))
            else:
                mpl_fig = recon.plot(kind=plot_type, show_errors=show_errors_recon, figsize=(12, 8))

            # Add nbragg fit curve if available
            if st.session_state.analysis is not None and plot_type == 'transmission':
                try:
                    result = st.session_state.analysis.result

                    # Get the axes from the reconstruction plot
                    axes = mpl_fig.get_axes()
                    if len(axes) >= 1:
                        ax_data = axes[0]

                        # Get nbragg best fit data
                        # The wavelength array that corresponds to best_fit is in result.userkws['wl']
                        wavelength_angstrom = result.userkws['wl']  # in Angstroms
                        best_fit_transmission = result.best_fit

                        # Convert wavelength to time-of-flight using proper physics
                        L = 9.0  # Flight path in meters (default)
                        time_us = wavelength_to_tof(wavelength_angstrom, L)  # time in microseconds
                        time_ms = time_us / 1000  # time in milliseconds

                        # Plot nbragg fit on the same axes
                        ax_data.plot(time_ms, best_fit_transmission,
                                   label='nbragg fit', color='green', linewidth=2, linestyle='--')
                        ax_data.legend()

                        st.success(f"✅ nbragg fit overlay added ({len(best_fit_transmission)} points, {time_ms.min():.2f}-{time_ms.max():.2f} ms)")

                except Exception as e:
                    st.error(f"❌ Could not add nbragg fit to plot: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            elif plot_type == 'transmission':
                if st.session_state.analysis is None:
                    st.info("💡 Enable 'Apply nbragg Analysis' in sidebar (Stage 6) to see the nbragg fit curve")

            # Convert to Plotly for interactivity
            plotly_fig = mpl_to_plotly(mpl_fig, show_errors=show_errors_recon)

            # Apply log scale if requested (only to top plot)
            if recon_log_scale:
                plotly_fig.update_yaxes(type="log", row=1, col=1)

            st.plotly_chart(plotly_fig, use_container_width=True)
            plt.close(mpl_fig)
        else:
            st.info("Run reconstruction to see results here.")

    with tab3:
        st.header("Statistics & Metrics")

        # Debug indicator for nbragg analysis
        if st.session_state.analysis is not None:
            st.success("✅ nbragg analysis results available")
        else:
            st.info("💡 Enable 'Apply nbragg Analysis' in sidebar (Stage 6) to see fit results below")

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
                    f"{len(kernel_absolute)}" if kernel_absolute else "1"
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

        # Show nbragg fit results if available
        if st.session_state.analysis is not None:
            st.markdown("---")
            st.subheader("nbragg Fit Results")

            try:
                result = st.session_state.analysis.result

                # Show reduced chi-squared
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Reduced χ² (nbragg)", f"{result.redchi:.4f}")

                with col2:
                    # Quality indicator for nbragg fit
                    if result.redchi < 2:
                        st.success(f"✅ Excellent nbragg fit")
                    elif result.redchi < 5:
                        st.info(f"ℹ️ Good nbragg fit")
                    else:
                        st.warning(f"⚠️ Poor nbragg fit")

                # Show fit parameters table
                st.markdown("**Fitted Parameters**")

                # Extract parameters from lmfit result
                params_data = []
                for param_name, param in result.params.items():
                    # Handle stderr which might be None
                    if param.stderr is not None:
                        stderr_str = f"{param.stderr:.4e}"
                    else:
                        stderr_str = "N/A"

                    params_data.append({
                        'Parameter': param_name,
                        'Value': f"{param.value:.4e}",
                        'Stderr': stderr_str,
                        'Vary': 'Yes' if param.vary else 'No'
                    })

                params_table = pd.DataFrame(params_data)
                st.dataframe(params_table, hide_index=True, use_container_width=True)

                # Show fit report in expander
                with st.expander("View Full Fit Report"):
                    fit_report = result.fit_report()
                    st.text(fit_report)

            except Exception as e:
                st.error(f"Error displaying nbragg fit results: {e}")

    with tab4:
        st.header("GroupBy - Parameter Sweep")

        if st.session_state.recon is not None:
            st.markdown("""
            Run a parameter sweep to explore how different parameter values affect reconstruction quality.
            """)

            # Run sweep button at the top
            run_sweep = st.button("🚀 Run Parameter Sweep", type="primary", use_container_width=True, key="run_sweep_top")

            st.markdown("---")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Sweep Configuration")

                # Parameter selection
                sweep_params = {
                    'pulse_duration': 'Pulse Duration (µs)',
                    'noise_power': 'Noise Power',
                    'iterations': 'Lucy-Richardson Iterations',
                    'flux': 'Flux (n/cm²/s)',
                    'freq': 'Frequency (Hz)',
                    'n_frames': 'Number of Frames (equally spaced)',
                    'n_frames_random': 'Number of Frames (random spacing)',
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
                    'freq': (10, 100, 10),
                    'n_frames': (2, 5, 1),
                    'n_frames_random': (2, 5, 1),
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
                    'chi2_per_dof': 'χ²/dof (Reduced Chi-squared)',
                    'rmse': 'RMSE (Root Mean Square Error)',
                    'nrmse': 'NRMSE (Normalized RMSE)',
                    'r_squared': 'R² (Coefficient of Determination)',
                }

                # Add nbragg parameters if analysis is enabled
                if apply_analysis and ANALYSIS_AVAILABLE:
                    y_param_options['nbragg_redchi'] = 'nbragg Reduced χ²'
                    y_param_options['nbragg_thickness'] = 'nbragg Thickness (cm)'

                y_param = st.selectbox(
                    "Y-axis Parameter",
                    options=list(y_param_options.keys()),
                    format_func=lambda x: y_param_options[x],
                    help="Select which metric to plot"
                )

                # Duplicate run button at bottom for convenience
                st.markdown("---")
                run_sweep_bottom = st.button("🚀 Run Parameter Sweep", type="primary", use_container_width=True, key="run_sweep_bottom")

            with col2:
                st.subheader("Results")

                if run_sweep or run_sweep_bottom:
                    # Initialize session state for sweep results
                    if 'sweep_results' not in st.session_state:
                        st.session_state.sweep_results = None

                    with st.spinner(f"Running parameter sweep for {sweep_params[param_to_sweep]}..."):
                        try:
                            # Manual sweep without Analysis (simpler and doesn't fail)
                            # We don't use Workflow here - just manual loop with Data and Reconstruct
                            # Get sweep parameter values
                            if use_num_points:
                                param_values = np.linspace(low_val, high_val, num_points)
                            else:
                                param_values = np.arange(low_val, high_val + step_val/2, step_val)

                            # Progress bar
                            progress_placeholder = st.empty()
                            progress_bar = progress_placeholder.progress(0.0)

                            results = []
                            recon_objects = []  # Store reconstruction objects for individual viewing

                            for i, value in enumerate(param_values):
                                try:
                                    # Reload data
                                    data_sweep = Data(signal_path, openbeam_path,
                                                     flux=flux_orig, duration=duration_orig, freq=freq_orig)

                                    # Apply wavelength filtering (by filtering time range)
                                    if data_sweep.data is not None and data_sweep.op_data is not None:
                                        # Filter signal data
                                        mask_signal = (data_sweep.data['time'] >= tof_min_us) & (data_sweep.data['time'] <= tof_max_us)
                                        data_sweep.data = data_sweep.data[mask_signal].copy()
                                        data_sweep.table = data_sweep.data  # Update legacy reference

                                        # Filter openbeam data
                                        mask_openbeam = (data_sweep.op_data['time'] >= tof_min_us) & (data_sweep.op_data['time'] <= tof_max_us)
                                        data_sweep.op_data = data_sweep.op_data[mask_openbeam].copy()
                                        data_sweep.openbeam_table = data_sweep.op_data  # Update legacy reference

                                    # Apply stages
                                    if apply_convolution:
                                        if param_to_sweep == 'pulse_duration':
                                            data_sweep.convolute_response(value, bin_width=bin_width)
                                        else:
                                            data_sweep.convolute_response(pulse_duration, bin_width=bin_width)

                                    if apply_poisson:
                                        if param_to_sweep == 'flux':
                                            data_sweep.poisson_sample(flux=value, freq=freq_new,
                                                                     measurement_time=measurement_time, seed=seed_poisson)
                                        elif param_to_sweep == 'freq':
                                            data_sweep.poisson_sample(flux=flux_new, freq=int(value),
                                                                     measurement_time=measurement_time, seed=seed_poisson)
                                        else:
                                            data_sweep.poisson_sample(flux=flux_new, freq=freq_new,
                                                                     measurement_time=measurement_time, seed=seed_poisson)

                                    if apply_overlap:
                                        # Handle n_frames sweeps
                                        if param_to_sweep == 'n_frames':
                                            # Equally spaced frames
                                            n = int(value)
                                            # Use existing spacing from kernel differences or default
                                            spacing = kernel[1] if len(kernel) > 1 else 25
                                            sweep_kernel = [i * spacing for i in range(n)]
                                            data_sweep.overlap(kernel=sweep_kernel)
                                        elif param_to_sweep == 'n_frames_random':
                                            # Randomly spaced frames
                                            n = int(value)
                                            np.random.seed(seed_poisson if seed_poisson else 42)  # Reproducible random
                                            max_time = frame_time_ms if 'frame_time_ms' in locals() else 50  # Use frame time from Poisson
                                            sweep_kernel = sorted(np.random.uniform(0, max_time, n))
                                            sweep_kernel[0] = 0  # First frame always at 0
                                            data_sweep.overlap(kernel=sweep_kernel)
                                        else:
                                            data_sweep.overlap(kernel=kernel_absolute)

                                    # Reconstruct
                                    if param_to_sweep == 'noise_power':
                                        recon_sweep = Reconstruct(data_sweep, tmin=tmin, tmax=tmax)
                                        recon_sweep.filter(kind='wiener', noise_power=value)
                                    elif param_to_sweep == 'iterations':
                                        recon_sweep = Reconstruct(data_sweep, tmin=tmin, tmax=tmax)
                                        recon_sweep.filter(kind='lucy', iterations=int(value))
                                    else:
                                        recon_sweep = Reconstruct(data_sweep, tmin=tmin, tmax=tmax)
                                        recon_sweep.filter(kind=recon_method, **recon_params)

                                    # Get reconstruction statistics
                                    stats = recon_sweep.get_statistics()

                                    result_dict = {
                                        param_to_sweep: value,
                                        'chi2': stats.get('chi2', np.nan),
                                        'chi2_per_dof': stats.get('chi2_per_dof', np.nan),
                                        'rmse': stats.get('rmse', np.nan),
                                        'nrmse': stats.get('nrmse', np.nan),
                                        'r_squared': stats.get('r_squared', np.nan),
                                        'n_points': stats.get('n_points', 0)
                                    }

                                    # Run nbragg analysis if enabled
                                    if apply_analysis and ANALYSIS_AVAILABLE:
                                        try:
                                            analysis_sweep = Analysis(xs=nbragg_model,
                                                                     vary_background=vary_background,
                                                                     vary_response=vary_response)
                                            nbragg_result = analysis_sweep.fit(recon_sweep)

                                            # Extract nbragg parameters
                                            result_dict['nbragg_redchi'] = nbragg_result.redchi

                                            # Try to get thickness parameter (it might be named differently)
                                            thickness_param = None
                                            for param_name in nbragg_result.params.keys():
                                                if 'thickness' in param_name.lower() or 'L' in param_name or 'length' in param_name.lower():
                                                    thickness_param = nbragg_result.params[param_name].value
                                                    break
                                            result_dict['nbragg_thickness'] = thickness_param if thickness_param is not None else np.nan
                                        except Exception as e_nbragg:
                                            result_dict['nbragg_redchi'] = np.nan
                                            result_dict['nbragg_thickness'] = np.nan
                                    else:
                                        result_dict['nbragg_redchi'] = np.nan
                                        result_dict['nbragg_thickness'] = np.nan

                                    results.append(result_dict)

                                    # Store reconstruction object for individual viewing
                                    recon_objects.append({
                                        'value': value,
                                        'recon': recon_sweep,
                                        'data': data_sweep
                                    })

                                except Exception as e:
                                    st.warning(f"Error at {param_to_sweep}={value:.4g}: {e}")
                                    results.append({
                                        param_to_sweep: value,
                                        'chi2': np.nan,
                                        'error': str(e)
                                    })

                                # Update progress
                                progress_bar.progress((i + 1) / len(param_values))

                            # Store results
                            results_df = pd.DataFrame(results)
                            st.session_state.sweep_results = results_df
                            st.session_state.sweep_recon_objects = recon_objects
                            st.session_state.sweep_param_name = param_to_sweep

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
                                # Minimize chi2, rmse, nrmse, nbragg_redchi; Maximize r_squared
                                minimize_params = ['chi2', 'chi2_per_dof', 'rmse', 'nrmse', 'nbragg_redchi']
                                best_idx = valid_df[y_param].idxmin() if y_param in minimize_params else valid_df[y_param].idxmax()
                                best_value = results_df.loc[best_idx, param_to_sweep]
                                st.metric("Best Value", f"{best_value:.4g}")
                            else:
                                st.metric("Best Value", "N/A")

                    with summary_col2:
                        if y_param in results_df.columns:
                            valid_df = results_df.dropna(subset=[y_param])
                            if len(valid_df) > 0:
                                # Minimize chi2, rmse, nrmse, nbragg_redchi; Maximize r_squared
                                minimize_params = ['chi2', 'chi2_per_dof', 'rmse', 'nrmse', 'nbragg_redchi']
                                best_metric = valid_df[y_param].min() if y_param in minimize_params else valid_df[y_param].max()
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
                            # Minimize chi2, rmse, nrmse, nbragg_redchi; Maximize r_squared
                            minimize_params = ['chi2', 'chi2_per_dof', 'rmse', 'nrmse', 'nbragg_redchi']
                            best_idx = valid_df[y_param].idxmin() if y_param in minimize_params else valid_df[y_param].idxmax()
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

                    # Individual reconstruction viewer
                    if 'sweep_recon_objects' in st.session_state and st.session_state.sweep_recon_objects:
                        st.markdown("---")
                        st.markdown("### Individual Reconstruction Plots")

                        recon_objs = st.session_state.sweep_recon_objects
                        param_name = st.session_state.sweep_param_name

                        # Controls for individual viewer
                        viewer_col1, viewer_col2 = st.columns([3, 1])

                        with viewer_col2:
                            show_errors_individual = st.checkbox("Show Error Bars", value=True, key="individual_errors")

                        with viewer_col1:
                            # Slider to select which reconstruction to view
                            idx = st.slider(
                                f"Select {sweep_params.get(param_name, param_name)}",
                                min_value=0,
                                max_value=len(recon_objs) - 1,
                                value=0,
                                format=f"{sweep_params.get(param_name, param_name)} = %.4g"
                            )

                        selected = recon_objs[idx]
                        st.markdown(f"**{sweep_params.get(param_name, param_name)}: {selected['value']:.4g}**")

                        # Plot reconstruction
                        try:
                            mpl_fig = selected['recon'].plot(kind='transmission', show_errors=show_errors_individual,
                                                            figsize=(12, 8), ylim=(0, 1))
                            plotly_fig = mpl_to_plotly(mpl_fig, show_errors=show_errors_individual)
                            st.plotly_chart(plotly_fig, use_container_width=True)
                            plt.close(mpl_fig)
                        except Exception as e:
                            st.error(f"Error plotting reconstruction: {e}")

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
    - 🔬 **Analysis (nbragg)**: Fit reconstructed data with material cross-section models
    - 🔁 **GroupBy**: Parameter sweep for optimization and sensitivity analysis

    ## Reconstruction Methods

    - **Wiener Filter**: Fast, works well with known noise
    - **Wiener Smooth**: Wiener with pre-smoothing (follows paper approach, good for noisy data)
    - **Wiener Adaptive**: Scipy adaptive noise estimation + Wiener deconvolution
    - **Lucy-Richardson**: Iterative, good for positive-valued data
    - **Tikhonov**: Regularization-based, smooth results

    ## nbragg Models

    - **iron**: Fe_sg229_Iron-alpha (recommended, default)
    - **iron_with_cellulose**: Fe_sg225_Iron-gamma + cellulose background
    - **iron_square_response**: Fe_sg225_Iron-gamma with square response function

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
