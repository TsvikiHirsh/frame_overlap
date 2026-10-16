"""
Streamlit App for Adaptive Bragg Edge Measurement

Interactive demonstration of adaptive chopper pattern optimization
for Bragg edge measurements in neutron imaging.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import our adaptive measurement modules
from frame_overlap import (
    BraggEdge,
    BraggEdgeSample,
    IncidentSpectrum,
    TOFCalibration,
    MeasurementSimulator,
    PatternLibrary,
    ForwardModel,
    BraggEdgeMeasurementSystem,
    AdaptiveEdgeOptimizer,
    MeasurementTarget,
    PerformanceEvaluator,
    PerformanceMetrics
)


# Page configuration
st.set_page_config(
    page_title="Adaptive Bragg Edge Measurement",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def create_sample(material, edge_position, edge_height, edge_width, strain):
    """Create Bragg edge sample"""
    edge = BraggEdge(
        position=edge_position * (1 + strain),
        height=edge_height,
        width=edge_width
    )
    return BraggEdgeSample(
        edges=[edge],
        background_transmission=0.95,
        material=material
    )


def simulate_measurement_comparison(
    flight_path,
    edge_position,
    edge_height,
    edge_width,
    strain,
    precision_required,
    flux,
    time_per_pattern,
    strategy
):
    """Run simulation comparing adaptive vs uniform strategies"""

    # Create sample
    sample = create_sample('Sample', edge_position, edge_height, edge_width, strain)

    # Create measurement system
    system = BraggEdgeMeasurementSystem(
        flight_path=flight_path,
        wavelength_range=(edge_position - 2, edge_position + 2),
        n_wavelength_bins=500,
        n_time_bins=1000
    )

    # Create target
    target = MeasurementTarget(
        material='Sample',
        expected_edge=edge_position,
        precision_required=precision_required,
        max_measurement_time=300.0
    )

    # Create optimizer
    optimizer = AdaptiveEdgeOptimizer(system, target, strategy=strategy)

    # Run simulation comparison
    try:
        adaptive_result, uniform_result = optimizer.simulate_comparison(
            sample,
            flux=flux,
            measurement_time_per_pattern=time_per_pattern
        )
        return adaptive_result, uniform_result, system, sample
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        return None, None, None, None


def plot_transmission_curve(wavelength, transmission, edge_position, title="Transmission Curve"):
    """Plot transmission curve with edge marker"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=wavelength,
        y=transmission,
        mode='lines',
        name='Transmission',
        line=dict(color='blue', width=2)
    ))

    # Add edge marker
    fig.add_vline(
        x=edge_position,
        line_dash="dash",
        line_color="red",
        annotation_text="Edge",
        annotation_position="top"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (Å)",
        yaxis_title="Transmission",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_chopper_pattern(pattern, time_grid, title="Chopper Pattern"):
    """Plot chopper pattern"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_grid[:len(pattern)] * 1e6,  # Convert to microseconds
        y=pattern,
        mode='lines',
        fill='tozeroy',
        name='Pattern',
        line=dict(color='green', width=1)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (μs)",
        yaxis_title="Chopper State (0=Closed, 1=Open)",
        hovermode='x unified',
        height=300,
        yaxis=dict(tickvals=[0, 1], ticktext=['Closed', 'Open'])
    )

    return fig


def plot_measurement_signal(time, signal, title="Detected Signal"):
    """Plot detected signal"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time * 1e6,  # Convert to microseconds
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color='purple', width=1)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (μs)",
        yaxis_title="Counts",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_convergence(adaptive_history, uniform_history):
    """Plot convergence comparison"""
    fig = go.Figure()

    # Adaptive strategy
    times_a = [h[0] for h in adaptive_history]
    prec_a = [h[1] for h in adaptive_history]

    fig.add_trace(go.Scatter(
        x=times_a,
        y=prec_a,
        mode='lines+markers',
        name='Adaptive',
        line=dict(color='blue', width=2)
    ))

    # Uniform strategy
    times_u = [h[0] for h in uniform_history]
    prec_u = [h[1] for h in uniform_history]

    fig.add_trace(go.Scatter(
        x=times_u,
        y=prec_u,
        mode='lines+markers',
        name='Uniform',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title="Convergence: Precision vs Time",
        xaxis_title="Measurement Time (s)",
        yaxis_title="Precision (Å)",
        yaxis_type="log",
        hovermode='x unified',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig


def plot_pattern_comparison(adaptive_patterns, uniform_patterns, time_grid):
    """Plot pattern comparison"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Adaptive Strategy - First 3 Patterns", "Uniform Strategy - First 3 Patterns")
    )

    colors = ['blue', 'green', 'red']

    # Plot adaptive patterns
    for i, pattern in enumerate(adaptive_patterns[:3]):
        fig.add_trace(
            go.Scatter(
                x=time_grid[:len(pattern)] * 1e6,
                y=pattern + i * 1.2,
                mode='lines',
                name=f'Adaptive {i+1}',
                line=dict(color=colors[i], width=1),
                showlegend=True
            ),
            row=1, col=1
        )

    # Plot uniform patterns
    for i, pattern in enumerate(uniform_patterns[:3]):
        fig.add_trace(
            go.Scatter(
                x=time_grid[:len(pattern)] * 1e6,
                y=pattern + i * 1.2,
                mode='lines',
                name=f'Uniform {i+1}',
                line=dict(color=colors[i], width=1),
                showlegend=True
            ),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Time (μs)")
    fig.update_yaxes(title_text="Pattern (offset for visibility)")
    fig.update_layout(height=600, hovermode='x unified')

    return fig


# Main app
def main():
    st.markdown('<div class="main-header">⚛️ Adaptive Bragg Edge Measurement Optimizer</div>', unsafe_allow_html=True)

    st.markdown("""
    This interactive app demonstrates **adaptive chopper pattern optimization** for Bragg edge measurements
    in neutron Time-of-Flight (TOF) imaging. Compare different strategies and see how adaptive methods
    can dramatically reduce measurement time while maintaining precision.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Measurement system parameters
    st.sidebar.subheader("Measurement System")
    flight_path = st.sidebar.slider(
        "Flight Path (m)",
        min_value=5.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
        help="Distance from chopper to detector"
    )

    flux = st.sidebar.select_slider(
        "Neutron Flux (n/s)",
        options=[1e5, 5e5, 1e6, 5e6, 1e7],
        value=1e6,
        format_func=lambda x: f"{x:.0e}",
        help="Neutron flux at sample position"
    )

    time_per_pattern = st.sidebar.slider(
        "Time per Pattern (s)",
        min_value=1.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
        help="Measurement time for each chopper pattern"
    )

    # Sample parameters
    st.sidebar.subheader("Sample Parameters")

    material_preset = st.sidebar.selectbox(
        "Material Preset",
        options=["Custom", "Iron (Fe)", "Aluminum (Al)", "Copper (Cu)"],
        help="Select material or use custom parameters"
    )

    if material_preset == "Iron (Fe)":
        edge_position_default = 4.05
        edge_height_default = 0.6
        edge_width_default = 0.08
    elif material_preset == "Aluminum (Al)":
        edge_position_default = 4.15
        edge_height_default = 0.5
        edge_width_default = 0.10
    elif material_preset == "Copper (Cu)":
        edge_position_default = 4.17
        edge_height_default = 0.55
        edge_width_default = 0.09
    else:
        edge_position_default = 4.0
        edge_height_default = 0.5
        edge_width_default = 0.1

    edge_position = st.sidebar.slider(
        "Edge Position (Å)",
        min_value=2.0,
        max_value=8.0,
        value=edge_position_default,
        step=0.05,
        help="Bragg edge position in wavelength"
    )

    edge_height = st.sidebar.slider(
        "Edge Height",
        min_value=0.1,
        max_value=1.0,
        value=edge_height_default,
        step=0.05,
        help="Edge contrast (0 = no edge, 1 = complete absorption)"
    )

    edge_width = st.sidebar.slider(
        "Edge Width (Å)",
        min_value=0.01,
        max_value=0.5,
        value=edge_width_default,
        step=0.01,
        help="Edge broadening (resolution + microstructure)"
    )

    strain = st.sidebar.slider(
        "Applied Strain",
        min_value=-0.01,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%f",
        help="Strain shifts edge position: λ = λ₀(1+ε)"
    )

    # Optimization parameters
    st.sidebar.subheader("Optimization Settings")

    precision_required = st.sidebar.slider(
        "Target Precision (Å)",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Required precision in edge position"
    )

    strategy = st.sidebar.selectbox(
        "Optimization Strategy",
        options=["bayesian", "gradient", "multi_resolution"],
        help="Adaptive optimization strategy"
    )

    # Run simulation button
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

    # Main content
    if run_button:
        with st.spinner("Running simulation..."):
            adaptive_result, uniform_result, system, sample = simulate_measurement_comparison(
                flight_path,
                edge_position,
                edge_height,
                edge_width,
                strain,
                precision_required,
                flux,
                time_per_pattern,
                strategy
            )

        if adaptive_result is not None and uniform_result is not None:
            # Results summary
            st.markdown('<div class="section-header">📊 Results Summary</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Adaptive Time",
                    f"{adaptive_result.measurement_time:.1f} s",
                    help="Total measurement time for adaptive strategy"
                )

            with col2:
                st.metric(
                    "Uniform Time",
                    f"{uniform_result.measurement_time:.1f} s",
                    help="Total measurement time for uniform strategy"
                )

            with col3:
                speedup = uniform_result.measurement_time / adaptive_result.measurement_time
                st.metric(
                    "Speedup",
                    f"{speedup:.2f}x",
                    delta=f"{adaptive_result.time_saved:.1f}% faster" if adaptive_result.time_saved else None,
                    delta_color="normal",
                    help="Time saved with adaptive strategy"
                )

            with col4:
                st.metric(
                    "Final Precision",
                    f"{adaptive_result.edge_uncertainty:.4f} Å",
                    help="Achieved precision in edge position"
                )

            # Tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Convergence",
                "🎯 Sample & Transmission",
                "🔄 Chopper Patterns",
                "📊 Measurements",
                "📋 Detailed Metrics"
            ])

            with tab1:
                st.markdown("### Convergence Comparison")
                st.markdown("""
                This plot shows how quickly each strategy converges to the target precision.
                **Lower is better** - the adaptive strategy reaches the target faster.
                """)

                fig_conv = plot_convergence(
                    adaptive_result.convergence_history,
                    uniform_result.convergence_history
                )
                st.plotly_chart(fig_conv, use_container_width=True)

                # Show convergence data
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Adaptive Strategy**")
                    adaptive_df = pd.DataFrame(
                        adaptive_result.convergence_history,
                        columns=['Time (s)', 'Precision (Å)']
                    )
                    st.dataframe(adaptive_df.head(10), use_container_width=True)

                with col2:
                    st.markdown("**Uniform Strategy**")
                    uniform_df = pd.DataFrame(
                        uniform_result.convergence_history,
                        columns=['Time (s)', 'Precision (Å)']
                    )
                    st.dataframe(uniform_df.head(10), use_container_width=True)

            with tab2:
                st.markdown("### Sample Transmission Curve")

                wavelength = system.wavelength_grid
                transmission = sample.transmission(wavelength)

                fig_trans = plot_transmission_curve(
                    wavelength,
                    transmission,
                    edge_position * (1 + strain)
                )
                st.plotly_chart(fig_trans, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.info(f"""
                    **True Edge Position:** {edge_position * (1 + strain):.4f} Å
                    **Measured (Adaptive):** {adaptive_result.edge_position:.4f} Å
                    **Error:** {abs(adaptive_result.edge_position - edge_position * (1 + strain)):.4f} Å
                    """)

                with col2:
                    measured_strain = (adaptive_result.edge_position - edge_position) / edge_position
                    strain_error = abs(measured_strain - strain)
                    st.info(f"""
                    **True Strain:** {strain*1e6:.1f} με
                    **Measured Strain:** {measured_strain*1e6:.1f} με
                    **Strain Error:** {strain_error*1e6:.1f} με
                    """)

            with tab3:
                st.markdown("### Chopper Pattern Comparison")
                st.markdown("""
                Compare the first few chopper patterns used by each strategy.
                Notice how the adaptive patterns focus on regions of interest.
                """)

                fig_patterns = plot_pattern_comparison(
                    adaptive_result.patterns,
                    uniform_result.patterns,
                    system.time_grid
                )
                st.plotly_chart(fig_patterns, use_container_width=True)

                # Pattern statistics
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Adaptive Pattern Stats**")
                    avg_duty = np.mean([np.sum(p)/len(p) for p in adaptive_result.patterns[:3]])
                    st.metric("Average Duty Cycle", f"{avg_duty:.2%}")
                    st.metric("Patterns Used", adaptive_result.n_patterns)

                with col2:
                    st.markdown("**Uniform Pattern Stats**")
                    avg_duty = np.mean([np.sum(p)/len(p) for p in uniform_result.patterns[:3]])
                    st.metric("Average Duty Cycle", f"{avg_duty:.2%}")
                    st.metric("Patterns Used", uniform_result.n_patterns)

            with tab4:
                st.markdown("### Measured Signals")
                st.markdown("First measurement from each strategy:")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Adaptive Strategy**")
                    if len(adaptive_result.measurements) > 0:
                        fig_meas_a = plot_measurement_signal(
                            system.time_grid[:len(adaptive_result.measurements[0])],
                            adaptive_result.measurements[0],
                            "Adaptive - First Measurement"
                        )
                        st.plotly_chart(fig_meas_a, use_container_width=True)

                with col2:
                    st.markdown("**Uniform Strategy**")
                    if len(uniform_result.measurements) > 0:
                        fig_meas_u = plot_measurement_signal(
                            system.time_grid[:len(uniform_result.measurements[0])],
                            uniform_result.measurements[0],
                            "Uniform - First Measurement"
                        )
                        st.plotly_chart(fig_meas_u, use_container_width=True)

            with tab5:
                st.markdown("### Detailed Performance Metrics")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Adaptive Strategy**")
                    st.json({
                        "Edge Position": f"{adaptive_result.edge_position:.6f} Å",
                        "Edge Uncertainty": f"{adaptive_result.edge_uncertainty:.6f} Å",
                        "Measurement Time": f"{adaptive_result.measurement_time:.2f} s",
                        "Number of Patterns": adaptive_result.n_patterns,
                        "Strain": f"{adaptive_result.strain*1e6:.2f} με" if adaptive_result.strain else "N/A",
                        "Time Saved": f"{adaptive_result.time_saved:.1f}%" if adaptive_result.time_saved else "N/A"
                    })

                with col2:
                    st.markdown("**Uniform Strategy**")
                    st.json({
                        "Edge Position": f"{uniform_result.edge_position:.6f} Å",
                        "Edge Uncertainty": f"{uniform_result.edge_uncertainty:.6f} Å",
                        "Measurement Time": f"{uniform_result.measurement_time:.2f} s",
                        "Number of Patterns": uniform_result.n_patterns,
                        "Strain": f"{uniform_result.strain*1e6:.2f} με" if uniform_result.strain else "N/A"
                    })

    else:
        # Show introduction when not running
        st.markdown('<div class="section-header">ℹ️ How It Works</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Adaptive Measurement Strategy

            1. **Initial Measurement**: Start with broad sampling
            2. **Edge Localization**: Identify region with Bragg edge
            3. **Focused Sampling**: Concentrate measurements on edge region
            4. **Bayesian Update**: Refine edge position estimate
            5. **Iterate**: Repeat until target precision achieved

            **Key Benefits:**
            - ✅ Faster convergence
            - ✅ Better neutron economy
            - ✅ Higher information gain per measurement
            - ✅ Adaptive to sample variations
            """)

        with col2:
            st.markdown("""
            ### Mathematical Framework

            **Forward Model:**
            ```
            y = A·x + noise
            ```
            where y = measured signal, x = transmission curve

            **Bayesian Update:**
            ```
            P(θ|D_new) ∝ P(D_new|θ) × P(θ|D_old)
            ```

            **Information Gain:**
            ```
            I = H(θ) - H(θ|D)
            ```

            **Optimization Goal:**
            ```
            minimize: measurement_time
            subject to: precision ≤ target
            ```
            """)

        st.markdown('<div class="section-header">🚀 Getting Started</div>', unsafe_allow_html=True)

        st.markdown("""
        1. **Configure** your measurement system in the sidebar
        2. **Select** a sample material or customize parameters
        3. **Set** the target precision and optimization strategy
        4. **Click** "Run Simulation" to compare adaptive vs uniform strategies
        5. **Explore** the results in the different tabs

        Try adjusting the parameters to see how they affect the optimization!
        """)

        # Example parameters
        with st.expander("📚 Example Scenarios"):
            st.markdown("""
            ### Scenario 1: Fast Strain Measurement
            - **Material**: Iron (Fe)
            - **Strain**: 0.001 (1000 με)
            - **Target Precision**: 0.005 Å
            - **Expected Speedup**: 2-3x

            ### Scenario 2: High-Precision Measurement
            - **Material**: Aluminum (Al)
            - **Strain**: 0.0001 (100 με)
            - **Target Precision**: 0.001 Å
            - **Expected Speedup**: 3-5x

            ### Scenario 3: Low Flux Conditions
            - **Flux**: 1×10⁵ n/s
            - **Time per Pattern**: 20 s
            - **Target Precision**: 0.01 Å
            - **Expected Speedup**: 2-4x
            """)


if __name__ == "__main__":
    main()
