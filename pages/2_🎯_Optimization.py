"""
Optimization Page
Optimize R-factor to maximize Touschek lifetime.
"""
import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add local mbtrack2 if available
mbtrack2_local = project_root / "mbtrack2-stable"
if mbtrack2_local.exists():
    sys.path.insert(0, str(mbtrack2_local))

# Add local pycolleff if available
pycolleff_local = project_root / "collective_effects-master" / "pycolleff"
if pycolleff_local.exists():
    sys.path.insert(0, str(pycolleff_local))

from utils.presets import get_preset, get_preset_names, load_config_with_source
from utils.config_manager import ConfigManager
from utils.config_utils import initialize_session_config, get_saved_configs_for_accelerator, load_current_config
from utils.albums_wrapper import (
    create_ring_from_params,
    create_cavity_from_params,
    run_optimization
)
from utils.visualization import plot_optimization_result

st.set_page_config(page_title="Optimization", page_icon="🎯", layout="wide")

st.title("🎯 R-Factor Optimization")
st.markdown("Find optimal harmonic cavity phase to maximize Touschek lifetime.")

# Initialize session configuration
initialize_session_config()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Initialize preset with default or current
    # Default to first preset if nothing selected
    if 'preset_select' not in st.session_state:
        st.session_state.preset_select = get_preset_names()[0]
    
    # Callback to update session state when preset changes (Sync with Parameter Scans)
    def update_inputs_from_preset():
        name = st.session_state.preset_select
        new_preset, _ = load_config_with_source(name)
        if new_preset:
            # Update Ring parameters
            ring = new_preset.get("ring", {})
            st.session_state.ring_circumference = float(ring.get("circumference", 0))
            st.session_state.ring_energy = float(ring.get("energy", 0))
            st.session_state.ring_momentum = float(ring.get("momentum_compaction", 0))
            st.session_state.ring_eloss_kev = float(ring.get("energy_loss_per_turn", 0)) * 1e6  # GeV to keV
            st.session_state.ring_harmonic = int(ring.get("harmonic_number", 0))
            st.session_state.ring_damping = float(ring.get("damping_time", 0))
            
            # Update Main Cavity keys
            mc = new_preset.get("main_cavity", {})
            st.session_state.mc_voltage = float(mc.get("voltage", 0))
            st.session_state.mc_freq = float(mc.get("frequency", 0))
            st.session_state.mc_harm = int(mc.get("harmonic", 0))
            st.session_state.mc_q = float(mc.get("Q", 0))
            st.session_state.mc_roq = float(mc.get("R_over_Q", 0))
            
            # Update Harmonic Cavity keys
            hc = new_preset.get("harmonic_cavity", {})
            st.session_state.hc_voltage = float(hc.get("voltage", 0))
            st.session_state.hc_q = float(hc.get("Q", 0))
            st.session_state.hc_roq = float(hc.get("R_over_Q", 0))
            
            # Update Harmonic Frequency directly
            st.session_state.hc_freq = float(hc.get("frequency", 0))
            st.session_state.hc_harm = int(hc.get("harmonic", 0))
            
            # Calculate and update ratio (for Parameter Scans compatibility)
            mc_f = float(mc.get("frequency", 1))
            hc_f = float(hc.get("frequency", 0))
            if mc_f > 0:
                st.session_state.hc_ratio = float(round(hc_f / mc_f))
            else:
                st.session_state.hc_ratio = 4.0


    # Preset Selection
    preset_names = get_preset_names()
    # Ensure current selection is valid
    current_idx = 0
    if st.session_state.preset_select in preset_names:
        current_idx = preset_names.index(st.session_state.preset_select)
        
    preset_name = st.selectbox(
        "Select Preset",
        options=preset_names,
        index=current_idx,
        key="preset_select",
        on_change=update_inputs_from_preset
    )
    
    # Load the config
    preset, source_config = load_config_with_source(preset_name)
    
    if source_config:
        st.info(f"📝 Based on: **{source_config}**")
    
    st.markdown("---")
    st.markdown("### Optimization Settings")
    
    method = st.selectbox(
        "Solution Method",
        options=["Venturini", "Bosch", "Alves"],
        index=0
    )
    
    equilibrium_only = st.checkbox(
        "Equilibrium Only",
        value=False,
        help="Faster computation, only solves equilibrium distribution"
    )

# Main content
tab1, tab2, tab3 = st.tabs(["⚙️ Parameters", "🎯 Optimize", "📊 Results"])

with tab1:
    st.header("Configuration Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ring Parameters")
        ring_params = preset["ring"]
        
        # Use session state values if available, otherwise use preset defaults
        circumference = st.number_input(
            "Circumference (m)",
            value=float(st.session_state.get("ring_circumference", ring_params["circumference"])),
            min_value=1.0,
            max_value=10000.0,
            format="%.2f",
            key="ring_circumference"
        )
        
        energy = st.number_input(
            "Energy (GeV)",
            value=float(st.session_state.get("ring_energy", ring_params["energy"])),
            min_value=0.1,
            max_value=10.0,
            format="%.3f",
            key="ring_energy"
        )
        
        momentum_compaction = st.number_input(
            "Momentum Compaction",
            value=float(st.session_state.get("ring_momentum", ring_params["momentum_compaction"])),
            min_value=0.0,
            max_value=0.1,
            format="%.6f",
            key="ring_momentum"
        )
        
        energy_loss = st.number_input(
            "Energy Loss per Turn (keV)",
            value=float(st.session_state.get("ring_eloss_kev", ring_params["energy_loss_per_turn"] * 1e6)),
            min_value=0.0,
            max_value=10000.0,
            format="%.2f",
            key="ring_eloss_kev"
        )
        
        harmonic_number = st.number_input(
            "Harmonic Number",
            value=int(st.session_state.get("ring_harmonic", ring_params["harmonic_number"])),
            min_value=1,
            max_value=10000,
            key="ring_harmonic"
        )
        
        damping_time = st.number_input(
            "Damping Time (s)",
            value=float(st.session_state.get("ring_damping", ring_params["damping_time"])),
            min_value=0.0001,
            max_value=1.0,
            format="%.6f",
            key="ring_damping"
        )
        
        current = st.number_input(
            "Beam Current (A)",
            value=float(preset.get("current", 0.2)),
            min_value=0.0,
            max_value=10.0,
            format="%.3f"
        )
    
    with col2:
        st.subheader("Main Cavity")
        mc_params = preset["main_cavity"]
        
        mc_voltage = st.number_input(
            "Voltage (MV)",
            value=float(st.session_state.get("mc_voltage", mc_params["voltage"])),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="mc_voltage"
        )
        
        mc_frequency = st.number_input(
            "Frequency (MHz)",
            value=float(st.session_state.get("mc_freq", mc_params["frequency"])),
            min_value=1.0,
            max_value=5000.0,
            format="%.3f",
            key="mc_freq"
        )
        
        mc_harmonic = st.number_input(
            "Harmonic",
            value=int(st.session_state.get("mc_harm", mc_params["harmonic"])),
            min_value=1,
            max_value=10000,
            key="mc_harm"
        )
        
        mc_q = st.number_input(
            "Quality Factor Q",
            value=float(st.session_state.get("mc_q", mc_params["Q"])),
            min_value=100.0,
            max_value=1000000.0,
            format="%.0f",
            key="mc_q"
        )
        
        mc_roq = st.number_input(
            "R/Q (Ω)",
            value=float(st.session_state.get("mc_roq", mc_params["R_over_Q"])),
            min_value=1.0,
            max_value=1000.0,
            format="%.1f",
            key="mc_roq"
        )
        
        st.subheader("Harmonic Cavity")
        hc_params = preset["harmonic_cavity"]
        
        hc_voltage = st.number_input(
            "Voltage (MV)",
            value=float(st.session_state.get("hc_voltage", hc_params["voltage"])),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="hc_voltage"
        )
        
        hc_frequency = st.number_input(
            "Frequency (MHz)",
            value=float(st.session_state.get("hc_freq", hc_params["frequency"])),
            min_value=1.0,
            max_value=5000.0,
            format="%.3f",
            key="hc_freq"
        )
        
        hc_harmonic = st.number_input(
            "Harmonic",
            value=int(st.session_state.get("hc_harm", hc_params["harmonic"])),
            min_value=1,
            max_value=10000,
            key="hc_harm"
        )
        
        hc_q = st.number_input(
            "Quality Factor Q",
            value=float(st.session_state.get("hc_q", hc_params["Q"])),
            min_value=100.0,
            max_value=1000000.0,
            format="%.0f",
            key="hc_q"
        )
        
        hc_roq = st.number_input(
            "R/Q (Ω)",
            value=float(st.session_state.get("hc_roq", hc_params["R_over_Q"])),
            min_value=1.0,
            max_value=1000.0,
            format="%.1f",
            key="hc_roq"
        )

with tab2:
    st.header("Run Optimization")
    
    st.info("""
    The optimization will search for the harmonic cavity phase (psi) that maximizes the R-factor,
    which is related to the Touschek lifetime enhancement.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Initial Guess")
        psi0 = st.number_input(
            "Initial Psi (degrees)",
            value=90.0,
            min_value=-180.0,
            max_value=180.0,
            format="%.1f",
            help="Starting point for optimization"
        )
    
    with col2:
        st.subheader("Search Bounds")
        psi_min = st.number_input(
            "Min Psi (degrees)",
            value=60.0,
            min_value=0.1,
            max_value=180.0,
            format="%.1f",
            step=0.1
        )
        psi_max = st.number_input(
            "Max Psi (degrees)",
            value=90.0,
            min_value=-180.0,
            max_value=180.0,
            format="%.1f"
        )
    
    st.markdown("---")
    
    if st.button("🎯 Run Optimization", type="primary", width='stretch'):
        if psi_min >= psi_max:
            st.error("❌ Min Psi must be less than Max Psi!")
        elif psi0 < psi_min or psi0 > psi_max:
            st.warning("⚠️ Initial guess is outside bounds. Continuing anyway...")
        
        with st.spinner("Running optimization... This may take a few minutes."):
            try:
                # Create ring and cavities (convert keV to GeV for backend)
                ring = create_ring_from_params(
                    circumference, energy, momentum_compaction,
                    energy_loss / 1e6, harmonic_number, damping_time  # Convert keV to GeV (1 GeV = 1e6 keV)
                )
                
                main_cavity = create_cavity_from_params(
                    mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq
                )
                
                harmonic_cavity = create_cavity_from_params(
                    hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq
                )
                
                # Run optimization
                result = run_optimization(
                    ring, main_cavity, harmonic_cavity,
                    current=current,
                    psi0=psi0,
                    bounds=(psi_min, psi_max),
                    method=method,
                    equilibrium_only=equilibrium_only
                )
                
                # Store results
                st.session_state['opt_results'] = result
                
                if result['success']:
                    st.success("✅ Optimization completed successfully!")
                    st.info("Switch to the 'Results' tab to view the optimal parameters.")
                else:
                    st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

with tab3:
    st.header("Optimization Results")
    
    if 'opt_results' in st.session_state:
        result = st.session_state['opt_results']
        
        if result['success']:
            st.success("✅ Optimization Results")
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Initial Psi",
                    f"{result['psi0']:.2f}°",
                    help="Starting point for optimization"
                )
            
            with col2:
                st.metric(
                    "Optimal Psi",
                    f"{result['optimal_psi']:.2f}°",
                    delta=f"{result['optimal_psi'] - result['psi0']:.2f}°",
                    help="Optimized harmonic cavity phase"
                )
            
            with col3:
                if 'r_factor' in result:
                    st.metric(
                        "R-Factor",
                        f"{result['r_factor']:.4f}",
                        help="Touschek lifetime enhancement factor"
                    )
            
            st.markdown("---")
            
            # Visualization
            st.subheader("Optimization Visualization")
            fig = plot_optimization_result(
                psi0=result['psi0'],
                optimal_psi=result['optimal_psi'],
                bounds=(psi_min, psi_max) if 'psi_min' in locals() else (0, 180),
                r_factor=result.get('r_factor')
            )
            st.plotly_chart(fig, width='stretch')
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            st.info(f"""
            **Optimal Configuration:**
            - Set harmonic cavity phase to **{result['optimal_psi']:.2f}°**
            - Expected R-factor: **{result.get('r_factor', 'N/A')}**
            
            This configuration should maximize the Touschek lifetime for the given parameters.
            """)
            
        else:
            st.error(f"Last optimization failed: {result.get('error', 'Unknown error')}")
    else:
        st.info("👈 Configure parameters and run optimization to see results here.")
