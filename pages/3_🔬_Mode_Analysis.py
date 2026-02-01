"""
Mode Analysis Page
Analyze Robinson modes and track instabilities.
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

from utils.presets import get_preset, get_preset_names
from utils.albums_wrapper import (
    create_ring_from_params,
    create_cavity_from_params,
    analyze_robinson_modes
)
from utils.visualization import plot_mode_frequencies, plot_growth_rates

st.set_page_config(page_title="Mode Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Robinson Mode Analysis")
st.markdown("Track Robinson modes and analyze instabilities across parameter ranges.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Callback to update session state when preset changes
    def update_mode_analysis_from_preset():
        name = st.session_state.mode_preset_select
        new_preset = get_preset(name)
        if new_preset:
            # Update Ring parameters
            ring = new_preset.get("ring", {})
            st.session_state.mode_circumference = float(ring.get("circumference", 0))
            st.session_state.mode_energy = float(ring.get("energy", 0))
            st.session_state.mode_momentum = float(ring.get("momentum_compaction", 0))
            st.session_state.mode_eloss = float(ring.get("energy_loss_per_turn", 0)) * 1e6  # GeV to keV
            st.session_state.mode_harmonic = int(ring.get("harmonic_number", 0))
            st.session_state.mode_damping = float(ring.get("damping_time", 0))
            st.session_state.mode_current = float(new_preset.get("current", 0.2))
            
            # Update Main Cavity parameters
            mc = new_preset.get("main_cavity", {})
            st.session_state.mode_mc_voltage = float(mc.get("voltage", 0))
            st.session_state.mode_mc_freq = float(mc.get("frequency", 0))
            st.session_state.mode_mc_harm = int(mc.get("harmonic", 0))
            st.session_state.mode_mc_q = float(mc.get("Q", 0))
            st.session_state.mode_mc_roq = float(mc.get("R_over_Q", 0))
            
            # Update Harmonic Cavity parameters
            hc = new_preset.get("harmonic_cavity", {})
            st.session_state.mode_hc_voltage = float(hc.get("voltage", 0))
            st.session_state.mode_hc_freq = float(hc.get("frequency", 0))
            st.session_state.mode_hc_harm = int(hc.get("harmonic", 0))
            st.session_state.mode_hc_q = float(hc.get("Q", 0))
            st.session_state.mode_hc_roq = float(hc.get("R_over_Q", 0))
            
            # Update scan parameters
            scan_params = new_preset.get("scan_params", {})
            st.session_state.mode_psi_min = float(scan_params.get("psi_min", 1.0))
            st.session_state.mode_psi_max = float(scan_params.get("psi_max", 180.0))
            st.session_state.mode_psi_points = int(scan_params.get("psi_points", 50))
    
    preset_names = get_preset_names()
    default_index = preset_names.index("SOLEIL II") if "SOLEIL II" in preset_names else 0
    
    preset_name = st.selectbox(
        "Select Preset",
        options=preset_names,
        index=default_index,
        key="mode_preset_select",
        on_change=update_mode_analysis_from_preset
    )
    
    preset = get_preset(preset_name)
    
    st.markdown("---")
    st.markdown("### Analysis Settings")
    
    method = st.selectbox(
        "Solution Method",
        options=["Venturini", "Bosch", "Alves"],
        index=0
    )
    
    passive_hc = st.checkbox(
        "Passive Harmonic Cavity",
        value=preset.get("passive_hc", True)
    )

# Main content
tab1, tab2, tab3 = st.tabs(["⚙️ Parameters", "▶️ Run Analysis", "📈 Results"])

with tab1:
    st.header("Configuration Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ring Parameters")
        ring_params = preset["ring"]
        
        # Use session state values if available, otherwise use preset defaults
        circumference = st.number_input(
            "Circumference (m)",
            value=float(st.session_state.get("mode_circumference", ring_params["circumference"])),
            min_value=1.0,
            max_value=10000.0,
            format="%.2f",
            key="mode_circumference"
        )
        
        energy = st.number_input(
            "Energy (GeV)",
            value=float(st.session_state.get("mode_energy", ring_params["energy"])),
            min_value=0.1,
            max_value=10.0,
            format="%.3f",
            key="mode_energy"
        )
        
        momentum_compaction = st.number_input(
            "Momentum Compaction",
            value=float(st.session_state.get("mode_momentum", ring_params["momentum_compaction"])),
            min_value=0.0,
            max_value=0.1,
            format="%.6f",
            key="mode_momentum"
        )
        
        energy_loss = st.number_input(
            "Energy Loss per Turn (keV)",
            value=float(st.session_state.get("mode_eloss", ring_params["energy_loss_per_turn"] * 1e6)),
            min_value=0.0,
            max_value=10000.0,
            format="%.2f",
            key="mode_eloss"
        )
        
        harmonic_number = st.number_input(
            "Harmonic Number",
            value=int(st.session_state.get("mode_harmonic", ring_params["harmonic_number"])),
            min_value=1,
            max_value=10000,
            key="mode_harmonic"
        )
        
        damping_time = st.number_input(
            "Damping Time (s)",
            value=float(st.session_state.get("mode_damping", ring_params["damping_time"])),
            min_value=0.0001,
            max_value=1.0,
            format="%.6f",
            key="mode_damping"
        )
        
        current = st.number_input(
            "Beam Current (A)",
            value=float(st.session_state.get("mode_current", preset.get("current", 0.2))),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="mode_current"
        )
    
    with col2:
        st.subheader("Main Cavity")
        mc_params = preset["main_cavity"]
        
        mc_voltage = st.number_input(
            "Voltage (MV)",
            value=float(st.session_state.get("mode_mc_voltage", mc_params["voltage"])),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="mode_mc_voltage"
        )
        
        mc_frequency = st.number_input(
            "Frequency (MHz)",
            value=float(st.session_state.get("mode_mc_freq", mc_params["frequency"])),
            min_value=1.0,
            max_value=5000.0,
            format="%.3f",
            key="mode_mc_freq"
        )
        
        mc_harmonic = st.number_input(
            "Harmonic",
            value=int(st.session_state.get("mode_mc_harm", mc_params["harmonic"])),
            min_value=1,
            max_value=10000,
            key="mode_mc_harm"
        )
        
        mc_q = st.number_input(
            "Quality Factor Q",
            value=float(st.session_state.get("mode_mc_q", mc_params["Q"])),
            min_value=100.0,
            max_value=1000000.0,
            format="%.0f",
            key="mode_mc_q"
        )
        
        mc_roq = st.number_input(
            "R/Q (Ω)",
            value=float(st.session_state.get("mode_mc_roq", mc_params["R_over_Q"])),
            min_value=1.0,
            max_value=1000.0,
            format="%.1f",
            key="mode_mc_roq"
        )
        
        st.subheader("Harmonic Cavity")
        hc_params = preset["harmonic_cavity"]
        
        hc_voltage = st.number_input(
            "Voltage (MV)",
            value=float(st.session_state.get("mode_hc_voltage", hc_params["voltage"])),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="mode_hc_voltage"
        )
        
        hc_frequency = st.number_input(
            "Frequency (MHz)",
            value=float(st.session_state.get("mode_hc_freq", hc_params["frequency"])),
            min_value=1.0,
            max_value=5000.0,
            format="%.3f",
            key="mode_hc_freq"
        )
        
        hc_harmonic = st.number_input(
            "Harmonic",
            value=int(st.session_state.get("mode_hc_harm", hc_params["harmonic"])),
            min_value=1,
            max_value=10000,
            key="mode_hc_harm"
        )
        
        hc_q = st.number_input(
            "Quality Factor Q",
            value=float(st.session_state.get("mode_hc_q", hc_params["Q"])),
            min_value=100.0,
            max_value=1000000.0,
            format="%.0f",
            key="mode_hc_q"
        )
        
        hc_roq = st.number_input(
            "R/Q (Ω)",
            value=float(st.session_state.get("mode_hc_roq", hc_params["R_over_Q"])),
            min_value=1.0,
            max_value=1000.0,
            format="%.1f",
            key="mode_hc_roq"
        )

with tab2:
    st.header("Run Mode Analysis")
    
    st.info("""
    This analysis tracks Robinson modes across a range of harmonic cavity phases,
    showing mode frequencies, growth rates, and potential instabilities.
    """)
    
    st.subheader("Psi Range")
    col1, col2, col3 = st.columns(3)
    
    # Get scan parameters from preset or session state
    scan_params = preset.get("scan_params", {})
    default_psi_min = st.session_state.get("mode_psi_min", scan_params.get("psi_min", 1.0))
    default_psi_max = st.session_state.get("mode_psi_max", scan_params.get("psi_max", 180.0))
    default_psi_points = st.session_state.get("mode_psi_points", scan_params.get("psi_points", 50))
    
    with col1:
        psi_min = st.number_input(
            "Min (degrees)",
            value=float(default_psi_min),
            min_value=0.1,
            max_value=180.0,
            format="%.1f",
            step=0.1,
            key="mode_psi_min"
        )
    
    with col2:
        psi_max = st.number_input(
            "Max (degrees)",
            value=float(default_psi_max),
            min_value=-180.0,
            max_value=180.0,
            format="%.1f",
            key="mode_psi_max"
        )
    
    with col3:
        psi_points = st.slider(
            "Number of Points",
            min_value=10,
            max_value=100,
            value=int(default_psi_points),
            key="mode_psi_points"
        )
    
    st.markdown("---")
    
    if st.button("🔬 Run Analysis", type="primary", width='stretch'):
        if psi_min >= psi_max:
            st.error("❌ Min Psi must be less than Max Psi!")
        else:
            with st.spinner("Analyzing Robinson modes... This may take a few minutes."):
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
                    
                    # Run mode analysis
                    result = analyze_robinson_modes(
                        ring, main_cavity, harmonic_cavity,
                        current=current,
                        psi_range=(psi_min, psi_max, psi_points),
                        method=method,
                        passive_hc=passive_hc
                    )
                    
                    # Store results
                    st.session_state['mode_results'] = result
                    
                    if result['success']:
                        st.success("✅ Analysis completed successfully!")
                        st.info("Switch to the 'Results' tab to view the mode tracking.")
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())


with tab3:
    st.header("Mode Analysis Results")
    
    if 'mode_results' in st.session_state:
        result = st.session_state['mode_results']
        
        if result['success']:
            st.success("✅ Mode Analysis Results")
            
            st.info("📊 Mode tracking visualization will be displayed here.")
            
            # Display psi range
            st.markdown("### Analysis Range")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Psi Range", f"{result['psi_vals'][0]:.1f}° to {result['psi_vals'][-1]:.1f}°")
            with col2:
                st.metric("Number of Points", len(result['psi_vals']))
            
            # Placeholder for visualization
            st.markdown("### Mode Frequencies")
            st.info("💡 Mode frequency plots will show how Robinson modes evolve with harmonic cavity phase.")
            
            st.markdown("### Growth Rates")
            st.info("💡 Growth rate plots will identify unstable regions and mode coupling phenomena.")
            
        else:
            st.error(f"Last analysis failed: {result.get('error', 'Unknown error')}")
    else:
        st.info("👈 Configure parameters and run analysis to see mode tracking results here.")
