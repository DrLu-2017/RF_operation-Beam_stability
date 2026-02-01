"""
Double RF System Analytical Dashboard
Analytical analysis of double RF cavity systems for beam stability.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.presets import get_preset, get_preset_names
from utils.config_manager import ConfigManager

# Page configuration
st.set_page_config(
    page_title="Double RF System - ALBuMS",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Double RF System Analytical Dashboard")
st.markdown("Analytical analysis of double RF cavity systems for longitudinal beam stability.")

# Initialize config manager
config_mgr = ConfigManager()

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Preset selection
    preset_names = get_preset_names()
    selected_preset = st.selectbox(
        "Select Preset",
        preset_names,
        key="double_rf_preset"
    )
    
    # Load preset configuration
    preset = get_preset(selected_preset)

# Main configuration tabs
tab1, tab2, tab3 = st.tabs(["⚙️ Machine Parameters", "🔷 Main Cavity", "🔶 Harmonic Cavity"])

with tab1:
    st.subheader("Machine Parameters")
    col1, col2 = st.columns(2)
    
    # Get ring parameters from preset
    ring_params = preset.get("ring", {})
    
    with col1:
        energy = st.number_input(
            "Beam Energy (GeV)",
            min_value=0.1,
            max_value=10.0,
            value=float(ring_params.get("energy", 2.75)),
            step=0.01,
            key="energy_double_rf",
            help="Beam energy in GeV"
        )
        
        circumference = st.number_input(
            "Ring Circumference (m)",
            min_value=10.0,
            max_value=1000.0,
            value=float(ring_params.get("circumference", 354.0)),
            step=0.1,
            key="circumference_double_rf"
        )
        
        harmonic_number = st.number_input(
            "Harmonic Number",
            min_value=1,
            max_value=2000,
            value=int(ring_params.get("harmonic_number", 416)),
            step=1,
            key="harmonic_number_double_rf"
        )
    
    with col2:
        momentum_compaction = st.number_input(
            "Momentum Compaction Factor (α)",
            min_value=0.0,
            max_value=0.1,
            value=float(ring_params.get("momentum_compaction", 1.06e-4)),
            format="%.2e",
            key="momentum_compaction_double_rf"
        )
        
        beam_current = st.number_input(
            "Beam Current (mA)",
            min_value=0.0,
            max_value=1000.0,
            value=float(preset.get("current", 0.5)) * 1000,  # Convert A to mA
            step=10.0,
            key="beam_current_double_rf"
        )
        
        energy_loss = st.number_input(
            "Energy Loss per Turn (keV)",
            min_value=0.0,
            max_value=10000.0,
            value=float(ring_params.get("energy_loss_per_turn", 0.000743)) * 1e6,  # Convert GeV to keV
            step=1.0,
            key="energy_loss_double_rf",
            help="Energy loss per turn in keV"
        )

with tab2:
    st.subheader("Main Cavity Parameters")
    col1, col2 = st.columns(2)
    
    # Get main cavity parameters from preset
    mc_params = preset.get("main_cavity", {})
    
    with col1:
        main_voltage = st.number_input(
            "Main Cavity Voltage (MV)",
            min_value=0.0,
            max_value=10.0,
            value=float(mc_params.get('voltage', 1.7)),
            step=0.01,
            key="main_voltage_double_rf"
        )
        
        main_frequency = st.number_input(
            "Main Cavity Frequency (MHz)",
            min_value=10.0,
            max_value=1000.0,
            value=float(mc_params.get('frequency', 352.2)),
            step=0.1,
            key="main_frequency_double_rf"
        )
        
        main_ncav = st.number_input(
            "Number of Main Cavities",
            min_value=1,
            max_value=20,
            value=int(mc_params.get('Ncav', 1)),
            step=1,
            key="main_ncav_double_rf"
        )
    
    with col2:
        main_r_over_q = st.number_input(
            "Main Cavity R/Q (Ω)",
            min_value=0.0,
            max_value=1000.0,
            value=float(mc_params.get('R_over_Q', 140.0)),
            step=1.0,
            key="main_r_over_q_double_rf"
        )
        
        main_q_loaded = st.number_input(
            "Main Cavity QL",
            min_value=100.0,
            max_value=100000.0,
            value=float(mc_params.get('QL', mc_params.get('Q', 20000.0))),
            step=100.0,
            key="main_q_loaded_double_rf"
        )

with tab3:
    st.subheader("Harmonic Cavity Parameters")
    col1, col2 = st.columns(2)
    
    # Get harmonic cavity parameters from preset
    hc_params = preset.get("harmonic_cavity", {})
    
    # Calculate harmonic multiplier from frequencies
    mc_freq = float(mc_params.get('frequency', 352.2))
    hc_freq = float(hc_params.get('frequency', 1408.8))
    default_multiplier = int(round(hc_freq / mc_freq)) if mc_freq > 0 else 4
    
    with col1:
        harmonic_voltage = st.number_input(
            "Harmonic Cavity Voltage (MV)",
            min_value=0.0,
            max_value=10.0,
            value=float(hc_params.get('voltage', 0.35)),
            step=0.01,
            key="harmonic_voltage_double_rf"
        )
        
        harmonic_multiplier = st.number_input(
            "Harmonic Number Multiplier",
            min_value=1,
            max_value=10,
            value=int(hc_params.get('harmonic_number', default_multiplier)),
            step=1,
            key="harmonic_multiplier_double_rf",
            help="Multiplier for harmonic cavity frequency (typically 3 or 4)"
        )
        
        harmonic_ncav = st.number_input(
            "Number of Harmonic Cavities",
            min_value=1,
            max_value=20,
            value=int(hc_params.get('Ncav', 1)),
            step=1,
            key="harmonic_ncav_double_rf"
        )
    
    with col2:
        harmonic_r_over_q = st.number_input(
            "Harmonic Cavity R/Q (Ω)",
            min_value=0.0,
            max_value=1000.0,
            value=float(hc_params.get('R_over_Q', 29.6)),
            step=1.0,
            key="harmonic_r_over_q_double_rf"
        )
        
        harmonic_q_loaded = st.number_input(
            "Harmonic Cavity QL",
            min_value=100.0,
            max_value=100000.0,
            value=float(hc_params.get('Q', 31000.0)),
            step=100.0,
            key="harmonic_q_loaded_double_rf"
        )
        
        psi = st.number_input(
            "Phase Offset ψ (degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=180.0,
            step=1.0,
            key="psi_double_rf",
            help="Phase offset between main and harmonic cavities"
        )

# Analysis section
st.markdown("---")
st.header("📊 Analysis Results")

# Calculate derived parameters
c = 299792458  # speed of light in m/s
harmonic_frequency = main_frequency * harmonic_multiplier
revolution_frequency = c / circumference * 1e-6  # in MHz
synchrotron_frequency = 0.0  # Placeholder for calculation

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Harmonic Frequency", f"{harmonic_frequency:.2f} MHz")
    st.metric("Revolution Frequency", f"{revolution_frequency:.6f} MHz")

with col2:
    st.metric("Main RF Frequency", f"{main_frequency:.2f} MHz")
    st.metric("Energy Loss", f"{energy_loss:.1f} keV")

with col3:
    st.metric("Beam Energy", f"{energy:.2f} GeV")
    st.metric("Beam Current", f"{beam_current:.1f} mA")

with col4:
    st.metric("Harmonic Number", f"{harmonic_number}")
    st.metric("Momentum Compaction", f"{momentum_compaction:.2e}")

# Stability analysis
st.markdown("---")
st.subheader("🔬 Stability Analysis")

if st.button("Run Stability Analysis", type="primary"):
    with st.spinner("Calculating stability regions..."):
        # Placeholder for actual stability calculation
        st.info("Stability analysis requires mbtrack2 backend. This is a placeholder.")
        
        # Create example plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Example stability diagram
        psi_range = np.linspace(-180, 180, 100)
        current_range = np.linspace(0, 1000, 100)
        PSI, CURRENT = np.meshgrid(psi_range, current_range)
        
        # Placeholder stability data
        stability = np.sin(PSI * np.pi / 180) * np.exp(-CURRENT / 500)
        
        contour = ax.contourf(PSI, CURRENT, stability, levels=20, cmap='RdYlGn')
        ax.set_xlabel('Phase Offset ψ (degrees)', fontsize=12)
        ax.set_ylabel('Beam Current (mA)', fontsize=12)
        ax.set_title('Stability Diagram (Example)', fontsize=14, fontweight='bold')
        plt.colorbar(contour, ax=ax, label='Stability Metric')
        
        st.pyplot(fig)
        plt.close()

# Information box
with st.expander("ℹ️ About Double RF Systems"):
    st.markdown("""
    ### Double RF System Configuration
    
    A double RF system uses two sets of cavities operating at different frequencies:
    
    - **Main Cavity**: Provides the primary acceleration voltage
    - **Harmonic Cavity**: Operates at a harmonic of the main frequency (typically 3× or 4×)
    
    ### Benefits
    
    - **Bunch Lengthening**: Increases Touschek lifetime
    - **Improved Stability**: Can suppress certain instabilities
    - **Flexible Operation**: Allows tuning of bunch shape and length
    
    ### Key Parameters
    
    - **ψ (Psi)**: Phase offset between main and harmonic cavities
    - **Voltage Ratio**: Ratio of harmonic to main cavity voltage
    - **R/Q**: Shunt impedance per unit Q-factor
    - **QL**: Loaded quality factor
    """)

# Save configuration button
if st.button("💾 Save Configuration"):
    config_data = {
        'energy': energy,
        'circumference': circumference,
        'harmonic_number': harmonic_number,
        'momentum_compaction': momentum_compaction,
        'beam_current': beam_current,
        'energy_loss_kev': energy_loss,
        'main_voltage': main_voltage,
        'main_frequency': main_frequency,
        'main_ncav': main_ncav,
        'main_r_over_q': main_r_over_q,
        'main_q_loaded': main_q_loaded,
        'harmonic_voltage': harmonic_voltage,
        'harmonic_multiplier': harmonic_multiplier,
        'harmonic_ncav': harmonic_ncav,
        'harmonic_r_over_q': harmonic_r_over_q,
        'harmonic_q_loaded': harmonic_q_loaded,
        'psi': psi
    }
    
    # Update session state
    for key, value in config_data.items():
        st.session_state[key] = value
    
    st.success("✅ Configuration saved to session!")
