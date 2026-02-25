"""
Double RF System Analytical Dashboard
Professional RF cavity analysis with comprehensive physics calculations.
Integrated from cavity_operation/rf_system_pro.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.presets import get_preset, get_preset_names
from utils.config_manager import ConfigManager
from utils.rf_calculations import (
    run_physics_engine,
    calculate_harmonic_cavity_params,
    calculate_fundamental_cavity_params,
    calculate_rf_feedback_params,
    calculate_potential_well
)
from utils.ui_utils import fmt, render_display_settings
from cavity_operation.clbi_calculator import DoubleRF_CLBI
import scipy.constants as const
from utils.hofmann_model import DoubleRFSystem as HofmannModel
from utils.rf_calibration import RFVoltageCalibrator, DecayPowerCalibrator
from utils.venturini_model import (
    flat_potential_conditions,
    passive_cavity_beam_loading,
    required_detuning_flat_potential,
    robinson_stability,
    bunch_lengthening_factor,
    validate_against_app,
    classify_operating_regime
)
from utils.pedersen_model import (
    drfb_impedance_reduction,
    robinson_limit,
    double_rf_stability_with_feedback,
    scan_drfb_gain,
    scan_current_stability,
    impedance_spectrum,
    operational_guidelines,
    nyquist_contour,
    nyquist_multi_current,
    nyquist_hc_passive,
    bode_plot_data
)
from utils.bosch_model import (
    bosch_analysis,
    scan_current_bosch,
    equilibrium_phases_active,
    synchrotron_frequency_spread as bosch_synch_spread,
    coupled_bunch_growth_rate as bosch_cb_growth,
    robinson_two_cavity as bosch_robinson,
)
from utils.jacob_model import (
    jacob_robinson_analysis,
    scan_current_jacob
)
from utils.alves_model import alves_analysis, ALBUMS_AVAILABLE
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Double RF System - Beam",
    page_icon="🔧",
    layout="wide"
)

if not st.session_state.get("authentication_status"):
    st.info("Please login from the Home page.")
    st.stop()

st.title("🔧 RF System Analytical Dashboard")
st.markdown("Professional analysis of double RF cavity systems with beam analysis")

# Initialize config manager
config_mgr = ConfigManager()

# Sidebar for configuration
with st.sidebar:
    st.markdown("<div style='text-align: center;'><h1 style='color: #4facfe;'>DRFB</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Navigation")
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/0_🔧_Double_RF_System.py", label="Double RF System", icon="🔧")
    st.page_link("pages/1_📈_Semi_Analytic.py", label="Semi-Analytic Tools", icon="📈")
    st.page_link("pages/2_🚀_MBTrack2_Remote.py", label="MBTrack2 Remote Job", icon="🚀")

    st.markdown("---")
    
    st.header("⚙️ Configuration")
    
    # Preset selection with on_change callback
    def on_preset_change():
        """Callback to update all parameters when preset changes"""
        preset = get_preset(st.session_state.double_rf_preset)
        
        # Update ring parameters
        ring_params = preset.get("ring", {})
        st.session_state.energy_double_rf = float(ring_params.get("energy", 2.75))
        st.session_state.harmonic_number_double_rf = int(ring_params.get("harmonic_number", 416))
        st.session_state.u0_kev_double_rf = float(ring_params.get("energy_loss_per_turn", 0.000743)) * 1e6
        st.session_state.alpha_c_double_rf = float(ring_params.get("momentum_compaction", 4.4e-4))
        st.session_state.sigma_z0_double_rf = float(ring_params.get("sigma_z0", 15e-12))
        
        # Update main cavity parameters
        mc_params = preset.get("main_cavity", {})
        st.session_state.main_voltage_double_rf = float(mc_params.get('voltage', 1.7))
        st.session_state.main_frequency_double_rf = float(mc_params.get('frequency', 352.2))
        st.session_state.main_ncav_double_rf = int(mc_params.get('Ncav', 4))
        st.session_state.q0_fund_double_rf = float(mc_params.get('Q0', mc_params.get('Q', 35700)))
        st.session_state.q_ext_double_rf = float(mc_params.get('Q_ext', 6364))
        st.session_state.beta_fund_double_rf = float(mc_params.get('beta', 5.5))
        st.session_state.rshunt_mohm_double_rf = float(mc_params.get('Rs', 5.0))
        st.session_state.rf_gain_double_rf = float(mc_params.get('rf_feedback_gain', 1.3))
        st.session_state.tau_cav_us_double_rf = float(mc_params.get('tau_us', 4.87))
        
        # Update harmonic cavity parameters
        hc_params = preset.get("harmonic_cavity", {})
        st.session_state.harmonic_voltage_mode = "Optimized (Formula)"
        st.session_state.harmonic_ncav_double_rf = int(hc_params.get('Ncav', 3))
        st.session_state.qh0_double_rf = float(hc_params.get('Q0', hc_params.get('Q', 31000)))
        st.session_state.rhshunt_mohm_double_rf = float(hc_params.get('Rs', 0.92))
        st.session_state.beta_harm_double_rf = float(hc_params.get('beta', 0.0))
        st.session_state.tau_hcav_us_double_rf = float(hc_params.get('tau_us', 7.0))
        st.session_state.harmonic_ratio_double_rf = int(hc_params.get('harmonic_number', 4))
        
        # Update beam current
        st.session_state.beam_current_double_rf = float(preset.get("current", 0.5)) * 1000  # A to mA
    
    preset_names = get_preset_names()
    
    # Set default to SOLEIL II
    default_index = 0
    if "SOLEIL II" in preset_names:
        default_index = preset_names.index("SOLEIL II")
    
    selected_preset = st.selectbox(
        "Select Preset",
        preset_names,
        index=default_index,
        key="double_rf_preset",
        on_change=on_preset_change
    )
    
    # Load preset configuration
    preset = get_preset(selected_preset)
    
    # Operating Phase Selection
    st.subheader("🎯 Operating Phase")
    operation_phase = preset.get("operation_phase", "Phase 1")
    if "SOLEIL II" in selected_preset:
        phase_display = st.radio(
            "Phase:",
            options=["Phase 1", "Phase 2"],
            index=0 if operation_phase == "Phase 1" else 1,
            help="Phase 1: 787 keV (471+316) | Phase 2: 830 keV (471+359)"
        )
    else:
        phase_display = "N/A"
        st.info("Phase selection available for SOLEIL II presets")

    # Global UI settings
    render_display_settings()

# Main configuration sections
st.markdown("---")

# Get parameters from preset
ring_params = preset.get("ring", {})
mc_params = preset.get("main_cavity", {})
hc_params = preset.get("harmonic_cavity", {})

# Storage Ring Parameters
with st.expander("◇ Storage Ring Parameters", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        energy = st.number_input(
            "Energy E₀ (GeV)",
            min_value=0.1,
            max_value=10.0,
            value=float(ring_params.get("energy", 2.75)),
            step=0.01,
            key="energy_double_rf"
        )
        
        f0_mhz = st.number_input(
            "RF Frequency f₀ (MHz)",
            min_value=100.0,
            max_value=500.0,
            value=float(mc_params.get('frequency', 352.2)),
            step=0.0001,
            format="%.4f",
            key="main_frequency_double_rf"
        )
    
    with col2:
        h_rf = st.number_input(
            "Harmonic Number h",
            min_value=1,
            max_value=2000,
            value=int(ring_params.get("harmonic_number", 416)),
            step=1,
            key="harmonic_number_double_rf"
        )
        
        imax_ma = st.number_input(
            "Max Beam Current I_max (mA)",
            min_value=1.0,
            max_value=1000.0,
            value=500.0,
            step=10.0,
            key="imax_double_rf"
        )
        
        alpha_c = st.number_input(
            "Momentum Compaction α_c",
            min_value=1e-6,
            max_value=0.1,
            value=float(ring_params.get("momentum_compaction", 4.4e-4)),
            step=1e-6,
            format="%.2e",
            key="alpha_c_double_rf"
        )
    
    with col3:
        u0_kev = st.number_input(
            "Energy Loss U₀ (keV)",
            min_value=1.0,
            max_value=20000.0,
            value=float(ring_params.get("energy_loss_per_turn", 0.000743)) * 1e6,
            step=1.0,
            key="u0_kev_double_rf",
            help="Energy loss per turn in keV"
        )
        
        if "SOLEIL II" in selected_preset and phase_display != "N/A":
            if phase_display == "Phase 1":
                st.info("**Phase 1:** 471 keV (Synch.) + 316 keV (IDs) = **787 keV**")
            else:
                st.info("**Phase 2:** 471 keV (Synch.) + 359 keV (IDs) = **830 keV**")

# Fundamental Cavity Parameters
with st.expander("● Fundamental Cavities (n=1)", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ncav = st.slider(
            "Number of Cavities",
            min_value=1,
            max_value=4,
            value=int(mc_params.get('Ncav', 4)),
            key="main_ncav_double_rf"
        )
        
        vfcav_kv = st.number_input(
            "Total RF Voltage (kV)",
            min_value=10.0,
            max_value=30000.0,
            value=float(mc_params.get('voltage', 1.7)) * 1000,  # MV to kV
            step=10.0,
            key="v_rf_total_double_rf"
        )
        
        q0_fund = st.number_input(
            "Unloaded Q (Q₀)",
            min_value=1000,
            max_value=2000000000,
            value=int(mc_params.get('Q0', mc_params.get('Q', 35700))),
            step=100,
            key="q0_fund_double_rf"
        )
    
    with col2:
        qext_fund = st.number_input(
            "External Q (Q_ext)",
            min_value=1000,
            max_value=50000,
            value=int(mc_params.get('Q_ext', 6364)),
            step=100,
            key="q_ext_double_rf"
        )
        
        beta_fund = st.number_input(
            "Coupling Factor β",
            min_value=1.0,
            max_value=20.0,
            value=float(mc_params.get('beta', 5.5)),
            step=0.1,
            key="beta_fund_double_rf"
        )
        
        rshunt_mohm = st.number_input(
            "Shunt Impedance R (MΩ)",
            min_value=0.1,
            max_value=2000.0,
            value=float(mc_params.get('Rs', 5.0)),
            step=0.1,
            key="rshunt_mohm_double_rf"
        )
    
    with col3:
        tau_cav_us = st.number_input(
            "Cavity Decay Time τ (μs)",
            min_value=1.0,
            max_value=20.0,
            value=float(mc_params.get('tau_us', 4.87)),
            step=0.01,
            key="tau_cav_us_double_rf"
        )
        
        st.markdown("**RF Feedback Control:**")
        rf_gain = st.number_input(
            "RF Feedback Gain",
            min_value=0.0,
            max_value=5.0,
            value=float(mc_params.get('rf_feedback_gain', 1.3)),
            step=0.1,
            key="rf_gain_double_rf"
        )

# Harmonic Cavity Parameters
nh_harm = st.session_state.get('harmonic_ratio_double_rf', 4)
with st.expander(f"◐ Harmonic Cavities (n={nh_harm}, {f0_mhz*nh_harm/1000:.3f} GHz)", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nhcav = st.slider(
            "Number of Harmonic Cavities",
            min_value=1,
            max_value=3,
            value=int(hc_params.get('Ncav', 3)),
            key="harmonic_ncav_double_rf"
        )
        
        fh_calculated = f0_mhz * nh_harm / 1000  # Convert to GHz
        st.metric("Harmonic Number (n)", nh_harm, help="Fixed at 4th harmonic")
        st.metric("Harmonic Frequency (GHz)", f"{fh_calculated:.3f}", help=f"{f0_mhz} MHz × {nh_harm}")
    
    with col2:
        # Harmonic Voltage Mode Selection
        vh_mode = st.radio(
            "Harmonic Voltage Mode",
            ["Optimized (Formula)", "Manual Fixed Value"],
            key="harmonic_voltage_mode",
            help="Choose calculation method for harmonic voltage"
        )
        
        if vh_mode == "Optimized (Formula)":
            vh_auto_kv = None
            st.info(f"Formula: V_h,opt = √(V_c²/n² - U₀²/(n²-1))")
        else:
            vh_auto_kv = st.number_input(
                "Harmonic Voltage V_h (kV)",
                min_value=1.0,
                max_value=10000.0,
                value=400.0,
                step=10.0,
                key="vh_manual_double_rf"
            )
    
    with col3:
        qh0 = st.number_input(
            "Loaded Q (Q₀)",
            min_value=1000,
            max_value=2000000000,
            value=int(hc_params.get('Q0', hc_params.get('Q', 31000))),
            step=100,
            key="qh0_double_rf"
        )
        
        rhshunt_mohm = st.number_input(
            "Shunt Impedance R (MΩ)",
            min_value=0.01,
            max_value=2000.0,
            value=float(hc_params.get('Rs', 0.92)),
            step=0.01,
            key="rhshunt_mohm_double_rf"
        )
        
        beta_harm = st.number_input(
            "Coupling Factor β",
            min_value=0.0,
            max_value=10.0,
            value=float(hc_params.get('beta', 0.0)),
            step=0.1,
            key="beta_harm_double_rf"
        )
        
        tau_hcav_us = st.number_input(
            "Cavity Decay Time τ (μs)",
            min_value=1.0,
            max_value=20.0,
            value=float(hc_params.get('tau_us', 7.0)),
            step=0.1,
            key="tau_hcav_us_double_rf"
        )

# Beam & RF Settings
with st.expander("⚡ Beam & RF Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        current_input = st.slider(
            "Beam Current I (mA)",
            min_value=0.01,
            max_value=float(imax_ma),
            value=min(250.0, float(imax_ma)),
            step=1.0,
            key="beam_current_double_rf"
        )
    
    with col2:
        beta_fixed = st.number_input(
            "Fixed Coupling Factor β",
            min_value=1.0,
            max_value=10.0,
            value=float(mc_params.get('beta', 5.5)),
            step=0.1,
            key="beta_fixed_double_rf"
        )

# Calculate live results
res = run_physics_engine(
    current_input, vfcav_kv, beta_fixed, ncav, nhcav, nh_harm,
    rshunt_mohm, rhshunt_mohm, u0_kev, vh_auto_kv
)

# --- UI Dashboard ---
st.markdown("---")
st.header("📊 Analysis Results")

# Phase indicator with color coding
if "SOLEIL II" in selected_preset and phase_display != "N/A":
    if phase_display == "Phase 1":
        st.markdown(f"### 🟢 **{phase_display}** - U₀ = {u0_kev:.0f} keV (471 keV + 316 keV IDs)")
    else:
        st.markdown(f"### 🟠 **{phase_display}** - U₀ = {u0_kev:.0f} keV (471 keV + 359 keV IDs)")

st.write(f"**Energy:** {fmt(energy)} GeV | **Frequency:** {fmt(f0_mhz, 3)} MHz | **Harmonic:** n={nh_harm}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sync Phase φ_s", fmt(res['phi_s']) + " °")
col2.metric("Optimal β", fmt(res['beta_opt']))
col3.metric("Beam Power/Cav", fmt(res['p_beam']) + " kW")
col4.metric("Reflection |ρ|", fmt(res['rho'], 3))

# --- Generate Scans for Plots ---
i_range = np.linspace(0.01, imax_ma, 100)
scans = [run_physics_engine(i, vfcav_kv, beta_fixed, ncav, nhcav, nh_harm, 
                            rshunt_mohm, rhshunt_mohm, u0_kev, vh_auto_kv) for i in i_range]

# Calculate harmonic cavity parameters
h_params = calculate_harmonic_cavity_params(imax_ma, res['vh_opt'], nhcav, nh_harm, f0_mhz, rhshunt_mohm, qh0)

# Calculate fundamental cavity detuning
f_params = calculate_fundamental_cavity_params(imax_ma, vfcav_kv/ncav, res['phi_s_rad'], 
                                              f0_mhz, q0_fund, rshunt_mohm)

# Calculate RF feedback parameters
rf_params = calculate_rf_feedback_params(f0_mhz, q0_fund, beta_fund, rf_gain)

# --- THE COMPREHENSIVE ANALYSIS TABS [REORGANIZED] ---
tabs = st.tabs([
    "⚡ Cavity Operation",
    "🌀 Beam Physics",
    "🎮 Stability & Control",
    "📊 Detailed Data & Theory",
    "🛠️ Voltage Calibration"
])

# === Tab 1: Cavity Operation ===
with tabs[0]:
    st.subheader("Operation Parameters: Power, Detuning & Matching")
    st.caption("Physics Source: Based on SOLEIL II design SMath sheets and ESLS RF Workshop (2025) methodologies.")
    
    subtabs_op = st.tabs(["Main Power", "Harmonic Power", "Detuning", "Reflection"])
    
    with subtabs_op[0]: # Plot 1
        st.caption("Main Cavity Power vs Current")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=i_range, y=[s['p_inc'] for s in scans], name="Incident (Pi)", line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=i_range, y=[s['p_beam'] for s in scans], name="Beam (Pb)", line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=i_range, y=[s['p_ref'] for s in scans], name="Reflected (Pr)", line=dict(color='red')))
        fig1.update_layout(title="Main Cavity Power (Total)", xaxis_title="Current (mA)", yaxis_title="kW", template="plotly_white")
        st.plotly_chart(fig1, width='stretch')
        
    with subtabs_op[1]: # Plot 3
        st.caption("Harmonic Cavity Power vs Current")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=i_range, y=[s['ph_diss'] for s in scans], name="Ph_diss (per cavity)", line=dict(color='purple')))
        fig3.update_layout(title="Harmonic Cavity Dissipated Power", xaxis_title="Current (mA)", yaxis_title="kW", template="plotly_white")
        st.plotly_chart(fig3, width='stretch')
        
    with subtabs_op[2]: # Detuning Analysis
        st.caption("Required Detuning for Passive/Active Operation")
        col_det1, col_det2, col_det3 = st.columns(3)
        col_det1.metric("Fundamental δf (kHz)", f"{f_params['detuning_khz']:.3f}", help="Detuning at I_max for fundamental cavities")
        col_det2.metric("Harmonic δfh (kHz)", f"{h_params['detuning_khz']:.3f}", help="Detuning at I_max for harmonic cavities")
        col_det3.metric("Harmonic Phase (°)", f"{h_params['phi_hs']:.2f}", help="Phase of harmonic voltage")
        
        st.markdown("---")
        fig_det = go.Figure()
        fig_det.add_vline(x=f_params['detuning_khz'], line_dash="dash", line_color="blue", annotation_text=f"Fund.: {f_params['detuning_khz']:.2f} kHz")
        fig_det.add_vline(x=h_params['detuning_khz'], line_dash="dot", line_color="red", annotation_text=f"Harm.: {h_params['detuning_khz']:.2f} kHz")
        fig_det.update_layout(title="Operating Detuning Points", xaxis_title="Detuning (kHz)", template="plotly_white", height=300)
        st.plotly_chart(fig_det, width='stretch')
        
    with subtabs_op[3]: # Plot 5 Reflection
        st.caption("Main Cavity Coupling Match")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=i_range, y=[s['rho'] for s in scans], name="Reflection |ρ|", line=dict(color='red')))
        fig5.update_layout(title="Coupler Match (Reflection Coefficient)", xaxis_title="Current (mA)", yaxis_title="|ρ|", template="plotly_white")
        st.plotly_chart(fig5, width='stretch')

# === Tab 2: Beam Physics ===
with tabs[1]:
    st.subheader("Beam-Cavity Interaction Physics")
    st.caption("Physics Source: Standard Longitudinal Dynamics (Wiedemann, Particle Accelerator Physics), Passive Cavity Theory.")
    
    subtabs_phys = st.tabs(["Voltages", "Potential Well", "Energy Loss", "Sync Phase"])
    
    with subtabs_phys[0]: # Plot 2 Voltage
        phi_deg = np.linspace(-180, 180, 1000)
        v_m = vfcav_kv * np.sin(np.radians(phi_deg + res['phi_s']))
        v_h = res['vh_opt'] * np.sin(np.radians(nh_harm * phi_deg))
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=phi_deg, y=v_m, name=f"Main ({vfcav_kv:.0f} kV)", line=dict(dash='dash', color='blue')))
        fig2.add_trace(go.Scatter(x=phi_deg, y=v_h, name="Harmonic (Opt)", line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=phi_deg, y=v_m+v_h, name="Total Voltage", line=dict(width=3, color='black')))
        fig2.update_layout(title="RF Voltage Distribution", xaxis_title="Phase (deg)", yaxis_title="Voltage (kV)", template="plotly_white")
        st.plotly_chart(fig2, width='stretch')
        
    with subtabs_phys[1]: # Plot 4 Potential Well
        phi_pot, potential = calculate_potential_well(vfcav_kv, res['vh_opt'], res['phi_s'], nh_harm, res['ut0'])
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=phi_pot, y=potential, name="Potential", fill='tozeroy', line=dict(color='darkblue')))
        fig4.update_layout(title="Longitudinal Potential Well", xaxis_title="Phase (deg)", yaxis_title="U (arb. units)", template="plotly_white")
        st.plotly_chart(fig4, width='stretch')
        
    with subtabs_phys[2]: # Energy Loss
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("U₀ (Base Loss)", f"{u0_kev:.2f} keV")
        col_e2.metric("Uh₀ (Harmonic)", f"{res['uh0']:.2f} keV")
        col_e3.metric("UT₀ (Total)", f"{res['ut0']:.2f} keV")
        
        fig_en = go.Figure()
        fig_en.add_trace(go.Scatter(x=i_range, y=[u0_kev]*len(i_range), name="U₀ (Base)", line=dict(dash='dot')))
        fig_en.add_trace(go.Scatter(x=i_range, y=[s['uh0'] for s in scans], name="Uh₀ (Harmonic)"))
        fig_en.add_trace(go.Scatter(x=i_range, y=[s['ut0'] for s in scans], name="UT₀ (Total)", line=dict(width=3)))
        fig_en.update_layout(title="Energy Loss vs Current", xaxis_title="Current (mA)", yaxis_title="Energy (keV)", template="plotly_white")
        st.plotly_chart(fig_en, width='stretch')
        
    with subtabs_phys[3]: # Sync Phase
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Synchronous Phase Φs", f"{res['phi_s']:.2f}°")
        col_p2.metric("Optimal β", f"{res['beta_opt']:.3f}")
        
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(x=i_range, y=[s['phi_s'] for s in scans], name="Synchronous Phase", line=dict(color='orange', width=2)))
        fig_ph.update_layout(title="Synchronous Phase vs Beam Current", xaxis_title="Current (mA)", yaxis_title="Phase (°)", template="plotly_white")
        st.plotly_chart(fig_ph, width='stretch')

# === Tab 3: Stability & Control ===
with tabs[2]:
    st.subheader("Longitudinal Stability & RF Control")
    st.caption("Physics Source: CLBI (Sacherer/Laclar formula), Landau Damping (Hofmann/Zotter dispersion relation).")
    
    # --- Global Stability Calculation (Runs for all subtabs in this section) ---
    beam_clbi = {
        'I0': current_input / 1000.0,
        'E0': energy * 1e9,
        'alpha_c': float(ring_params.get("momentum_compaction", 4.4e-4)), 
        'f_rev': (f0_mhz * 1e6) / h_rf,
        'h': h_rf,
        'U0': u0_kev * 1e3,
        'sigma_z0': st.session_state.get('sigma_z0_double_rf', float(ring_params.get("sigma_z0", 15e-12)))
    }
    
    mc_clbi = {
        'V': vfcav_kv * 1000, # Total voltage seen by beam
        'Q': q0_fund / (1 + beta_fund),
        'R_sh': rshunt_mohm * 1e6 * ncav, # Total ring shunt impedance
        'psi': np.arctan( 2 * (q0_fund / (1 + beta_fund)) * (f_params['detuning_khz'] * 1000) / (f0_mhz*1e6) ),
        'f_rf': f0_mhz * 1e6
    }
    
    hc_clbi = {
        'V': res['vh_opt'] * 1000, # Total harmonic voltage
        'Q': qh0,
        'R_sh': rhshunt_mohm * 1e6 * nhcav, # Total ring HC shunt impedance
        'psi': np.radians(h_params['phi_hs']),
        'n': nh_harm,
        'theta_rel': np.radians(res['phi_h_opt']) - nh_harm * res['phi_s_rad']
    }
    
    # Perform core calculation once
    calc_success = False
    try:
        clbi_obj = DoubleRF_CLBI(beam_clbi, mc_clbi, hc_clbi)
        phi_s_clbi = clbi_obj.solve_equilibrium()
        ws_double_clbi = clbi_obj.calculate_synchrotron_frequency()
        complex_shift_clbi = clbi_obj.calculate_complex_shift(m=1)
        
        # Landau Damping
        # Reference fs0 for bunch length scaling
        phi_guess_single = np.pi - np.arcsin(beam_clbi['U0'] / mc_clbi['V'])
        ws0_single = beam_clbi['f_rev'] * np.sqrt( h_rf * beam_clbi['alpha_c'] * (mc_clbi['V'] * abs(np.cos(phi_guess_single))) / (2*np.pi*beam_clbi['E0']) )
        
        # Scaling factor: Bunch lengthens as fs decreases (sigma ~ 1/fs in linear region)
        # Cap scaling at 10x for stability
        ratio_sigma = ws0_single / (ws_double_clbi / (2*np.pi)) if ws_double_clbi > 1.0 else 10.0
        ratio_sigma = min(max(ratio_sigma, 0.1), 10.0)
        
        sigma_z_est_clbi = beam_clbi['sigma_z0'] * ratio_sigma
        delta_ws_clbi = clbi_obj.calculate_frequency_spread(sigma_z_est_clbi)
        
        # Update session state for the diagram
        st.session_state.current_complex_shift = complex_shift_clbi
        st.session_state.current_delta_ws = delta_ws_clbi
        calc_success = True
    except Exception as e:
        calc_error = str(e)

    subtabs_stab = st.tabs(["🔥 Instability Analysis (CLBI)", "📡 RF Feedback", "🌊 Landau Damping", "📊 Bosch Model", "📊 Alves Model", "🎓 J. Jacob Model", "📚 Venturini Model", "🎓 Hofmann Model"])
    
    with subtabs_stab[0]: # CLBI Analysis
        st.markdown("""
        **Longitudinal Coupled Bunch Instability (CLBI)** Analysis.
        Includes Coupled Bunch Mode growth rates and **Landau Damping** estimation from frequency spread.
        """)
        
        if calc_success:
            growth_rate = complex_shift_clbi.imag
            landau_threshold = delta_ws_clbi / 4.0
            
            # Display Results
            col_clbi1, col_clbi2, col_clbi3 = st.columns(3)
            col_clbi1.metric("Corrected Sync Freq fs", f"{ws_double_clbi/(2*np.pi):.1f} Hz")
            col_clbi2.metric("Freq Spread Δfs", f"{delta_ws_clbi/(2*np.pi):.1f} Hz", help="Incoherent spread at 1-sigma")
            
            # Stability Check
            if growth_rate >= 1.0e9:
                col_clbi3.metric("Static Unstable", "Robinson", delta="Limit", delta_color="inverse")
                st.error(f"❌ **Static Robinson Instability!** Focusing lost.")
            elif growth_rate > landau_threshold:
                col_clbi3.metric("Unstable", f"{growth_rate:.2e} /s", delta=f"Thresh: {landau_threshold:.2e}", delta_color="inverse")
                st.error(f"⚠️ **Instability Detected!** Growth > Threshold.")
            elif growth_rate > 0:
                col_clbi3.metric("Landau Damped", f"{growth_rate:.2e} /s", delta=f"Thresh: {landau_threshold:.2e}", delta_color="normal")
                st.success(f"✅ **Stable (Landau Damped)**")
            else:
                col_clbi3.metric("Stable", f"{growth_rate:.2e} /s")
                st.success("✅ **Stable**")
                
            st.info(f"Physics Model: αc={beam_clbi['alpha_c']:.1e}, E={energy} GeV, I={current_input} mA")
        else:
            st.error(f"CLBI Calculation Error: {calc_error}")
    
    with subtabs_stab[1]: # RF Feedback — Pedersen-Shen Model
        st.subheader("RF Feedback & Stability (Pedersen-Shen Model)")
        st.caption(
            "Reference: Y. B. Shen et al., *Stability analysis of double-harmonic cavity system "
            "in heavy beam loading with its feedback loops by a mathematical method based on Pedersen model*, "
            "J. Phys.: Conf. Ser. **2687**, 072026 (2024)."
        )

        fb_subtabs = st.tabs([
            "Overview & Metrics",
            "Impedance Spectrum",
            "DRFB Gain Scan",
            "Stability vs Current",
            "Nyquist & Bode",
            "Guidelines"
        ])

        # -- Common calculations --
        rf_periods = st.sidebar.number_input(
            "DRFB Delay (RF Periods)", min_value=1, max_value=2000,
            value=int(st.session_state.get('rf_periods_double_rf', 704)),
            key="rf_periods_double_rf", help="Number of RF periods in feedback loop delay"
        )
        tau_drfb_s = rf_periods / (f0_mhz * 1e6)
        tau_rad_s = st.session_state.get('tau_rad_s', 10e-3)  # radiation damping time
        f_s_hz = st.session_state.get('synch_freq_hz', 1000.0)  # synchrotron frequency

        # Pedersen DRFB analysis
        ql_mc = q0_fund / (1 + beta_fund)
        pedersen_drfb = drfb_impedance_reduction(
            rshunt_mohm * 1e6, ql_mc, f0_mhz * 1e6, rf_gain, tau_drfb_s
        )

        # Robinson limit
        rob_mc = robinson_limit(
            vfcav_kv * 1e3, res['phi_s_rad'], current_input / 1e3,
            rshunt_mohm * 1e6, ql_mc, f0_mhz * 1e6, f_s_hz, tau_rad_s
        )

        # == Sub-tab 1: Overview & Metrics ==
        with fb_subtabs[0]:
            st.markdown("### DRFB Performance Summary")

            col_fb1, col_fb2, col_fb3, col_fb4 = st.columns(4)
            col_fb1.metric("Loaded Q (QL)", f"{ql_mc:.0f}")
            col_fb2.metric(
                "BW (No FB)",
                f"{pedersen_drfb['bw_0_kHz']:.2f} kHz"
            )
            col_fb3.metric(
                "BW (With DRFB)",
                f"{pedersen_drfb['bw_eff_kHz']:.2f} kHz",
                delta=f"x{pedersen_drfb['bw_ratio']:.1f}"
            )
            col_fb4.metric(
                "Rs Reduction",
                f"{abs(pedersen_drfb['reduction_dB']):.1f} dB",
                delta=f"{pedersen_drfb['reduction_factor']*100:.0f}%"
            )

            col_gm1, col_gm2, col_gm3, col_gm4 = st.columns(4)
            col_gm1.metric("Current Gain G", f"{rf_gain:.1f}")
            col_gm2.metric(
                "Max Gain G_max",
                f"{pedersen_drfb['G_max']:.1f}"
            )
            col_gm3.metric(
                "Gain Margin",
                f"{pedersen_drfb['gain_margin_dB']:.1f} dB"
            )
            col_gm4.metric(
                "Loop Delay",
                f"{tau_drfb_s * 1e6:.3f} us"
            )

            # Status indicator
            if rf_gain > pedersen_drfb['G_max']:
                st.error(
                    f"DRFB gain ({rf_gain:.1f}) exceeds maximum ({pedersen_drfb['G_max']:.1f}). "
                    "System may be UNSTABLE. Reduce gain or feedback delay."
                )
            elif rf_gain > 0.8 * pedersen_drfb['G_max']:
                st.warning(
                    f"DRFB gain is close to maximum. "
                    f"Margin: {pedersen_drfb['gain_margin_dB']:.1f} dB."
                )
            else:
                st.success(
                    f"DRFB operating within safe margin. "
                    f"Gain headroom: {pedersen_drfb['G_max'] - rf_gain:.1f}"
                )

            # Robinson limit summary
            st.markdown("---")
            st.markdown("### Robinson Stability")
            col_rob1, col_rob2, col_rob3 = st.columns(3)
            col_rob1.metric(
                "Robinson Limit (No FB)",
                f"{rob_mc['I_max_robinson_mA']:.0f} mA"
            )
            col_rob2.metric(
                "Robinson Limit (With DRFB)",
                f"{rob_mc['I_max_robinson_mA'] * (1 + rf_gain):.0f} mA",
                delta=f"x{1 + rf_gain:.1f}"
            )
            col_rob3.metric(
                "Beam Loading Y",
                f"{rob_mc['Y']:.3f}",
                delta="Heavy" if rob_mc['Y'] > 1.0 else "Normal",
                delta_color="inverse" if rob_mc['Y'] > 1.0 else "normal"
            )

            # Pedersen model explanation
            with st.expander("Pedersen-Shen Model Theory", expanded=False):
                st.markdown(r"""
                #### The Pedersen Model Framework

                The Pedersen model (extended by Shen et al., 2024) treats the beam-cavity 
                system as a feedback control problem with multiple loops:

                **System Components:**
                - **Beam**: Acts as a current source exciting the cavity
                - **Main Cavity (MC)**: Accelerating cavity with DRFB, ALC, PLL
                - **Harmonic Cavity (HC)**: Passive cavity, beam-driven only

                **Feedback Loops:**
                | Loop | Function | Timescale |
                |------|----------|----------|
                | **DRFB** | Reduces cavity impedance, expands bandwidth | Fast (us) |
                | **ALC** | Maintains voltage amplitude | Medium (ms) |
                | **PLL** | Tracks beam phase | Medium (ms) |

                **Key Formulas:**
                """)
                st.latex(r"G_{\max} = \frac{\pi Q_L}{2\omega_0 \tau_{\mathrm{delay}}} - 1")
                st.latex(r"\mathrm{BW}_{\mathrm{eff}} = \mathrm{BW}_0 \cdot (1 + G)")
                st.latex(r"R_{s,\mathrm{eff}} = \frac{R_s}{1 + G}")
                st.markdown(r"""
                **Robinson Instability Limit:**
                """)
                st.latex(r"I_{\max} = \frac{2V_c Q_L}{\omega_0 R_s} \cdot \alpha_{\mathrm{rad}} \cdot (1+G)")
                st.markdown(
                    "The DRFB improves the Robinson limit by a factor of $(1+G)$, "
                    "making it essential for heavy beam loading operation."
                )

        # == Sub-tab 2: Impedance Spectrum ==
        with fb_subtabs[1]:
            st.markdown("### Cavity Impedance Spectrum")
            st.markdown(
                "Shows how DRFB reduces the effective cavity impedance seen by the beam, "
                "broadening the bandwidth and reducing peak impedance."
            )

            imp_span = st.slider("Frequency Span (kHz)", 50, 2000, 500, key="imp_span_fb")
            imp_data = impedance_spectrum(
                f0_mhz * 1e6, rshunt_mohm * 1e6, ql_mc,
                G=rf_gain, f_span_kHz=imp_span
            )

            col_imp1, col_imp2 = st.columns(2)
            with col_imp1:
                fig_imp_abs = go.Figure()
                fig_imp_abs.add_trace(go.Scatter(
                    x=imp_data['f_kHz'], y=imp_data['Z_nat_abs'] / 1e6,
                    name='No Feedback', line=dict(color='#ff4b4b', width=2, dash='dash')
                ))
                fig_imp_abs.add_trace(go.Scatter(
                    x=imp_data['f_kHz'], y=imp_data['Z_fb_abs'] / 1e6,
                    name=f'With DRFB (G={rf_gain:.1f})',
                    line=dict(color='#4facfe', width=2)
                ))
                fig_imp_abs.update_layout(
                    title="|Z(f)| - Impedance Magnitude",
                    xaxis_title="Frequency offset (kHz)",
                    yaxis_title="|Z| (MOhm)",
                    template="plotly_white", height=400
                )
                st.plotly_chart(fig_imp_abs, width='stretch')

            with col_imp2:
                fig_imp_re = go.Figure()
                fig_imp_re.add_trace(go.Scatter(
                    x=imp_data['f_kHz'], y=imp_data['Z_nat_real'] / 1e6,
                    name='Re[Z] No FB', line=dict(color='#ff4b4b', width=2, dash='dash')
                ))
                fig_imp_re.add_trace(go.Scatter(
                    x=imp_data['f_kHz'], y=imp_data['Z_fb_real'] / 1e6,
                    name=f'Re[Z] With DRFB',
                    line=dict(color='#4facfe', width=2)
                ))
                fig_imp_re.add_trace(go.Scatter(
                    x=imp_data['f_kHz'], y=imp_data['Z_nat_imag'] / 1e6,
                    name='Im[Z] No FB', line=dict(color='#ffa07a', width=1, dash='dot')
                ))
                fig_imp_re.add_trace(go.Scatter(
                    x=imp_data['f_kHz'], y=imp_data['Z_fb_imag'] / 1e6,
                    name='Im[Z] With DRFB',
                    line=dict(color='#87ceeb', width=1, dash='dot')
                ))
                fig_imp_re.update_layout(
                    title="Re[Z] and Im[Z] Components",
                    xaxis_title="Frequency offset (kHz)",
                    yaxis_title="Z (MOhm)",
                    template="plotly_white", height=400
                )
                st.plotly_chart(fig_imp_re, width='stretch')

            st.info(
                f"**Peak impedance reduction:** {rshunt_mohm:.1f} MOhm -> "
                f"{pedersen_drfb['Rs_eff_MOhm']:.1f} MOhm "
                f"({abs(pedersen_drfb['reduction_dB']):.1f} dB). "
                f"**Bandwidth expansion:** {pedersen_drfb['bw_0_kHz']:.1f} kHz -> "
                f"{pedersen_drfb['bw_eff_kHz']:.1f} kHz."
            )

        # == Sub-tab 3: DRFB Gain Scan ==
        with fb_subtabs[2]:
            st.markdown("### DRFB Gain Optimization")
            st.markdown(
                "Scan over DRFB gain to find the optimal operating point. "
                "Higher gain reduces impedance and improves Robinson limit, "
                "but is limited by loop delay."
            )

            gain_scan = scan_drfb_gain(
                V_mc=vfcav_kv * 1e3, Rs_mc=rshunt_mohm * 1e6,
                Q0_mc=q0_fund, beta_mc=beta_fund,
                f_rf=f0_mhz * 1e6, I_b=current_input / 1e3,
                phi_s=res['phi_s_rad'], f_s=f_s_hz,
                tau_rad=tau_rad_s, tau_drfb=tau_drfb_s
            )

            col_gs1, col_gs2 = st.columns(2)
            with col_gs1:
                fig_gs_bw = go.Figure()
                fig_gs_bw.add_trace(go.Scatter(
                    x=gain_scan['gains'], y=gain_scan['bw_kHz'],
                    name='Effective BW', line=dict(color='blue', width=2)
                ))
                fig_gs_bw.add_vline(
                    x=rf_gain, line_dash="dash", line_color="red",
                    annotation_text=f"Current: G={rf_gain:.1f}"
                )
                fig_gs_bw.add_vline(
                    x=gain_scan['G_max'], line_dash="dot", line_color="green",
                    annotation_text=f"G_max={gain_scan['G_max']:.1f}"
                )
                fig_gs_bw.update_layout(
                    title="Bandwidth vs DRFB Gain",
                    xaxis_title="DRFB Gain", yaxis_title="Bandwidth (kHz)",
                    template="plotly_white", height=380
                )
                st.plotly_chart(fig_gs_bw, width='stretch')

            with col_gs2:
                fig_gs_rs = go.Figure()
                fig_gs_rs.add_trace(go.Scatter(
                    x=gain_scan['gains'], y=gain_scan['Rs_eff_MOhm'],
                    name='Effective Rs', line=dict(color='orange', width=2)
                ))
                fig_gs_rs.add_vline(
                    x=rf_gain, line_dash="dash", line_color="red",
                    annotation_text=f"Current"
                )
                fig_gs_rs.add_vline(
                    x=gain_scan['G_max'], line_dash="dot", line_color="green",
                    annotation_text=f"G_max"
                )
                fig_gs_rs.update_layout(
                    title="Effective Impedance vs DRFB Gain",
                    xaxis_title="DRFB Gain", yaxis_title="Rs_eff (MOhm)",
                    template="plotly_white", height=380
                )
                st.plotly_chart(fig_gs_rs, width='stretch')

            # Robinson limit vs gain
            fig_gs_rob = go.Figure()
            fig_gs_rob.add_trace(go.Scatter(
                x=gain_scan['gains'], y=gain_scan['I_max_mA'],
                name='Robinson Limit', line=dict(color='purple', width=2),
                fill='tozeroy', fillcolor='rgba(147,112,219,0.1)'
            ))
            fig_gs_rob.add_hline(
                y=current_input, line_dash="dash", line_color="red",
                annotation_text=f"Operating Current: {current_input:.0f} mA"
            )
            fig_gs_rob.add_vline(
                x=rf_gain, line_dash="dash", line_color="red",
                annotation_text=f"G={rf_gain:.1f}"
            )
            fig_gs_rob.add_vline(
                x=gain_scan['G_max'], line_dash="dot", line_color="green",
                annotation_text=f"G_max={gain_scan['G_max']:.1f}"
            )
            fig_gs_rob.update_layout(
                title="Robinson Current Limit vs DRFB Gain",
                xaxis_title="DRFB Gain",
                yaxis_title="Max Stable Current (mA)",
                template="plotly_white", height=400
            )
            st.plotly_chart(fig_gs_rob, width='stretch')

            st.info(
                f"**Without DRFB:** Robinson limit = {gain_scan['robinson_base_mA']:.0f} mA. "
                f"**With G={rf_gain:.1f}:** limit = {gain_scan['robinson_base_mA'] * (1 + rf_gain):.0f} mA. "
                f"**At G_max={gain_scan['G_max']:.1f}:** limit = {gain_scan['robinson_base_mA'] * (1 + gain_scan['G_max']):.0f} mA."
            )

        # == Sub-tab 4: Stability vs Current ==
        with fb_subtabs[3]:
            st.markdown("### Stability vs Beam Current")
            st.markdown(
                "Shows how the growth rate of Robinson-type instabilities evolves "
                "with increasing beam current, and when it exceeds radiation damping."
            )

            i_max_scan = float(st.session_state.get('imax_double_rf', 500))
            current_scan = scan_current_stability(
                V_mc=vfcav_kv * 1e3, Rs_mc=rshunt_mohm * 1e6,
                Q0_mc=q0_fund, beta_mc=beta_fund,
                f_rf=f0_mhz * 1e6, f_s=f_s_hz,
                tau_rad=tau_rad_s, G_drfb=rf_gain, tau_drfb=tau_drfb_s,
                V_hc=res['vh_opt'] * 1e3, Rs_hc=rhshunt_mohm * 1e6,
                Q0_hc=qh0, m=nh_harm,
                delta_f_hc=h_params['detuning_khz'] * 1e3,
                phi_s=res['phi_s_rad'],
                i_max_mA=i_max_scan
            )

            fig_cs = go.Figure()
            fig_cs.add_trace(go.Scatter(
                x=current_scan['I_mA'], y=current_scan['mc_growth'],
                name='MC Growth Rate', line=dict(color='blue', width=2)
            ))
            fig_cs.add_trace(go.Scatter(
                x=current_scan['I_mA'], y=current_scan['hc_growth'],
                name='HC Growth Rate', line=dict(color='orange', width=2)
            ))
            fig_cs.add_trace(go.Scatter(
                x=current_scan['I_mA'], y=current_scan['total_growth'],
                name='Total Growth', line=dict(color='red', width=2, dash='dash')
            ))
            fig_cs.add_trace(go.Scatter(
                x=current_scan['I_mA'], y=current_scan['damping_rate'],
                name='Radiation Damping', line=dict(color='green', width=2, dash='dot')
            ))
            fig_cs.add_vline(
                x=current_input, line_dash="dash", line_color="purple",
                annotation_text=f"I = {current_input:.0f} mA"
            )
            fig_cs.update_layout(
                title="Growth Rate vs Beam Current (Pedersen-Shen Model)",
                xaxis_title="Beam Current (mA)",
                yaxis_title="Growth/Damping Rate (1/s)",
                template="plotly_white", height=450,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_cs, width='stretch')
            st.caption(
                "Blue: MC contribution. Orange: HC contribution. "
                "Red dashed: total. Green dotted: radiation damping threshold."
            )

            # Stability margin plot
            fig_margin = go.Figure()
            fig_margin.add_trace(go.Scatter(
                x=current_scan['I_mA'], y=current_scan['stability_margin'],
                name='Stability Margin',
                line=dict(color='teal', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,128,128,0.1)'
            ))
            fig_margin.add_hline(y=0, line_color='red', line_width=1)
            fig_margin.add_vline(
                x=current_input, line_dash="dash", line_color="purple",
                annotation_text=f"I = {current_input:.0f} mA"
            )
            fig_margin.update_layout(
                title="Stability Margin vs Current",
                xaxis_title="Beam Current (mA)",
                yaxis_title="Margin (Damping - Growth) [1/s]",
                template="plotly_white", height=350
            )
            st.plotly_chart(fig_margin, width='stretch')
            st.caption("Positive margin = stable. Crossing zero = Robinson instability threshold.")


        # == Sub-tab 5: Nyquist & Bode Plots ==
        with fb_subtabs[4]:
            st.markdown("### Nyquist & Bode Stability Analysis")
            st.markdown(
                r"Nyquist diagram of the open-loop transfer function $T_{OL}(j\omega)$ "
                r"following the Pedersen-Shen model. The system is **stable** if the contour "
                r"does **not** encircle the critical point $(-1, 0)$."
            )

            nyq_col_ctrl1, nyq_col_ctrl2 = st.columns(2)
            with nyq_col_ctrl1:
                nyq_phi_fb = st.slider(
                    "DRFB Phase Offset (°)", -180, 180, 0,
                    step=5, key="nyq_phi_fb",
                    help="Phase offset in the feedback path"
                )
            with nyq_col_ctrl2:
                nyq_span = st.slider(
                    "Frequency Span (x BW)", 3, 30, 10,
                    key="nyq_span",
                    help="Frequency range as multiple of cavity half-bandwidth"
                )

            phi_fb_rad = np.radians(nyq_phi_fb)

            # --- Main Cavity Nyquist ---
            st.markdown("#### Main Cavity (with DRFB)")

            nq_tabs = st.tabs(["Nyquist Diagram", "Multi-Current Nyquist", "HC Nyquist", "Bode Plot"])

            # -- Tab: Single Nyquist --
            with nq_tabs[0]:
                # Compute contours: with and without beam
                nc_no_beam = nyquist_contour(
                    f0_mhz * 1e6, rshunt_mohm * 1e6, ql_mc,
                    rf_gain, tau_drfb_s, phi_fb_rad,
                    I_b=0, V_c=vfcav_kv * 1e3, phi_s=res['phi_s_rad'],
                    include_beam=False, f_span_factor=nyq_span
                )
                nc_with_beam = nyquist_contour(
                    f0_mhz * 1e6, rshunt_mohm * 1e6, ql_mc,
                    rf_gain, tau_drfb_s, phi_fb_rad,
                    I_b=current_input / 1e3, V_c=vfcav_kv * 1e3,
                    phi_s=res['phi_s_rad'],
                    include_beam=True, f_span_factor=nyq_span
                )

                fig_nyq = go.Figure()

                # DRFB only (no beam)
                fig_nyq.add_trace(go.Scatter(
                    x=nc_no_beam['real'], y=nc_no_beam['imag'],
                    name=f'DRFB only (G={rf_gain:.1f})',
                    line=dict(color='#4facfe', width=1.5, dash='dash'),
                    mode='lines'
                ))

                # With beam loading
                fig_nyq.add_trace(go.Scatter(
                    x=nc_with_beam['real'], y=nc_with_beam['imag'],
                    name=f'DRFB + Beam ({current_input:.0f} mA)',
                    line=dict(color='#ff6b6b', width=2),
                    mode='lines'
                ))

                # Critical point (-1, 0)
                fig_nyq.add_trace(go.Scatter(
                    x=[-1], y=[0],
                    name='Critical Point (-1, 0)',
                    mode='markers',
                    marker=dict(
                        color='red', size=14, symbol='x-thin-open',
                        line=dict(width=3)
                    )
                ))

                # Unit circle
                theta_uc = np.linspace(0, 2*np.pi, 100)
                fig_nyq.add_trace(go.Scatter(
                    x=np.cos(theta_uc), y=np.sin(theta_uc),
                    name='Unit Circle',
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', showlegend=False
                ))

                # Closest approach line
                fig_nyq.add_trace(go.Scatter(
                    x=[-1, nc_with_beam['closest_point_real']],
                    y=[0, nc_with_beam['closest_point_imag']],
                    name=f"Min. dist: {nc_with_beam['min_dist_to_critical']:.3f}",
                    line=dict(color='orange', width=1, dash='dot'),
                    mode='lines+markers',
                    marker=dict(size=6, color='orange')
                ))

                # Determine plot range from data
                all_re = np.concatenate([nc_no_beam['real'], nc_with_beam['real']])
                all_im = np.concatenate([nc_no_beam['imag'], nc_with_beam['imag']])
                pad = 0.3
                x_lim = max(abs(all_re.min()), abs(all_re.max()), 1.5) + pad
                y_lim = max(abs(all_im.min()), abs(all_im.max()), 1.5) + pad
                lim = max(x_lim, y_lim)

                fig_nyq.update_layout(
                    title="Nyquist Diagram — MC Open-Loop T_OL(jω)",
                    xaxis_title="Re[T_OL]",
                    yaxis_title="Im[T_OL]",
                    template="plotly_white",
                    height=550,
                    xaxis=dict(
                        range=[-lim, lim],
                        zeroline=True, zerolinewidth=1, zerolinecolor='black',
                        scaleanchor='y', scaleratio=1
                    ),
                    yaxis=dict(
                        range=[-lim, lim],
                        zeroline=True, zerolinewidth=1, zerolinecolor='black'
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_nyq, width='stretch')

                # Stability metrics
                col_nyq1, col_nyq2, col_nyq3, col_nyq4 = st.columns(4)
                col_nyq1.metric(
                    "Winding Number",
                    f"{nc_with_beam['winding_number']}",
                    delta="Stable" if nc_with_beam['is_stable'] else "UNSTABLE",
                    delta_color="normal" if nc_with_beam['is_stable'] else "inverse"
                )
                gm_val = nc_with_beam['gain_margin_dB']
                gm_str = f"{gm_val:.1f} dB" if gm_val < 1e6 else "∞"
                col_nyq2.metric("Gain Margin", gm_str)
                pm_val = nc_with_beam['phase_margin_deg']
                pm_str = f"{pm_val:.1f}°" if pm_val < 1e6 else "∞"
                col_nyq3.metric("Phase Margin", pm_str)
                col_nyq4.metric(
                    "Min. Distance",
                    f"{nc_with_beam['min_dist_to_critical']:.3f}"
                )

                if nc_with_beam['is_stable']:
                    st.success(
                        f"**Stable**: The Nyquist contour does not encircle (-1, 0). "
                        f"Winding number = {nc_with_beam['winding_number']}."
                    )
                else:
                    st.error(
                        f"**UNSTABLE**: The Nyquist contour encircles (-1, 0). "
                        f"Winding number = {nc_with_beam['winding_number']}. "
                        "Reduce DRFB gain or beam current."
                    )

            # -- Tab: Multi-Current Nyquist --
            with nq_tabs[1]:
                st.markdown(
                    "Nyquist contours for different beam currents, showing how "
                    "beam loading progressively shifts the contour toward the "
                    "critical point."
                )

                i_max_nyq = float(st.session_state.get('imax_double_rf', 500))
                n_currents = st.slider("Number of Current Steps", 3, 8, 5, key="n_curr_nyq")
                i_list = np.linspace(0, i_max_nyq, n_currents + 1)[1:].tolist()  # exclude 0

                mc_nyq = nyquist_multi_current(
                    f0_mhz * 1e6, rshunt_mohm * 1e6, ql_mc,
                    rf_gain, tau_drfb_s, phi_fb_rad,
                    vfcav_kv * 1e3, res['phi_s_rad'],
                    i_list, f_span_factor=nyq_span, n_points=1000
                )

                # Color scale from blue (low current) to red (high current)
                colors = [
                    f'hsl({240 - int(240 * i / (len(mc_nyq["contours"]) - 1))}, 80%, 55%)'
                    if len(mc_nyq['contours']) > 1 else 'hsl(240, 80%, 55%)'
                    for i in range(len(mc_nyq['contours']))
                ]

                fig_mc_nyq = go.Figure()
                for idx, c in enumerate(mc_nyq['contours']):
                    stability_marker = "✓" if c['is_stable'] else "✗"
                    fig_mc_nyq.add_trace(go.Scatter(
                        x=c['real'], y=c['imag'],
                        name=f"{c['I_mA']:.0f} mA {stability_marker}",
                        line=dict(color=colors[idx], width=1.5),
                        mode='lines'
                    ))

                # Critical point
                fig_mc_nyq.add_trace(go.Scatter(
                    x=[-1], y=[0],
                    name='(-1, 0)',
                    mode='markers',
                    marker=dict(
                        color='red', size=14, symbol='x-thin-open',
                        line=dict(width=3)
                    )
                ))

                # Unit circle
                fig_mc_nyq.add_trace(go.Scatter(
                    x=np.cos(theta_uc), y=np.sin(theta_uc),
                    name='Unit Circle',
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', showlegend=False
                ))

                # Auto-scale
                all_re_mc = np.concatenate([c['real'] for c in mc_nyq['contours']])
                all_im_mc = np.concatenate([c['imag'] for c in mc_nyq['contours']])
                lim_mc = max(
                    abs(all_re_mc.min()), abs(all_re_mc.max()),
                    abs(all_im_mc.min()), abs(all_im_mc.max()),
                    1.5
                ) + pad

                fig_mc_nyq.update_layout(
                    title="Nyquist Diagram — Beam Current Dependence",
                    xaxis_title="Re[T_OL]",
                    yaxis_title="Im[T_OL]",
                    template="plotly_white",
                    height=550,
                    xaxis=dict(
                        range=[-lim_mc, lim_mc],
                        zeroline=True, zerolinewidth=1, zerolinecolor='black',
                        scaleanchor='y', scaleratio=1
                    ),
                    yaxis=dict(
                        range=[-lim_mc, lim_mc],
                        zeroline=True, zerolinewidth=1, zerolinecolor='black'
                    ),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_mc_nyq, width='stretch')

                # Summary table
                st.markdown("**Stability Summary:**")
                summary_data = []
                for c in mc_nyq['contours']:
                    gm = c['gain_margin_dB']
                    pm = c['phase_margin_deg']
                    summary_data.append({
                        'Current (mA)': f"{c['I_mA']:.0f}",
                        'Stable': '✅' if c['is_stable'] else '❌',
                        'Winding #': c['winding_number'],
                        'Gain Margin': f"{gm:.1f} dB" if gm < 1e6 else '∞',
                        'Phase Margin': f"{pm:.1f}°" if pm < 1e6 else '∞',
                        'Min. Dist.': f"{c['min_dist_to_critical']:.3f}"
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)

            # -- Tab: HC Nyquist --
            with nq_tabs[2]:
                st.markdown("#### Passive Harmonic Cavity Nyquist")
                st.markdown(
                    r"The HC has no DRFB. Its stability depends on beam loading "
                    r"and detuning. The contour shows $T_{HC}(j\omega) = Y_{HC} \sin\phi_s \cdot H_{HC}(j\omega)$."
                )

                nc_hc = nyquist_hc_passive(
                    f0_mhz * 1e6, rhshunt_mohm * 1e6, qh0, nh_harm,
                    current_input / 1e3, res['vh_opt'] * 1e3,
                    h_params['detuning_khz'] * 1e3,
                    res['phi_s_rad'],
                    f_span_factor=nyq_span
                )

                fig_hc_nyq = go.Figure()
                fig_hc_nyq.add_trace(go.Scatter(
                    x=nc_hc['real'], y=nc_hc['imag'],
                    name=f'HC (I={current_input:.0f} mA)',
                    line=dict(color='#ffa726', width=2),
                    mode='lines'
                ))
                fig_hc_nyq.add_trace(go.Scatter(
                    x=[-1], y=[0],
                    name='(-1, 0)',
                    mode='markers',
                    marker=dict(
                        color='red', size=14, symbol='x-thin-open',
                        line=dict(width=3)
                    )
                ))
                fig_hc_nyq.add_trace(go.Scatter(
                    x=np.cos(theta_uc), y=np.sin(theta_uc),
                    line=dict(color='gray', width=1, dash='dot'),
                    mode='lines', showlegend=False
                ))

                all_re_hc = nc_hc['real']
                all_im_hc = nc_hc['imag']
                lim_hc = max(
                    abs(all_re_hc.min()), abs(all_re_hc.max()),
                    abs(all_im_hc.min()), abs(all_im_hc.max()),
                    1.5
                ) + pad

                fig_hc_nyq.update_layout(
                    title="Nyquist Diagram — Passive HC",
                    xaxis_title="Re[T_HC]",
                    yaxis_title="Im[T_HC]",
                    template="plotly_white",
                    height=500,
                    xaxis=dict(
                        range=[-lim_hc, lim_hc],
                        zeroline=True, zerolinewidth=1, zerolinecolor='black',
                        scaleanchor='y', scaleratio=1
                    ),
                    yaxis=dict(
                        range=[-lim_hc, lim_hc],
                        zeroline=True, zerolinewidth=1, zerolinecolor='black'
                    )
                )
                st.plotly_chart(fig_hc_nyq, width='stretch')

                col_hc1, col_hc2, col_hc3 = st.columns(3)
                col_hc1.metric(
                    "HC Stable",
                    "Yes" if nc_hc['is_stable'] else "No",
                    delta="Stable" if nc_hc['is_stable'] else "UNSTABLE",
                    delta_color="normal" if nc_hc['is_stable'] else "inverse"
                )
                col_hc2.metric("Beam Loading Y_hc", f"{nc_hc['Y_hc']:.3f}")
                col_hc3.metric(
                    "Min. Distance",
                    f"{nc_hc['min_dist_to_critical']:.3f}"
                )

            # -- Tab: Bode Plot --
            with nq_tabs[3]:
                st.markdown("#### Bode Plot — Open-Loop Transfer Function")
                st.markdown(
                    "Magnitude and phase of the open-loop transfer function vs frequency. "
                    "**Gain margin** is read where phase crosses -180°. "
                    "**Phase margin** is read where magnitude crosses 0 dB."
                )

                bode = bode_plot_data(
                    f0_mhz * 1e6, rshunt_mohm * 1e6, ql_mc,
                    rf_gain, tau_drfb_s, phi_fb_rad,
                    I_b=current_input / 1e3, V_c=vfcav_kv * 1e3,
                    phi_s=res['phi_s_rad'],
                    include_beam=True, f_span_factor=nyq_span
                )

                from plotly.subplots import make_subplots
                fig_bode = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=('Magnitude (dB)', 'Phase (°)'),
                    vertical_spacing=0.08
                )

                # Magnitude
                fig_bode.add_trace(go.Scatter(
                    x=bode['f_kHz'], y=bode['mag_dB'],
                    name='|T_OL|',
                    line=dict(color='#4facfe', width=2)
                ), row=1, col=1)
                fig_bode.add_hline(
                    y=0, line_dash='dash', line_color='red',
                    annotation_text='0 dB', row=1, col=1
                )

                # Phase
                fig_bode.add_trace(go.Scatter(
                    x=bode['f_kHz'], y=bode['phase_deg'],
                    name='Phase',
                    line=dict(color='#ff6b6b', width=2)
                ), row=2, col=1)
                fig_bode.add_hline(
                    y=-180, line_dash='dash', line_color='red',
                    annotation_text='-180°', row=2, col=1
                )

                fig_bode.update_layout(
                    height=600,
                    template='plotly_white',
                    title_text='Bode Plot — DRFB Open-Loop T_OL(jω)',
                    showlegend=False
                )
                fig_bode.update_xaxes(title_text='Frequency (kHz)', row=2, col=1)
                fig_bode.update_yaxes(title_text='|T_OL| (dB)', row=1, col=1)
                fig_bode.update_yaxes(title_text='Phase (°)', row=2, col=1)
                st.plotly_chart(fig_bode, width='stretch')

                col_bode1, col_bode2, col_bode3 = st.columns(3)
                col_bode1.metric(
                    "Peak Gain",
                    f"{bode['mag_dB'].max():.1f} dB"
                )
                col_bode2.metric(
                    "BW (0 dB)",
                    f"{pedersen_drfb['bw_eff_kHz']:.1f} kHz"
                )
                gm_val_b = nc_with_beam['gain_margin_dB']
                col_bode3.metric(
                    "Gain Margin",
                    f"{gm_val_b:.1f} dB" if gm_val_b < 1e6 else "∞"
                )

        # == Sub-tab 6: Operational Guidelines ==
        with fb_subtabs[5]:
            st.markdown("### Operational Guidelines (Shen 2024)")

            guidelines = operational_guidelines(
                I_b_mA=current_input, V_mc_kV=vfcav_kv,
                Rs_mc_MOhm=rshunt_mohm, Q0_mc=q0_fund, beta_mc=beta_fund,
                G_drfb=rf_gain, tau_drfb_us=tau_drfb_s * 1e6,
                f_rf_MHz=f0_mhz,
                V_hc_kV=res['vh_opt'], Rs_hc_MOhm=rhshunt_mohm,
                Q0_hc=qh0, delta_f_hc_kHz=h_params['detuning_khz'], m=nh_harm
            )

            # Status
            if guidelines['status'] == 'CRITICAL':
                st.error("### Status: CRITICAL")
            elif guidelines['status'] == 'WARNING':
                st.warning("### Status: WARNING")
            else:
                st.success("### Status: OK")

            # Warnings
            if guidelines['warnings']:
                st.markdown("#### Warnings")
                for w in guidelines['warnings']:
                    st.warning(w)

            # Recommendations
            st.markdown("#### Recommendations")
            for r_text in guidelines['recommendations']:
                st.info(r_text)

            # Operating procedure
            st.markdown("---")
            st.markdown("#### Injection / Current Ramp Procedure (Shen 2024)")
            st.markdown("""
            1. **Before injection**: Set MC pre-detuning slightly above resonance.
               DRFB will regulate voltage. HC should be far-detuned.
            
            2. **During fill**: Monitor Robinson margin. As current increases:
               - MC beam loading increases -> DRFB compensates
               - Reduce HC detuning progressively toward flat-potential target
            
            3. **At design current**: Verify HC is near flat-potential detuning.
               Check that DRFB gain margin > 6 dB.
            
            4. **Stability check**: Total growth rate must stay below radiation damping.
               Use the *Stability vs Current* tab to verify.
            """)

            st.markdown("---")
            st.markdown("#### Reference")
            st.markdown(
                "Y. B. Shen, Q. Gu, Z. G. Jiang, D. Gu, Z. H. Zhu, "
                "*Stability analysis of double-harmonic cavity system in heavy beam loading "
                "with its feedback loops by a mathematical method based on Pedersen model*, "
                "J. Phys.: Conf. Ser. **2687**, 072026 (2024). "
                "[DOI: 10.1088/1742-6596/2687/7/072026](https://doi.org/10.1088/1742-6596/2687/7/072026)"
            )

    with subtabs_stab[2]: # Landau Damping
        st.subheader("🌊 Landau Damping Mechanism")
        st.markdown("""
        Landau damping is a critical stability mechanism in double RF systems. By adding a harmonic cavity, 
        the longitudinal potential well is flattened, which introduces a significant **synchrotron frequency spread**. 
        This spread leads to the decoherence of collective motions, thereby suppressing instabilities.
        """)
        
        # Define qualitative data for comparison (as requested)
        phi_ld = np.linspace(-1.5, 1.5, 100)
        # Potential Wells: Standard (x^2) vs Harmonic (x^4)
        standard_u = phi_ld**2
        harmonic_u = phi_ld**4
        # Synchrotron Frequencies: Constant vs Amplitude-dependent
        standard_fs = np.ones_like(phi_ld) * 0.8
        harmonic_fs = np.abs(phi_ld) * 0.6 

        col_ld1, col_ld2 = st.columns(2)
        
        with col_ld1:
            fig_pot = go.Figure()
            # Standard RF
            fig_pot.add_trace(go.Scatter(
                x=phi_ld, y=standard_u, 
                name='Standard RF ($x^2$)', 
                line=dict(color='#4facfe', width=2)
            ))
            # Harmonic RF
            fig_pot.add_trace(go.Scatter(
                x=phi_ld, y=harmonic_u, 
                name='Harmonic RF ($x^4$)', 
                line=dict(color='#ff4b4b', width=3)
            ))
            fig_pot.update_layout(
                title="Potential Well Comparison [cite: 127]", 
                xaxis_title="Phase (φ)", 
                yaxis_title="Potential (U)", 
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=400,
                margin=dict(t=50, b=50, l=50, r=20)
            )
            st.plotly_chart(fig_pot, width='stretch')

        with col_ld2:
            fig_fs = go.Figure()
            # Constant fs
            fig_fs.add_trace(go.Scatter(
                x=phi_ld, y=standard_fs, 
                name='Constant f_s (No Damping)', 
                line=dict(color='#4facfe', dash='dash', width=2)
            ))
            # Variable fs
            fig_fs.add_trace(go.Scatter(
                x=phi_ld, y=harmonic_fs, 
                name='Variable f_s (Landau Spread)', 
                line=dict(color='#ffa500', width=3)
            ))
            fig_fs.update_layout(
                title="Synchrotron Frequency Spread [cite: 134, 138]", 
                xaxis_title="Amplitude (Phase)", 
                yaxis_title="Frequency (f_s)", 
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=400,
                margin=dict(t=50, b=50, l=50, r=20)
            )
            st.plotly_chart(fig_fs, width='stretch')
        
        st.info("**Mechanism:** The harmonic RF system creates a 'flat-bottom' potential ($x^4$ at the center), which results in a synchrotron frequency that increases with the oscillation amplitude. This decoherence suppresses the growth of coupled-bunch modes.")

        st.markdown("---")
        st.subheader("📊 Quantifying Stability: The Dispersion Relation")
        st.latex(r"1 = -i \Delta \Omega \int \frac{\psi'(J)}{\Omega - \omega(J)} dJ")
        
        st.markdown(r"""
        **The Stability Criterion:** If the impedance-driven frequency shift $\Delta \Omega$ falls inside the boundary 
        defined by the inverse dispersion integral, the beam remains stable despite the presence of wakefields.
        """)

        # Get results from session with safety defaults
        c_shift = st.session_state.get('current_complex_shift', complex(0.0, 0.0))
        d_ws_raw = st.session_state.get('current_delta_ws', 0.0)
        d_ws = d_ws_raw / (2*np.pi) if d_ws_raw > 0 else 50.0 # Default 50Hz for scale if calc failed
        
        # Define stability boundary based on frequency spread (Inverse Dispersion Relation for Gaussian)
        v_boundary = np.linspace(-1.5, 1.5, 100)
        # Higher spread = Larger stable region
        bound_x = d_ws * v_boundary
        bound_y = d_ws * (0.8 + 0.5 * v_boundary**2) # V/U shape scaling with spread
        
        fig_sd = go.Figure()
        
        # Stable Region (Filled Area)
        # Close the polygon by going back from right to left at the top
        fig_sd.add_trace(go.Scatter(
            x=np.concatenate([bound_x, bound_x[::-1]]), 
            y=np.concatenate([bound_y, [10 * d_ws] * len(bound_x)]), 
            fill='toself',
            fillcolor='rgba(135, 206, 235, 0.3)',
            line=dict(color='blue', width=1),
            name='Stable Region'
        ))
        
        # Coherent Frequency Shift Point (Scale from clbi rad/s to Hz)
        px = c_shift.real / (2*np.pi)
        py = c_shift.imag / (2*np.pi)
        
        # Determine stability state for visualization
        limit_y = d_ws * (0.8 + 0.5 * (px/d_ws)**2) if d_ws > 0 else 0
        is_stable = (py < limit_y) and (getattr(clbi_obj, 'is_statically_unstable', False) == False)
        
        fig_sd.add_trace(go.Scatter(
            x=[px], y=[py],
            mode='markers+text',
            marker=dict(
                color='green' if is_stable else 'red', 
                size=18, 
                symbol='diamond',
                line=dict(color='black', width=2)
            ),
            text=["STABLE" if is_stable else "UNSTABLE"],
            textposition="top center",
            name='Current Operating Point'
        ))
        
        # Dynamic axis range with minimums to avoid empty plots
        x_limit = max(2 * d_ws, abs(px) * 1.5, 100)
        y_limit_max = max(3 * d_ws, py * 1.5, 200)
        y_limit_min = -0.5 * d_ws
        
        # Axes and Layout
        fig_sd.update_layout(
            title="Landau Stability Diagram (Dispersion Relation)",
            xaxis_title="Real(ΔΩ) - Coherent Frequency Shift (Hz)",
            yaxis_title="Imag(ΔΩ) - Growth Rate (Hz)",
            template="plotly_white",
            height=500,
            xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black', range=[-x_limit, x_limit]),
            yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black', range=[y_limit_min, y_limit_max]),
            showlegend=True
        )
        
        st.plotly_chart(fig_sd, width='stretch')
        st.caption(f"Calculated Frequency Spread (Δfs): {d_ws_raw/(2*np.pi):.1f} Hz. Coherent Frequency Shift: {px:.1f} Hz. Stability boundary defines the suppression limit.")

    # ── Sub-tab 4: Bosch Model ──
    with subtabs_stab[3]:
        st.subheader("📊 Bosch Instability Model")
        st.caption(
            "Reference: R. A. Bosch & C. S. Hsue, *Suppression of longitudinal coupled-bunch "
            "instabilities by a passive higher harmonic cavity*, Part. Accel. **42**, 81 (1993); "
            "R. A. Bosch, *Instability analysis of an active higher-harmonic cavity*, PAC97 (1997)."
        )

        bosch_mode_tabs = st.tabs(["Overview", "Current Scan", "Theory"])

        # Get parameters from session state
        bosch_E_MeV = st.session_state.get('energy_gev_double_rf', 2.75) * 1e3  # GeV -> MeV
        bosch_alpha = st.session_state.get('alpha_c_double_rf', 3.46e-4)
        bosch_C = st.session_state.get('circumference_m_double_rf', 353.97)
        bosch_T0 = bosch_C / 3e8  # revolution period
        bosch_sigma_E = st.session_state.get('energy_spread_double_rf', 1.017e-3)

        with bosch_mode_tabs[0]:  # Overview
            st.markdown("### Bosch Analysis — Current Operating Point")

            bosch_col1, bosch_col2 = st.columns(2)
            with bosch_col1:
                bosch_cavity_mode = st.radio(
                    "Harmonic Cavity Mode",
                    ["Passive", "Active"],
                    index=0,
                    key="bosch_hc_mode",
                    horizontal=True
                )
            with bosch_col2:
                bosch_Z_hom = st.number_input(
                    "HOM Z (kΩ)", value=10.0, min_value=0.1,
                    key="bosch_Z_hom",
                    help="Parasitic HOM impedance driving coupled-bunch instability"
                )

            try:
                bosch_result = bosch_analysis(
                    E_MeV=bosch_E_MeV,
                    alpha_c=bosch_alpha,
                    T0_s=bosch_T0,
                    sigma_E_over_E=bosch_sigma_E,
                    tau_rad_s=tau_rad_s,
                    V1_kV=vfcav_kv,
                    R1_MOhm=rshunt_mohm,
                    Q01=q0_fund,
                    beta1=beta_fund,
                    f_rf_MHz=f0_mhz,
                    R2_MOhm=rhshunt_mohm,
                    Q02=qh0,
                    beta2=0.0 if bosch_cavity_mode == "Passive" else 5.0,
                    nu=nh_harm,
                    I_mA=current_input,
                    Vs_kV=u0_kev,
                    mode='passive' if bosch_cavity_mode == "Passive" else 'active',
                    phi_z2_passive_deg=np.degrees(np.arctan(2 * qh0 * h_params['detuning_khz'] * 1e3 / (nh_harm * f0_mhz * 1e6))),
                    Z_hom_kOhm=bosch_Z_hom,
                    natural_sigma_t_ps=st.session_state.get('sigma_z0_mm', 3.0) / 0.3,  # mm -> ps
                )

                if bosch_result.get('valid', False):
                    s = bosch_result['summary']

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("HC Voltage", f"{s['V2_kV']:.1f} kV")
                    m2.metric("Bunch Length", f"{s['sigma_t_ps']:.0f} ps")
                    m3.metric("Lengthening", f"×{s['lengthening']:.2f}")
                    m4.metric("f_s", f"{s['f_s_kHz']:.2f} kHz")

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("ψ₁", f"{s['psi1_deg']:.1f}°")
                    m6.metric("ψ₂", f"{s['psi2_deg']:.1f}°")

                    rob_status = "✅ Stable" if s['robinson_stable'] else "❌ Unstable"
                    m7.metric("Robinson", rob_status)

                    cb_status = "✅ Damped" if s['cb_stable'] else "❌ Growing"
                    m8.metric("CB Instability", cb_status)

                    # Detailed results
                    with st.expander("Detailed Results", expanded=False):
                        det_c1, det_c2 = st.columns(2)
                        with det_c1:
                            st.markdown("**Robinson Analysis:**")
                            rob = bosch_result['robinson']
                            st.markdown(f"- Damping rate α_R = {rob['alpha_R']:.2e} s⁻¹")
                            st.markdown(f"- Robinson freq = {rob['Omega_robinson_kHz']:.3f} kHz")
                            st.markdown(f"- Eq. phase: {'Stable' if rob['eq_phase_stable'] else 'UNSTABLE'}")
                            st.markdown(f"- Tuning φ₁ = {rob['phi1_deg']:.1f}°, φ₂ = {rob['phi2_deg']:.1f}°")

                        with det_c2:
                            st.markdown("**Coupled-Bunch:**")
                            cb = bosch_result['coupled_bunch']
                            st.markdown(f"- Growth rate = {cb['growth_rate']:.2e} s⁻¹")
                            st.markdown(f"- Landau threshold = {cb['landau_threshold_Hz']:.1f} Hz")
                            st.markdown(f"- Form factor F = {cb['F_cb']:.4f}")
                            st.markdown(f"- Landau damped: {'Yes' if cb['landau_stable'] else 'No'}")

                        st.markdown("**Synchrotron Frequency:**")
                        sfs = bosch_result['synchrotron']
                        st.markdown(f"- f_s = {sfs['f_s_kHz']:.3f} kHz")
                        st.markdown(f"- Spread σf_s = {sfs['sigma_f_s_Hz']:.1f} Hz")
                        st.markdown(f"- Relative spread = {sfs['relative_spread']:.4f}")
                        st.markdown(f"- Potential type: {'Quadratic' if sfs['is_quadratic'] else 'Quartic (flat)'}")

                    # Overall status
                    if s['overall_stable']:
                        st.success(
                            f"**Overall: STABLE** at {current_input:.0f} mA. "
                            f"Bunch lengthened by ×{s['lengthening']:.1f}. "
                            "Robinson damped, coupled-bunch suppressed."
                        )
                    else:
                        issues = []
                        if not s['robinson_stable']:
                            issues.append("Robinson instability")
                        if not s['eq_phase_stable']:
                            issues.append("Equilibrium phase instability")
                        if not s['cb_stable']:
                            issues.append("Coupled-bunch instability")
                        st.error(
                            f"**UNSTABLE** at {current_input:.0f} mA: " +
                            ", ".join(issues)
                        )
                else:
                    st.error(f"Bosch analysis error: {bosch_result.get('error', 'Unknown')}")

            except Exception as bosch_err:
                st.error(f"Bosch model calculation error: {bosch_err}")

        with bosch_mode_tabs[1]:  # Current Scan
            st.markdown("### Growth Rates vs Beam Current (Bosch Model)")

            try:
                bosch_scan = scan_current_bosch(
                    E_MeV=bosch_E_MeV,
                    alpha_c=bosch_alpha,
                    T0_s=bosch_T0,
                    sigma_E_over_E=bosch_sigma_E,
                    tau_rad_s=tau_rad_s,
                    V1_kV=vfcav_kv,
                    R1_MOhm=rshunt_mohm,
                    Q01=q0_fund,
                    beta1=beta_fund,
                    f_rf_MHz=f0_mhz,
                    R2_MOhm=rhshunt_mohm,
                    Q02=qh0,
                    beta2=0.0 if bosch_cavity_mode == "Passive" else 5.0,
                    nu=nh_harm,
                    Vs_kV=u0_kev,
                    I_max_mA=float(st.session_state.get('imax_double_rf', 500)),
                    n_points=40,
                    mode='passive' if bosch_cavity_mode == "Passive" else 'active',
                )

                # Robinson damping rate vs current
                fig_rob = go.Figure()
                fig_rob.add_trace(go.Scatter(
                    x=bosch_scan['I_mA'], y=bosch_scan['robinson_rate'],
                    name='Robinson α_R',
                    line=dict(color='#4facfe', width=2)
                ))
                fig_rob.add_hline(y=0, line_dash='dash', line_color='red',
                                  annotation_text='Stability boundary')
                fig_rob.add_hline(y=bosch_scan['rad_damping_rate'],
                                  line_dash='dot', line_color='green',
                                  annotation_text='Radiation damping')
                fig_rob.add_vline(x=current_input, line_dash='dash',
                                  line_color='orange',
                                  annotation_text=f'I={current_input:.0f} mA')
                fig_rob.update_layout(
                    title='Robinson Damping Rate vs Beam Current',
                    xaxis_title='Beam Current (mA)',
                    yaxis_title='α_R (s⁻¹)',
                    template='plotly_white', height=400
                )
                st.plotly_chart(fig_rob, width='stretch')
                st.caption(
                    "Positive α_R = stable. Negative = Robinson instability. "
                    "Includes contributions from both fundamental and harmonic cavities."
                )

                # Bunch length and synchrotron frequency
                from plotly.subplots import make_subplots
                fig_bl = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Bunch Length', 'Synchrotron Frequency')
                )
                fig_bl.add_trace(go.Scatter(
                    x=bosch_scan['I_mA'], y=bosch_scan['sigma_t_ps'],
                    name='σ_t',
                    line=dict(color='#ff6b6b', width=2)
                ), row=1, col=1)
                fig_bl.add_trace(go.Scatter(
                    x=bosch_scan['I_mA'], y=bosch_scan['f_s_kHz'],
                    name='f_s',
                    line=dict(color='#51cf66', width=2)
                ), row=1, col=2)
                fig_bl.update_xaxes(title_text='Current (mA)', row=1, col=1)
                fig_bl.update_xaxes(title_text='Current (mA)', row=1, col=2)
                fig_bl.update_yaxes(title_text='σ_t (ps)', row=1, col=1)
                fig_bl.update_yaxes(title_text='f_s (kHz)', row=1, col=2)
                fig_bl.update_layout(
                    template='plotly_white', height=350, showlegend=False
                )
                st.plotly_chart(fig_bl, width='stretch')

            except Exception as scan_err:
                st.error(f"Current scan error: {scan_err}")

        with bosch_mode_tabs[2]:  # Theory
            st.markdown("### Bosch Model Theory")
            st.markdown(r"""
            **Key formulas from Bosch (1993) and Bosch (1997, PAC97):**

            #### 1. Robinson Damping Rate (Eq. 10/21)
            $$\alpha_R = \frac{4\alpha e I}{ET_0} \left[ F_1^2 R_1 Q_1 \tan\varphi_1 \cos^2\varphi_{1+} \cos^2\varphi_{1-} + F_2^2 R_2 Q_2 \tan\varphi_2 \cos^2\varphi_{2+} \cos^2\varphi_{2-} \right]$$

            where $\varphi_{i\pm}$ are the tuning angles at the synchrotron sidebands.

            #### 2. Equilibrium Phase Stability (PAC97 Eq. 19)
            $$F_1 V_1 \sin\psi_1 + \nu F_2 V_2 \sin\psi_2 > R_1 F_1^2 I \sin 2\varphi_1 + \nu R_2 F_2^2 I \sin 2\varphi_2$$

            #### 3. Coupled-Bunch Growth (Eq. 26)
            $$\Delta\omega_{CB} = \frac{e I \alpha \omega_{CB} Z(\omega_{CB}) F_{\omega_{CB}}^2}{2 E T_0 \omega_s}$$

            #### 4. Landau Damping Threshold
            - **Quadratic well**: $|\Delta\omega_{CB}| < 0.78\,\sigma\omega_s$ (Eq. from Krinsky & Wang)
            - **Quartic well**: $|\Omega_{CB}| < 0.6\,\Delta\omega_s$ (PAC97 Eq. 17)

            #### 5. Synchrotron Frequency Spread (Eq. 24)
            $$\sigma\omega_s = \omega_s (\omega_g \sigma_t)^2 \sqrt{\frac{8c^2}{a} + \frac{4b^2}{a^2}}$$

            where $a$, $b$, $c$ are the quadratic, cubic, and quartic potential coefficients.

            #### 6. Form Factor (Gaussian bunch)
            $$F(\omega) = \exp(-\omega^2 \sigma_t^2 / 2)$$

            ---
            **References:**
            - [1] R. A. Bosch & C. S. Hsue, Part. Accel. **42**, 81–99 (1993)
            - [2] R. A. Bosch, PAC97, pp. 862–864 (1997)
            - [3] S. Krinsky & J. M. Wang, Part. Accel. **17**, 109 (1985)
            """)

    # -- Sub-tab 4: Alves Instability Model --
    with subtabs_stab[4]:
        st.subheader("📊 Alves Instability Model (LMCI)")
        st.caption(
            "Primary Reference: F. H. de Sa and M. B. Alves, 'Analytical model for the longitudinal "
            "stability of double-RF systems...', Phys. Rev. Accel. Beams 26, 094402 (2023)."
        )
        
        if not ALBUMS_AVAILABLE:
            st.error("ALBuMS (mbtrack2/pycolleff) is not available for this analysis.")
            st.info("Check if `mbtrack2` and `pycolleff` are correctly installed in the environment.")
        else:
            try:
                # Extract parameters for Alves analysis
                with st.spinner("Calculating Alves Stability (LMCI)..."):
                    alves_res = alves_analysis(
                        E_GeV=energy,
                        alpha_c=alpha_c,
                        circumference_m=circumference,
                        h_rf=h_rf,
                        U0_keV=u0_kev,
                        damping_time_s=tau_rad_s,
                        v1_MV=vfcav_kv / 1e3,
                        rs1_MOhm=rshunt_mohm,
                        q1=q0_fund,
                        v2_MV=res['vh_opt'] / 1e3,
                        rs2_MOhm=rhshunt_mohm,
                        q2=qh0,
                        nh_harm=nh_harm,
                        i_beam_mA=current_input,
                        phi_h_deg=h_params['phi_hs'], # Phase of HC
                        passive_hc=(hc_mode == 'Passive')
                    )
                
                if alves_res['success']:
                    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                    col_a1.metric("Bunch Length", f"{alves_res['bunch_length_ps']:.1f} ps")
                    col_a2.metric("R-Factor", f"{alves_res['r_factor']:.2f}")
                    
                    fmci_status = "✅ Stable" if not alves_res['robinson_unstable'] else "❌ UNSTABLE"
                    col_a3.metric("FMCI Stability", fmci_status)
                    
                    ptbl_status = "✅ Stable" if not alves_res['ptbl_unstable'] else "❌ UNSTABLE"
                    col_a4.metric("PTBL Stability", ptbl_status)
                    
                    if not alves_res['robinson_unstable'] and not alves_res['ptbl_unstable']:
                        st.success(f"**Overall: STABLE** at {current_input:.0f} mA (Alves LMCI method).")
                    else:
                        if alves_res['robinson_unstable']:
                            st.error("❌ **FMCI Detected**: Fast Mode Coupling Instability (FMCI) predicted.")
                        if alves_res['ptbl_unstable']:
                            st.warning("⚠️ **PTBL Detected**: Periodic Transient Beam Loading instability predicted.")
                            
                    with st.expander("Alves Model Details & Theory"):
                        st.markdown(r"""
                        The **Alves method** (2023) generalizes the longitudinal stability analysis for double-RF systems 
                        by treating the problem as a Mode Coupling Instability (LMCI). 
                        
                        Key features:
                        - **FMCI**: Fast Mode Coupling Instability, which generalizes the classic Robinson instability 
                          to include the effect of potential well distortion and mode coupling.
                        - **PTBL**: Periodic Transient Beam Loading instability, particularly relevant for gaps in filling patterns.
                        - **Xi ($\xi$ parameter)**: Measures the ratio between beam load and RF restoring force.
                        """)
                        st.latex(r"\xi = \frac{I_b R_{sh}}{V_{rf} \cos \phi_{rf}}")
                        st.markdown(f"**Current Operating conditions:** $\\xi \\approx {alves_res['xi']:.3f}$.")
                else:
                    st.error(f"Alves analysis error: {alves_res['error']}")
            except Exception as alves_err:
                st.error(f"Alves calculation failed: {alves_err}")

    # -- Sub-tab 5: J. Jacob Robinson Analysis --
    with subtabs_stab[5]:
        st.subheader("🎓 J. Jacob DC Robinson Analysis")
        st.caption(
            "Primary Reference: J. Jacob, 'Passive vs Active Systems, DC Robinson, DLLRF', "
            "HarmonLIP'2022 Workshop, MAX IV, Lund, Sweden (Oct. 2022)."
        )

        st.markdown(r"""
        The **DC Robinson Stability** (or Phase Stability) is evaluated using the Robinson term $M_{total}$. 
        Stability is achieved in the zero-frequency limit ($\Omega \to 0$) when the total longitudinal restoring 
        force derivative is positive ($M_{total} > 0$).
        """)

        # Prepare Jacob parameters
        v1_v = vfcav_kv * 1e3
        rs1_ohm = rshunt_mohm * 1e6
        ql1 = q0_fund / (1 + beta_fund)
        psi1_rad = np.arctan(2 * ql1 * f_params['detuning_khz'] * 1e3 / (f0_mhz * 1e6))
        phi1_rad = res['phi_s_rad']
        g1_fb = rf_gain

        v2_v = res['vh_opt'] * 1e3
        rs2_ohm = rhshunt_mohm * 1e6
        ql2 = qh0 
        psi2_rad = np.arctan(2 * ql2 * h_params['detuning_khz'] * 1e3 / (f0_mhz * nh_harm * 1e6))
        phi2_rad = np.radians(h_params['phi_hs'])

        jacob_res = jacob_robinson_analysis(
            alpha=alpha_c, energy_ev=energy*1e9, h1=h_rf, u0_ev=u0_kev*1e3,
            v1_v=v1_v, rs1_ohm=rs1_ohm, ql1=ql1, psi1_rad=psi1_rad, phi1_rad=phi1_rad, g1=g1_fb,
            v2_v=v2_v, rs2_ohm=rs2_ohm, ql2=ql2, psi2_rad=psi2_rad, phi2_rad=phi2_rad, n_h=nh_harm,
            g2=0.0,
            i_beam_a=current_input/1e3
        )

        # Display Results
        col_j1, col_j2, col_j3 = st.columns(3)
        col_j1.metric("MC Term M₁", f"{jacob_res['m1']:.2e}")
        col_j2.metric("HC Term M₂", f"{jacob_res['m2']:.2e}")
        
        status_color = "normal" if jacob_res['is_stable'] else "inverse"
        col_j3.metric(
            "Total M", 
            f"{jacob_res['m_total']:.2e}", 
            delta="Stable" if jacob_res['is_stable'] else "UNSTABLE",
            delta_color=status_color
        )

        if jacob_res['is_stable']:
            st.success(f"✅ **DC Robinson Stable**: Restoring force factor M = {jacob_res['m_total']:.2e} > 0.")
        else:
            st.error(f"❌ **DC Robinson UNSTABLE**: Total restoring force negative (M = {jacob_res['m_total']:.2e}).")

        # Scan vs Current
        st.markdown("---")
        st.markdown("#### Robinson Margin vs. Beam Current")
        i_range_j = np.linspace(0, imax_ma, 100)
        j_scan = scan_current_jacob(
            i_range_j, alpha_c, energy*1e9, h_rf, u0_kev*1e3,
            v1_v, rs1_ohm, ql1, psi1_rad, phi1_rad, g1_fb,
            v2_v, rs2_ohm, ql2, psi2_rad, phi2_rad, nh_harm, 0.0
        )

        fig_j = go.Figure()
        fig_j.add_trace(go.Scatter(x=i_range_j, y=j_scan['m1'], name="M₁ (Main Cavity)", line=dict(dash='dot', color='blue')))
        fig_j.add_trace(go.Scatter(x=i_range_j, y=j_scan['m2'], name="M₂ (Harmonic)", line=dict(dash='dot', color='orange')))
        fig_j.add_trace(go.Scatter(x=i_range_j, y=j_scan['m_total'], name="M_total", line=dict(width=3, color='black')))
        fig_j.add_hline(y=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig_j, width='stretch')

        with st.expander("J. Jacob Model Theory", expanded=False):
            st.markdown(r"""
            #### Robinson Stability in Multi-Cavity Systems
            
            For a double RF system, the zero-frequency stability condition (DC Robinson) 
            is generalized by summing the contributions of all cavities.
            """)
            st.latex(r"M_k = \frac{n_k}{1+G_k} \left[ \frac{V_k \cos \phi_k - I_b R_{sk} \sin(2\psi_k)}{R_{sk} \cos^2 \psi_k} \right]")
    st.markdown(r"""
    Where:
    - $n_k$: harmonic ratio ($n_1=1$, $n_2=n_h$)
    - $G_k$: RF feedback gain
    - $V_k, \phi_k$: Peak voltage and phase (relative to beam)
    - $I_b, R_{sk}, \psi_k$: Beam current, shunt impedance, and tuning angle
    
    Stability requires $\sum M_k > 0$. 
    
    **Key Insights from J. Jacob:**
    1. **RF Feedback (DRFB)** significantly enhances stability by dividing the destabilizing beam-loading term by $(1+G)$.
    2. **Active systems** allow better control over $\phi_k$ and $V_k$, pushing the Robinson limit higher than passive systems.
    3. For **passive cavities**, the detuning $\psi_k$ must be carefully chosen to avoid the negative contribution to $M$ from beam loading.
    """)

    # === Sub-tab 6: Venturini Model ===
    with subtabs_stab[6]:
        st.subheader("📚 Theory & Validation — Venturini (2018)")
        st.caption(
            "Reference: M. Venturini, *Passive higher-harmonic rf cavities with general settings "
            "and multibunch instabilities in electron storage rings*, "
            "Phys. Rev. Accel. Beams **21**, 114404 (2018)."
        )

        theory_subtabs = st.tabs([
        "\U0001f6a6 Operating Regime",
        "\u2705 Formula Validation",
        "\u26a1 Beam Loading Explorer",
        "\U0001f4d6 User Guide"
        ])

    # Common calculations for this tab
        V1_V_theory = vfcav_kv * 1e3
        U0_eV_theory = u0_kev * 1e3
        I0_A_theory = current_input / 1e3
        Rs_HC_Ohm_theory = rhshunt_mohm * 1e6
        f_rf_Hz_theory = f0_mhz * 1e6

        fp_result = flat_potential_conditions(V1_V_theory, U0_eV_theory, nh_harm)
        det_result = required_detuning_flat_potential(
        I0_A_theory, Rs_HC_Ohm_theory, qh0, f_rf_Hz_theory, nh_harm,
        fp_result['V2'], fp_result['psi']
        )
        regime_result = classify_operating_regime(
        V1_V_theory, U0_eV_theory, nh_harm,
        res['vh_opt'] * 1e3,
        np.radians(res['phi_h_opt'])
        )

    # Sub-tab 1: Operating Regime
    with theory_subtabs[0]:
        st.markdown("### \U0001f6a6 Current Operating Regime")
        st.markdown(
            "The Venturini model classifies operating points relative to the "
            "**flat-potential condition**, where the first and second derivatives of "
            "the total RF voltage vanish at the synchronous phase."
        )

        regime_name = regime_result['regime']
        if "Flat" in regime_name:
            st.success(f"### \u2705 {regime_name}")
        elif "Under" in regime_name:
            st.warning(f"### \u26a0\ufe0f {regime_name}")
        else:
            st.error(f"### \u274c {regime_name}")

        col_reg1, col_reg2, col_reg3 = st.columns(3)
        col_reg1.metric(
            "V2 Ratio (actual / flat)",
            f"{regime_result['V2_ratio']:.3f}",
            delta=f"{(regime_result['V2_ratio'] - 1)*100:.1f}%",
            delta_color="normal" if abs(regime_result['V2_ratio'] - 1) < 0.2 else "inverse"
        )
        col_reg2.metric("V2 (Flat Potential)", f"{regime_result['V2_flat_kV']:.2f} kV")
        col_reg3.metric("V2 (Current App)", f"{regime_result['V2_actual_kV']:.2f} kV")

        st.info(f"**Description:** {regime_result['description']}")
        st.markdown(f"**Recommendation:** {regime_result['recommendation']}")

        st.markdown("---")
        st.markdown("### Detuning Feasibility")
        if det_result['achievable']:
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Required Detuning", f"{det_result['delta_f_kHz']:.3f} kHz")
            col_d2.metric("Tuning Angle", f"{det_result.get('phi_z_deg', 0):.1f} deg")
            col_d3.metric("Min Current for Flat", f"{det_result['I_min_mA']:.1f} mA")
            if I0_A_theory * 1e3 < det_result['I_min_mA']:
                st.warning(
                    f"Current ({current_input:.0f} mA) is below minimum "
                    f"({det_result['I_min_mA']:.1f} mA) needed for flat potential."
                )
            else:
                st.success(
                    f"Beam current ({current_input:.0f} mA) is sufficient. "
                    f"Flat potential achievable with df = {det_result['delta_f_kHz']:.3f} kHz."
                )
        else:
            st.error(
                f"Flat potential not achievable at {current_input:.0f} mA. "
                f"Minimum current required: {det_result['I_min_mA']:.1f} mA."
            )

        st.markdown("---")
        st.markdown("### Regime Map: V2 vs Beam Current")
        i_scan_theory = np.linspace(10, float(st.session_state.get('imax_double_rf', 500)), 80)
        v2_flat_scan = []
        v2_induced_scan = []
        for i_sv in i_scan_theory:
            fp_sv = flat_potential_conditions(V1_V_theory, U0_eV_theory, nh_harm)
            det_sv = required_detuning_flat_potential(
                i_sv / 1e3, Rs_HC_Ohm_theory, qh0, f_rf_Hz_theory, nh_harm,
                fp_sv['V2'], fp_sv['psi']
            )
            v2_flat_scan.append(abs(fp_sv['V2_kV']))
            if det_sv['achievable']:
                bl_sv = passive_cavity_beam_loading(
                    i_sv / 1e3, Rs_HC_Ohm_theory, qh0, f_rf_Hz_theory, nh_harm, det_sv['delta_f_Hz']
                )
                v2_induced_scan.append(bl_sv['V_ind_kV'])
            else:
                v2_max_sv = 2 * (i_sv / 1e3) * Rs_HC_Ohm_theory / 1e3
                v2_induced_scan.append(v2_max_sv)

        fig_regime = go.Figure()
        fig_regime.add_trace(go.Scatter(
            x=i_scan_theory, y=v2_flat_scan,
            name='V2 (Flat Potential Target)', line=dict(color='red', width=2, dash='dash')
        ))
        fig_regime.add_trace(go.Scatter(
            x=i_scan_theory, y=v2_induced_scan,
            name='V2 (Max Beam-Induced)', line=dict(color='blue', width=2),
            fill='tonexty', fillcolor='rgba(135,206,235,0.15)'
        ))
        fig_regime.add_trace(go.Scatter(
            x=[current_input], y=[res['vh_opt']],
            name='Current Operating Point', mode='markers',
            marker=dict(size=14, color='green', symbol='star', line=dict(width=2, color='black'))
        ))
        fig_regime.add_annotation(
            x=current_input, y=res['vh_opt'],
            text=f"  {regime_result['regime']}",
            showarrow=True, arrowhead=2, ax=60, ay=-40
        )
        fig_regime.update_layout(
            title="Operating Regime Map",
            xaxis_title="Beam Current (mA)",
            yaxis_title="Harmonic Voltage V2 (kV)",
            template="plotly_white", height=450,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_regime, width='stretch')
        st.caption(
            "Blue region: achievable range. Red dashed: flat-potential target. "
            "Star: current operating point."
        )

    # Sub-tab 2: Formula Validation
    with theory_subtabs[1]:
            st.markdown("### Formula Validation: App vs. Venturini Model")
            st.markdown(
            "Compares the app's physics engine with the analytic Venturini (2018) "
            "flat-potential theory."
            )

            validation = validate_against_app(
            V1_kV=vfcav_kv, U0_keV=u0_kev, m=nh_harm,
            I0_mA=current_input, Rs_HC_MOhm=rhshunt_mohm,
            Q0_HC=qh0, f_rf_MHz=f0_mhz, n_HC=nhcav,
            app_phi_s_deg=res['phi_s'], app_Vh_kV=res['vh_opt'],
            app_detuning_kHz=h_params['detuning_khz'],
            app_phi_h_deg=res['phi_h_opt']
            )

            comps = validation['comparisons']
            meta_v = validation['meta']

            st.markdown(
            f"**Overall Score:** {meta_v['n_matches']}/{meta_v['n_comparisons']} parameters match "
            f"({'All Good' if meta_v['n_matches'] == meta_v['n_comparisons'] else 'Discrepancies Found'})"
            )

            import pandas as pd
            val_rows = []
            for key, comp in comps.items():
                status = "Pass" if comp['match'] else "FAIL"
            val_rows.append({
                'Status': status,
                'Parameter': comp['parameter'],
                'Venturini': f"{comp['venturini']:.4f}" if isinstance(comp['venturini'], float) else str(comp['venturini']),
                'App Engine': f"{comp['app']:.4f}" if isinstance(comp['app'], float) else str(comp['app']),
                'Delta': f"{comp['delta']:.4f}" if isinstance(comp['delta'], float) else str(comp['delta']),
                'Unit': comp['unit'],
            })

            df_val = pd.DataFrame(val_rows)
            st.dataframe(df_val, width='stretch', hide_index=True)

            with st.expander("Formula Details", expanded=False):
                st.markdown("#### Synchronous Phase")
            st.latex(r"\phi_s = \pi - \arcsin\!\left[\frac{m^2}{m^2-1} \cdot \frac{U_0}{V_1}\right]")
            st.markdown(
                f"- Input: V1 = {vfcav_kv:.1f} kV, U0 = {u0_kev:.1f} keV, m = {nh_harm}\n"
                f"- Result: {fp_result['phi_s_deg']:.2f} deg (Venturini) vs {res['phi_s']:.2f} deg (App)"
            )

            st.markdown("#### Harmonic Voltage (Flat Potential)")
            st.latex(r"V_2 = -\frac{V_1 \cos\phi_s}{m^2 \cos\psi}")
            st.markdown(
                f"- Result: V2 = {abs(fp_result['V2_kV']):.2f} kV (Venturini) vs {res['vh_opt']:.2f} kV (App)"
            )
            st.info(
                "The app uses the equivalent formula: "
                "V_h,opt = sqrt(Vc^2/n^2 - U0^2/(n^2-1)), "
                "which gives the same result under flat-potential assumptions."
            )

            st.markdown("#### Harmonic Phase")
            st.latex(r"\tan\psi = -\frac{\tan\phi_s}{m}")
            st.markdown(
                f"- Result: psi = {fp_result['psi_deg']:.2f} deg (Venturini) vs {res['phi_h_opt']:.2f} deg (App)"
            )

            st.markdown("#### Passive Cavity Detuning")
            st.latex(r"\delta f = \frac{\tan\phi_z \cdot m f_{\mathrm{rf}}}{2 Q_0}, \quad \cos\phi_z = \frac{|V_2|}{2 I_0 R_s}")
            st.markdown(
                f"- Current: I0 = {current_input:.0f} mA, Rs = {rhshunt_mohm:.2f} MOhm, Q0 = {qh0:.0f}\n"
                f"- Result: df = {det_result['delta_f_kHz']:.3f} kHz (Venturini) vs {abs(h_params['detuning_khz']):.3f} kHz (App)"
            )

    # Sub-tab 3: Beam Loading Explorer
    with theory_subtabs[2]:
            st.markdown("### Beam Loading Explorer")
            st.markdown(
            "Explore how the passive harmonic cavity voltage and phase change with "
            "detuning and beam current."
            )

            col_bl1, col_bl2 = st.columns(2)

            with col_bl1:
                st.markdown("#### V2 vs. Detuning (Fixed Current)")
            det_range_khz = np.linspace(-200, 200, 200)
            v2_vs_det = []
            phi_z_vs_det = []
            for df_khz_val in det_range_khz:
                bl_res_val = passive_cavity_beam_loading(
                    I0_A_theory, Rs_HC_Ohm_theory, qh0, f_rf_Hz_theory, nh_harm, df_khz_val * 1e3
                )
                v2_vs_det.append(bl_res_val['V_ind_kV'])
                phi_z_vs_det.append(bl_res_val['phi_z_deg'])

            fig_bl1 = go.Figure()
            fig_bl1.add_trace(go.Scatter(
                x=det_range_khz, y=v2_vs_det,
                name='Induced V2', line=dict(color='#4facfe', width=2)
            ))
            fig_bl1.add_hline(
                y=abs(fp_result['V2_kV']),
                line_dash='dash', line_color='red',
                annotation_text=f"Flat Potential: {abs(fp_result['V2_kV']):.1f} kV"
            )
            if det_result['achievable']:
                fig_bl1.add_vline(
                    x=det_result['delta_f_kHz'],
                    line_dash='dot', line_color='green',
                    annotation_text=f"df = {det_result['delta_f_kHz']:.1f} kHz"
                )
            fig_bl1.update_layout(
                xaxis_title="Detuning df (kHz)",
                yaxis_title="Induced Voltage (kV)",
                template="plotly_white", height=400,
                title=f"Beam-Induced HC Voltage (I = {current_input:.0f} mA)"
            )
            st.plotly_chart(fig_bl1, width='stretch')

            with col_bl2:
                st.markdown("#### Tuning Angle vs. Detuning")
            fig_bl2 = go.Figure()
            fig_bl2.add_trace(go.Scatter(
                x=det_range_khz, y=phi_z_vs_det,
                name='Tuning Angle', line=dict(color='#ff7b00', width=2)
            ))
            if det_result['achievable']:
                fig_bl2.add_vline(
                    x=det_result['delta_f_kHz'],
                    line_dash='dot', line_color='green',
                    annotation_text="Flat Potential"
                )
            fig_bl2.update_layout(
                xaxis_title="Detuning df (kHz)",
                yaxis_title="Tuning Angle (deg)",
                template="plotly_white", height=400,
                title="Impedance Phase (Tuning Angle)"
            )
            st.plotly_chart(fig_bl2, width='stretch')

            st.markdown("---")
            st.markdown("#### V2 vs. Beam Current (at Flat-Potential Detuning)")
            i_range_bl = np.linspace(10, float(st.session_state.get('imax_double_rf', 500)), 80)
            v2_vs_i = []
            det_vs_i = []
            for i_val_bl in i_range_bl:
                det_i_bl = required_detuning_flat_potential(
                i_val_bl / 1e3, Rs_HC_Ohm_theory, qh0, f_rf_Hz_theory, nh_harm,
                fp_result['V2'], fp_result['psi']
            )
            det_vs_i.append(det_i_bl['delta_f_kHz'])
            if det_i_bl['achievable']:
                bl_i_val = passive_cavity_beam_loading(
                    i_val_bl / 1e3, Rs_HC_Ohm_theory, qh0, f_rf_Hz_theory, nh_harm, det_i_bl['delta_f_Hz']
                )
                v2_vs_i.append(bl_i_val['V_ind_kV'])
            else:
                v2_vs_i.append(2 * (i_val_bl / 1e3) * Rs_HC_Ohm_theory / 1e3)

            col_bi1, col_bi2 = st.columns(2)
            with col_bi1:
                fig_vi = go.Figure()
            fig_vi.add_trace(go.Scatter(
                x=i_range_bl, y=v2_vs_i,
                name='V2 (beam-induced)', line=dict(color='blue', width=2)
            ))
            fig_vi.add_hline(
                y=abs(fp_result['V2_kV']),
                line_dash='dash', line_color='red',
                annotation_text=f"Target: {abs(fp_result['V2_kV']):.1f} kV"
            )
            fig_vi.update_layout(
                title="Induced Voltage vs Current",
                xaxis_title="Beam Current (mA)",
                yaxis_title="V2 (kV)",
                template="plotly_white", height=380
            )
            st.plotly_chart(fig_vi, width='stretch')

            with col_bi2:
                fig_di = go.Figure()
            fig_di.add_trace(go.Scatter(
                x=i_range_bl, y=det_vs_i,
                name='Required Detuning', line=dict(color='orange', width=2)
            ))
            fig_di.update_layout(
                title="Required Detuning vs Current",
                xaxis_title="Beam Current (mA)",
                yaxis_title="Detuning df (kHz)",
                template="plotly_white", height=380
            )
            st.plotly_chart(fig_di, width='stretch')

            st.info(
            "As the beam current increases, less detuning is needed "
            "to achieve the flat-potential voltage. At the minimum current, the cavity "
            "must be exactly on resonance (df -> 0)."
            )

    # Sub-tab 4: User Guide
    with theory_subtabs[3]:
            st.markdown("### User Guide: Double RF System Operation")
            st.markdown(
            "*Based on: M. Venturini, Phys. Rev. Accel. Beams 21, 114404 (2018)*"
            )

            st.markdown("""
    #### 1. Purpose of the Harmonic Cavity System
    
            In fourth-generation light sources, the beam emittance is extremely low, making 
            **Touschek scattering** and **intra-beam scattering** the dominant lifetime-limiting 
            effects. A harmonic cavity (HC) system is introduced to **lengthen the bunches** by 
            a factor of 3-5x, thereby reducing the charge density and mitigating these effects.

    #### 2. The Flat Potential Condition

            The optimal operating point requires both the first and second derivatives
            of the total RF voltage to vanish at the synchronous phase. This creates a 
            **quartic potential well** instead of the usual quadratic well, leading to:
                - Maximum bunch lengthening
            - Large synchrotron frequency spread (strong Landau damping)
            - Vanishing linear synchrotron frequency at the bunch center

    #### 3. Key Formulas (Venturini 2018)

            | Formula | Expression | Physical Meaning |
            |---------|-----------|------------------|
            | Sync Phase | phi_s = pi - arcsin[m^2/(m^2-1) * U0/V1] | Phase at which energy gain = energy loss |
            | HC Voltage | V2 = -V1*cos(phi_s) / (m^2*cos(psi)) | Required harmonic voltage for flat potential |
            | HC Phase | tan(psi) = -tan(phi_s)/m | Required harmonic phase |
            | Detuning | df = tan(phi_z)*m*f_rf / (2*Q0) | Cavity detuning to achieve required voltage |
            | Tuning Angle | cos(phi_z) = |V2| / (2*I0*Rs) | Relates detuning to beam-induced voltage |

    #### 4. Operating Regimes

            | Regime | V2/V2(flat) | Bunch Shape | Stability |
            |--------|------------|-------------|----------|
            | **Under-stretched** | < 0.8 | Near-Gaussian | More stable, less lengthening |
            | **Flat Potential** | 0.8 - 1.2 | Flat-topped | Optimal Landau damping |
            | **Over-stretched** | > 1.2 | Double-peaked | Risk of instability |

    #### 5. Practical Operation Guide

            **Step-by-step for passive HC operation:**

            1. **Set the main RF voltage** V1 to provide adequate overvoltage.
    
            2. **Check minimum current:** The passive HC requires I_min = |V2| / (2*Rs).
               Below this current, flat potential is **not achievable**.
    
            3. **Adjust HC detuning**: Use the tuner to set df. 
               The detuning must be **positive** (above resonance) for Robinson stability.
    
            4. **Monitor regime:** Use the Operating Regime tab to verify 
               you are in the flat-potential regime (V2 ratio near 1.0).
    
            5. **During injection ramp:** As current increases, **reduce detuning** progressively 
               to maintain the flat-potential condition.

    #### 6. Stability Considerations

            - **Robinson Instability:** HC must be detuned to high-frequency side.
            - **Coupled-Bunch Instability:** HC narrow-band impedance can drive CBI. 
              HOM dampers are essential.
            - **Landau Damping:** The quartic potential provides large frequency spread that 
              naturally suppresses coupled-bunch instabilities.

    #### 7. Validation with This App

            Use the **Formula Validation** sub-tab to compare the app's physics engine 
            with the Venturini analytic model. Key checks:
                - Synchronous phase should match to < 1 deg
            - Harmonic voltage should match to < 5%
            - Detuning should match to < 10%
            """)

            st.markdown("---")
            st.markdown("#### References")
            st.markdown("""
            1. **M. Venturini** (2018). *Passive higher-harmonic rf cavities with general settings 
               and multibunch instabilities in electron storage rings*. 
               Phys. Rev. Accel. Beams **21**, 114404.
    
            2. **A. Hofmann & S. Myers** (1980). *Beam dynamics in a double RF system*. 
               Proc. 11th Int. Conf. on High-Energy Accelerators.
    
            3. **R. Warnock et al.** (2020). *Equilibrium of an arbitrary bunch train in 
               presence of a passive harmonic cavity*. Phys. Rev. Accel. Beams.
    
            4. **P. Borowiec et al.** (2025). *28th ESLS RF Workshops*. 
               SOLEIL II Design Parameters.
            """)

    # ── Sub-tab 7: Hofmann Analytic Model ──
    with subtabs_stab[7]:
        st.subheader("🎓 Hofmann & Myers Analytic Model")
        st.caption("Primary Reference: A. Hofmann and S. Myers, 'Beam dynamics in a double RF system', Proc. 11th Int. Conf. on High-Energy Accelerators (1980).")
        st.markdown("""
        This section implements the **Hofmann-Myers model** for flat potential conditions. 
        It solves for the exact harmonic voltage and phase needed to cancel the slope and curvature of the total RF voltage at the synchronous phase.
        """)
    
        try:
            # Initialize Hofmann Model with current dashboard parameters
            # Note: Dashboard uses keV for U0, kV for V1. Converter to eV/V for model.
            hofmann = HofmannModel(
                E0=energy * 1e9,
                alpha_c=float(ring_params.get("momentum_compaction", 1.0e-4)),
                U0=u0_kev * 1e3,
                V1=vfcav_kv * 1e3,
                h=h_rf,
                n=nh_harm,
                sigma_delta=float(ring_params.get("sigma_delta", 1.0e-3))
            )
            hofmann.set_flat_potential()
        
            # Calculate comparison data
            phi_rad = np.linspace(-0.6, 0.6, 1200)
            phi_deg = np.degrees(phi_rad)
        
            v_tot_hof = hofmann.get_voltage(phi_rad)
            pot_hof = hofmann.get_potential(phi_rad)
            dist_hof = hofmann.get_distribution(phi_rad)
        
            # Single RF Comparison
            hof_s = HofmannModel(hofmann.E0, hofmann.alpha_c, hofmann.U0, hofmann.V1, hofmann.h, hofmann.n, hofmann.sigma_delta)
            hof_s.phi_s1 = np.pi - np.arcsin(hofmann.U0 / hofmann.V1)
            hof_s.V2 = 0
            pot_s_hof = hof_s.get_potential(phi_rad)
            dist_s_hof = hof_s.get_distribution(phi_rad)
        
            col_m1, col_m2 = st.columns(2)
        
            with col_m1:
                st.markdown("#### Potential Well & Voltage")
                fig_hof1 = make_subplots(specs=[[{"secondary_y": True}]])
                fig_hof1.add_trace(go.Scatter(x=phi_deg, y=v_tot_hof/1e3, name="Voltage (kV)", line=dict(color='black', width=1, dash='dot')), secondary_y=False)
                fig_hof1.add_trace(go.Scatter(x=phi_deg, y=pot_hof, name="Potential (Double)", line=dict(color='red', width=2)), secondary_y=True)
                fig_hof1.add_trace(go.Scatter(x=phi_deg, y=pot_s_hof, name="Potential (Single)", line=dict(color='blue', dash='dash')), secondary_y=True)
                fig_hof1.update_layout(title="Potential Well Comparison", xaxis_title="Phase (deg)", template="plotly_white")
                fig_hof1.update_yaxes(title_text="Voltage [kV]", secondary_y=False)
                fig_hof1.update_yaxes(title_text="Potential [arb]", secondary_y=True)
                st.plotly_chart(fig_hof1, width='stretch')
            
            with col_m2:
                st.markdown("#### Bunch Distribution")
                fig_hof2 = go.Figure()
                fig_hof2.add_trace(go.Scatter(x=phi_deg, y=dist_hof, name="Flat-topped (Double)", fill='tozeroy', line=dict(color='red')))
                fig_hof2.add_trace(go.Scatter(x=phi_deg, y=dist_s_hof, name="Gaussian (Single)", line=dict(color='blue', dash='dash')))
                fig_hof2.update_layout(title="Bunch Density Profile", xaxis_title="Phase (deg)", yaxis_title="Normalized Density", template="plotly_white")
                st.plotly_chart(fig_hof2, width='stretch')
            
            st.markdown("---")
            st.markdown("#### Synchrotron Frequency Spread")
            phi_amps = np.linspace(0.0, 0.3, 30)
            fs_double = hofmann.get_synchrotron_frequency(phi_amps)
            qs0_hof = np.sqrt(hofmann.h * hofmann.alpha_c * hofmann.V1 * abs(np.cos(hof_s.phi_s1)) / (2.0 * np.pi * hofmann.E0))
        
            fig_hof3 = go.Figure()
            fig_hof3.add_trace(go.Scatter(x=np.degrees(phi_amps), y=fs_double/qs0_hof, mode='lines+markers', name="Double RF", line=dict(color='orange')))
            fig_hof3.add_trace(go.Scatter(x=np.degrees(phi_amps), y=[1.0]*len(phi_amps), name="Single RF (Fixed)", line=dict(color='blue', dash='dash')))
            fig_hof3.update_layout(
                title="Synchrotron Frequency vs Amplitude", 
                xaxis_title="Amplitude (deg)", 
                yaxis_title="fs / fs0",
                template="plotly_white",
                yaxis=dict(range=[0, 1.2])
            )
            st.plotly_chart(fig_hof3, width='stretch')
        
            st.success(fr"**Flat Potential Solution:** $k = V_2/V_1 = {hofmann.V2/hofmann.V1:.4f}$, $\phi_{{s,main}} = {np.degrees(hofmann.phi_s1):.2f}^\circ$, $\phi_{{s,harm}} = {np.degrees(hofmann.phi_s2):.2f}^\circ$")
    
        except Exception as e:
            st.error(f"Hofmann Model Error: {e}")
    
    # === Tab 4: Detailed Data & Theory ===
    with tabs[3]:
        st.subheader("Detailed Analysis & Physics Background")
        
        subtabs_det = st.tabs(["Harmonic Analysis Data", "Physics Interpretation"])
        
        with subtabs_det[0]: # Harmonic Analysis Data
            col_h1, col_h2, col_h3 = st.columns(3)
            col_h1.metric("Optimal Vh", f"{res['vh_opt']:.2f} kV")
            col_h2.metric("Vh per Cavity", f"{res['vh_cav']:.2f} kV")
            col_h3.metric("Opt Phase", f"{res['phi_h_opt']:.2f}°")
            
            st.markdown("**Formulas:**")
            st.latex(r"V_{h,\mathrm{opt}} = \sqrt{\frac{V_c^2}{n^2} - \frac{U_0^2}{(n^2-1)}}")
            
        with subtabs_det[1]: # Physics Interpretation
            # (Content from previous physics tab)
            st.markdown("### Passive Harmonic Cavity Physics")
            st.latex(r"\Delta f = \frac{f_r}{2 Q_L} \sqrt{ \frac{4 I_{DC}^2 R_{sh}^2}{V_{hc}^2} - 1 }")
            st.info("See full documentation in project wiki.")
    
    # === Tab 5: Voltage Calibration ===
    with tabs[4]:
        st.subheader("🛠️ RF Voltage Calibration (Beam-Based)")
        st.markdown("""
        This tool calibrates the RF cavity voltage by measuring the **synchrotron frequency ($f_s$)**. 
        The beam serves as a high-precision probe to deduce the actual gap voltage.
        """)
        
        subtabs_cal = st.tabs(["🎯 Beam-Based (Synch. Freq)", "⚡ Power-Based (Decay Method)"])
        
        # Get current machine parameters from dashboard state
        # Convert units: GeV to eV, keV to V, MHz to Hz
        f0_hz_cal = f0_mhz * 1e6
        h_val_cal = h_rf
        alpha_val_cal = alpha_c
        e0_val_cal = energy * 1e9
        u0_val_cal = u0_kev * 1e3
        
        # Initialize calibrator with current machine state
        calibrator = RFVoltageCalibrator(
    f0=f0_hz_cal,
    h=h_val_cal,
    alpha=alpha_val_cal,
    E0=e0_val_cal,
    U0=u0_val_cal
        )
    
        with subtabs_cal[0]:
            col_cal1, col_cal2 = st.columns(2)
            
            with col_cal1:
                st.markdown("#### 1. Input Measurements")
                v_read_kv = st.number_input(
                    "App Measured Total Voltage (kV)",
                    min_value=10.0,
                    max_value=30000.0,
                    value=vfcav_kv,
                    step=1.0,
                    key="cal_v_read",
                    help="The sum of voltages read from the LLRF system (V1 + V2 + ...)"
                )
                
                fs_meas_hz = st.number_input(
                    "Measured Sync Frequency $f_s$ (Hz)",
                    min_value=10.0,
                    max_value=100000.0,
                    value=3354.0,
                    step=1.0,
                    key="cal_fs_meas",
                    help="Synchrotron frequency measured from the beam spectrum"
                )
                
                # Perform calculation
                try:
                    v_actual_v = calibrator.calculate_calibrated_voltage(fs_meas_hz)
                    k_factor = calibrator.get_calibration_factor(v_read_kv * 1e3, fs_meas_hz)
                    error_pct = (k_factor - 1) * 100
                    
                    st.markdown("---")
                    st.markdown("#### 2. Calibration Results")
                    res_col_res1, res_col_res2 = st.columns(2)
                    res_col_res1.metric("Actual Voltage", f"{v_actual_v/1e3:.3f} kV", delta=f"{v_actual_v/1e3 - v_read_kv:.2f} kV")
                    res_col_res2.metric("Calibration Factor", f"{k_factor:.4f}", delta=f"{error_pct:.2f}%", delta_color="inverse")
                    
                    if abs(error_pct) > 5:
                        st.warning(f"⚠️ High error detected ({error_pct:.1f}%).")
                    else:
                        st.success("✅ Calibration within expected range.")
                        
                except Exception as e:
                    st.error(f"Calibration Error: {str(e)}")
            
            with col_cal2:
                st.markdown("#### 3. Theoretical $f_s$ vs. Voltage")
                # Generate theoretical curve
                v_range_kv_cal = np.linspace(max(u0_val_cal/1e3 + 10, v_read_kv * 0.5), v_read_kv * 1.5, 50)
                fs_theo_cal = [calibrator.calculate_theoretical_fs(v * 1e3) for v in v_range_kv_cal]
                
                fig_cal_plot = go.Figure()
                fig_cal_plot.add_trace(go.Scatter(x=v_range_kv_cal, y=fs_theo_cal, name="Theoretical $f_s$", line=dict(color='#4facfe', width=2)))
                fig_cal_plot.add_trace(go.Scatter(x=[v_actual_v/1e3], y=[fs_meas_hz], name="Measured Point", mode='markers+text',
                                                text=["Actual OP"], textposition="top center", marker=dict(color='red', size=12, symbol='star')))
                fig_cal_plot.update_layout(xaxis_title="Total Voltage (kV)", yaxis_title="Sync Frequency $f_s$ (Hz)", template="plotly_white", height=400)
                st.plotly_chart(fig_cal_plot, width='stretch')
            
        with subtabs_cal[1]:
            st.info("💡 **Pulse Decay Method (DESY Technique)**: Used primarily for Pulsed SRF systems to calibrate probe voltage using forward power and decay time constant.")
            col_pw1, col_pw2 = st.columns(2)
            
            with col_pw1:
                st.markdown("#### Physics Inputs")
                c_roq = st.number_input("Cavity R/Q (Ω)", min_value=1.0, max_value=2000.0, value=1036.0, help="Geometric shunt impedance")
                c_f0 = st.number_input("Resonance Freq (MHz)", value=f0_mhz)
                
                st.markdown("#### Measurements")
                tau_meas = st.number_input("Measured Decay Time τ (μs)", min_value=0.1, max_value=10000.0, value=500.0, help="Time for field to decay to 1/e")
                p_forward = st.number_input("Forward Power P_for (kW)", min_value=0.01, max_value=10000.0, value=100.0)
                adc_val = st.number_input("Probe ADC Reading (mV/Count)", min_value=1.0, value=500.0, key="adc_read")
            
            with col_pw2:
                p_calibrator = DecayPowerCalibrator(f0=c_f0*1e6, r_over_q=c_roq)
                ql_calc = p_calibrator.calculate_ql(tau_meas)
                v_phys_v = p_calibrator.calculate_voltage(tau_meas, p_forward)
                kt_factor = p_calibrator.get_calibration_constant(tau_meas, p_forward, adc_val)
                
                st.markdown("#### Results")
                st.metric("Calculated QL", f"{ql_calc:.2e}")
                st.metric("Physical Voltage V_phys", f"{v_phys_v/1e6:.3f} MV")
                st.metric("Calibration Constant Kt", f"{kt_factor:.2f} V/unit")
                
                st.markdown("---")
                st.latex(r"V_{cav} = 2 \sqrt{ (R/Q) \cdot Q_L \cdot P_{for} }")
                st.caption("Assumes cavity is perfectly tuned and over-coupled.")
            
            st.info("""
            **Physics Principle:** 
            1. **Beam-Based:** $f_s^2 \\propto V_{rf} \\cos(\\phi_s)$. Solving for $V_{rf}$ using machine parameters.
            2. **Power-Based:** Using the energy conservation in the cavity during the decay phase to deduce the stored energy and voltage.
            """)
    
    
    # Data Table for Reference
    st.markdown("---")
    st.subheader("📋 Reference Values (at Current Setpoint)")
    ref_data = {
        "Parameter": [
    "Total Voltage",
    "Optimal Beta",
    "Optimal Harmonic Voltage",
    "Reflection |ρ|",
    "Synchronous Phase",
    "Harmonic Phase (opt)"
        ],
        "Value": [
    fmt(res['ut0']) + " kV",
    fmt(res['beta_opt']),
    fmt(res['vh_opt']) + " kV",
    fmt(res['rho'], 4),
    fmt(res['phi_s']) + "°",
    fmt(res['phi_h_opt']) + "°"
        ]
    }
    st.table(ref_data)
    
    # Save configuration button
    st.markdown("---")
    if st.button("💾 Save Current Configuration", type="primary"):
        config_data = {
    'preset_name': selected_preset,
    'energy': energy,
    'harmonic_number': h_rf,
    'u0_kev': u0_kev,
    'main_voltage_kv': vfcav_kv,
    'main_frequency': f0_mhz,
    'main_ncav': ncav,
    'q0_fund': q0_fund,
    'beta_fund': beta_fund,
    'rshunt_mohm': rshunt_mohm,
    'rf_gain': rf_gain,
    'harmonic_ncav': nhcav,
    'harmonic_voltage_mode': vh_mode,
    'qh0': qh0,
    'rhshunt_mohm': rhshunt_mohm,
    'beam_current_ma': current_input,
    'beta_fixed': beta_fixed
        }
        
        # Update session state for cross-page sync
        for key, value in config_data.items():
            st.session_state[key] = value
        
        st.success("✅ Configuration saved to session! All pages will use these parameters.")
    
    # Information box
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    ### RF System Analytical Dashboard
    
    This professional dashboard integrates comprehensive RF cavity analysis with:
    
    - **Physics Engine**: Full calculation chain from SMath formulas
    - **Main Cavity Analysis**: Power, detuning, and coupling optimization
    - **Harmonic Cavity Analysis**: 4th harmonic system with optimal voltage calculation
    - **Theory & Validation**: Venturini (2018) model for operating regime guidance
    - **RF Feedback Control**: Bandwidth analysis and maximum gain calculation
    - **Energy Loss Tracking**: U₀, Uh₀, and UT₀ components
    - **Potential Well Visualization**: Longitudinal bunch stretching analysis
    
    ### Key Features
    
    - **Global Configuration Sync**: All parameters synchronized across pages
    - **Preset Management**: SOLEIL II Phase 1/2, Aladdin, MAX IV, and custom configurations
    - **Real-time Calculations**: Live updates as you adjust parameters
    - **Professional Visualizations**: Interactive Plotly charts
    - **Theory-guided Operation**: Operating regime classification and formula validation
    
    ### References & Technical Sources
    
    1. **Passive Harmonic Cavity Theory**:
       - *Venturini, M. (2018)*. "Passive higher-harmonic rf cavities with general settings and multibunch instabilities in electron storage rings". Phys. Rev. Accel. Beams **21**, 114404.

    2. **Feedback Stability (Pedersen Model)**:
       - *Shen, Y. B. et al. (2024)*. "Stability analysis of double-harmonic cavity system in heavy beam loading with its feedback loops by a mathematical method based on Pedersen model". J. Phys.: Conf. Ser. **2687**, 072026.

    3. **Higher-Harmonic Cavity Instabilities (Bosch Model)**:
       - *Bosch, R. A. & Hsue, C. S. (1993)*. "Suppression of longitudinal coupled-bunch instabilities by a passive higher harmonic cavity". Part. Accel. **42**, 81–99.
       - *Bosch, R. A. (1997)*. "Instability analysis of an active higher-harmonic cavity". Proc. PAC97, pp. 862–864.

    4. **Double RF Analytic Model**: 
       - *Hofmann, A. & Myers, S. (1980)*. "Beam dynamics in a double RF system". Proc. 11th Int. Conf. on High-Energy Accelerators.
       - *Hofmann, A. (2004)*. "The Physics of Synchrotron Radiation". Cambridge University Press.
    
    5. **Instability & Landau Damping**:
       - *Sacherer, F. J. (1973)*. "A longitudinal stability criterion for bunched beams". IEEE Trans. Nucl. Sci.
       - *Krinsky, S. & Wang, J. M. (1985)*. "Longitudinal instabilities of bunched beams subject to a non-harmonic RF potential". Particle Accelerators.
    
    6. **RF Cavity Operation**:
       - *ESLS RF Workshop (October 2025)*. SOLEIL II Design Parameters & RF Feedback Strategies.
       - Physics formulas derived from project-specific SMath calculation sheets for cavity matching and detuning.
    
    7. **Implementation**:
       - Based on `cavity_operation/rf_system_pro.py`, `utils/hofmann_model.py`, `utils/venturini_model.py`, `utils/pedersen_model.py`, and `utils/bosch_model.py`.
    """)
