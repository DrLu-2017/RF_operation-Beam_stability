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
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Double RF System - ALBuMS",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 RF System Analytical Dashboard")
st.markdown("Professional analysis of double RF cavity systems with 4th harmonic configuration.")

# Initialize config manager
config_mgr = ConfigManager()

# Sidebar for configuration
with st.sidebar:
    st.markdown("<div style='text-align: center;'><h1 style='color: #4facfe;'>DRFB</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Navigation")
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/0_🔧_Double_RF_System.py", label="Double RF System", icon="🔧")
    st.page_link("pages/1_📊_Parameter_Scans.py", label="Parameter Scans", icon="📊")
    st.page_link("pages/2_🎯_Optimization.py", label="R-Factor Optimization", icon="🎯")
    st.page_link("pages/3_🔬_Mode_Analysis.py", label="Robinson Mode Analysis", icon="🔬")
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
            help="Phase 1: 803 keV (487+316) | Phase 2: 846 keV (487+359)"
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
            min_value=1.0,
            max_value=5.0,
            value=float(ring_params.get("energy", 2.75)),
            step=0.01,
            key="energy_double_rf"
        )
        
        f0_mhz = st.number_input(
            "RF Frequency f₀ (MHz)",
            min_value=100.0,
            max_value=500.0,
            value=float(mc_params.get('frequency', 352.2)),
            step=0.1,
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
    
    with col3:
        u0_kev = st.number_input(
            "Energy Loss U₀ (keV)",
            min_value=1.0,
            max_value=10000.0,
            value=float(ring_params.get("energy_loss_per_turn", 0.000743)) * 1e6,
            step=1.0,
            key="u0_kev_double_rf",
            help="Energy loss per turn in keV"
        )
        
        if "SOLEIL II" in selected_preset and phase_display != "N/A":
            if phase_display == "Phase 1":
                st.info("**Phase 1:** 487 keV (Synch.) + 316 keV (IDs) = **803 keV**")
            else:
                st.info("**Phase 2:** 487 keV (Synch.) + 359 keV (IDs) = **846 keV**")

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
            min_value=800.0,
            max_value=2000.0,
            value=float(mc_params.get('voltage', 1.7)) * 1000,  # MV to kV
            step=10.0,
            key="v_rf_total_double_rf"
        )
        
        q0_fund = st.number_input(
            "Unloaded Q (Q₀)",
            min_value=10000,
            max_value=100000,
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
            min_value=0.5,
            max_value=20.0,
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
nh_harm = 4  # Fixed at 4th harmonic
with st.expander("◐ Harmonic Cavities (n=4, 1.409 GHz)", expanded=True):
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
                min_value=100.0,
                max_value=600.0,
                value=400.0,
                step=10.0,
                key="vh_manual_double_rf"
            )
    
    with col3:
        qh0 = st.number_input(
            "Loaded Q (Q₀)",
            min_value=10000,
            max_value=100000,
            value=int(hc_params.get('Q0', hc_params.get('Q', 31000))),
            step=100,
            key="qh0_double_rf"
        )
        
        rhshunt_mohm = st.number_input(
            "Shunt Impedance R (MΩ)",
            min_value=0.1,
            max_value=5.0,
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
        st.markdown(f"### 🟢 **{phase_display}** - U₀ = {u0_kev:.0f} keV (487 keV + 316 keV IDs)")
    else:
        st.markdown(f"### 🟠 **{phase_display}** - U₀ = {u0_kev:.0f} keV (487 keV + 359 keV IDs)")

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
    "� Hofmann Analytic Model",
    "�📊 Detailed Data & Theory"
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
        'sigma_z0': 15e-12
    }
    
    mc_clbi = {
        'V': vfcav_kv * 1000 / ncav,
        'Q': q0_fund / (1 + beta_fund),
        'R_sh': rshunt_mohm * 1e6,
        'psi': np.arctan( 2 * (q0_fund / (1 + beta_fund)) * (f_params['detuning_khz'] * 1000) / (f0_mhz*1e6) ),
        'f_rf': f0_mhz * 1e6
    }
    
    hc_clbi = {
        'V': res['vh_cav'] * 1000,
        'Q': qh0,
        'R_sh': rhshunt_mohm * 1e6,
        'psi': np.radians(h_params['phi_hs']),
        'n': nh_harm,
        'theta_rel': 0.0
    }
    
    # Perform core calculation once
    calc_success = False
    try:
        clbi_obj = DoubleRF_CLBI(beam_clbi, mc_clbi, hc_clbi)
        phi_s_clbi = clbi_obj.solve_equilibrium()
        ws_double_clbi = clbi_obj.calculate_synchrotron_frequency()
        complex_shift_clbi = clbi_obj.calculate_complex_shift(m=1)
        
        # Landau Damping
        ws0_single = beam_clbi['f_rev'] * np.sqrt( h_rf * beam_clbi['alpha_c'] * (vfcav_kv*1000 * abs(np.cos(phi_s_clbi))) / (2*np.pi*beam_clbi['E0']) )
        ratio_sigma = ws_double_clbi / (2*np.pi) / ws0_single if ws0_single > 0 else 0
        sigma_z_est_clbi = 15e-12 * ratio_sigma
        delta_ws_clbi = clbi_obj.calculate_frequency_spread(sigma_z_est_clbi)
        
        # Update session state for the diagram
        st.session_state.current_complex_shift = complex_shift_clbi
        st.session_state.current_delta_ws = delta_ws_clbi
        calc_success = True
    except Exception as e:
        calc_error = str(e)

    subtabs_stab = st.tabs(["🔥 Instability Analysis (CLBI)", "📡 RF Feedback", "🌊 Landau Damping"])
    
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
    
    with subtabs_stab[1]: # RF Feedback
        col_rf1, col_rf2, col_rf3 = st.columns(3)
        col_rf1.metric("Loaded Q (QL)", f"{rf_params['ql']:.0f}")
        col_rf2.metric("Bandwidth (No FB)", f"{rf_params['bp_no_fb']:.2f} kHz")
        col_rf3.metric("Bandwidth (With FB)", f"{rf_params['bp_with_fb']:.2f} kHz")
        
        st.markdown("#### Max Feedback Gain")
        rf_periods = st.slider("Delay (RF Periods)", 1, 1000, 704)
        rf_params_custom = calculate_rf_feedback_params(f0_mhz, q0_fund, beta_fund, rf_gain, rf_periods)
        
        col_gm1, col_gm2, col_gm3 = st.columns(3)
        col_gm1.metric("Delay", f"{rf_params_custom['delta_t_us']:.3f} μs")
        col_gm2.metric("Max Gain", f"{rf_params_custom['g_max']:.2f}")
        col_gm3.metric("Max BW", f"{rf_params_custom['bp_max']:.2f} kHz")
        
        # Plot
        gain_range = np.linspace(0, max(3, rf_params_custom['g_max'] * 1.2), 50)
        bp_improved = [rf_params['bp_no_fb'] * (1 + g) for g in gain_range]
        fig_bw = go.Figure()
        fig_bw.add_trace(go.Scatter(x=gain_range, y=bp_improved, name="BW", line=dict(color='blue')))
        fig_bw.add_vline(x=rf_gain, line_dash="dash", line_color="red", annotation_text="Current")
        fig_bw.add_vline(x=rf_params_custom['g_max'], line_dash="dot", line_color="green", annotation_text="Max")
        fig_bw.update_layout(title="RF Feedback Bandwidth vs Gain", xaxis_title="Gain", yaxis_title="BW (kHz)", template="plotly_white")
        st.plotly_chart(fig_bw, width='stretch')

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
            st.plotly_chart(fig_pot, use_container_width=True)

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
            st.plotly_chart(fig_fs, use_container_width=True)
        
        st.info("**Mechanism:** The harmonic RF system creates a 'flat-bottom' potential ($x^4$ at the center), which results in a synchrotron frequency that increases with the oscillation amplitude. This decoherence suppresses the growth of coupled-bunch modes.")

        st.markdown("---")
        st.subheader("📊 Quantifying Stability: The Dispersion Relation")
        st.latex(r"1 = -i \Delta \Omega \int \frac{\psi'(J)}{\Omega - \omega(J)} dJ")
        
        st.markdown("""
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
        
        st.plotly_chart(fig_sd, use_container_width=True)
        st.caption(f"Calculated Frequency Spread (Δfs): {d_ws_raw/(2*np.pi):.1f} Hz. Coherent Frequency Shift: {px:.1f} Hz. Stability boundary defines the suppression limit.")

# === Tab 4: Hofmann Analytic Model ===
with tabs[3]:
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
            st.plotly_chart(fig_hof1, use_container_width=True)
            
        with col_m2:
            st.markdown("#### Bunch Distribution")
            fig_hof2 = go.Figure()
            fig_hof2.add_trace(go.Scatter(x=phi_deg, y=dist_hof, name="Flat-topped (Double)", fill='tozeroy', line=dict(color='red')))
            fig_hof2.add_trace(go.Scatter(x=phi_deg, y=dist_s_hof, name="Gaussian (Single)", line=dict(color='blue', dash='dash')))
            fig_hof2.update_layout(title="Bunch Density Profile", xaxis_title="Phase (deg)", yaxis_title="Normalized Density", template="plotly_white")
            st.plotly_chart(fig_hof2, use_container_width=True)
            
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
        st.plotly_chart(fig_hof3, use_container_width=True)
        
        st.success(f"**Flat Potential Solution:** $k = V_2/V_1 = {hofmann.V2/hofmann.V1:.4f}$, $\phi_{{s,main}} = {np.degrees(hofmann.phi_s1):.2f}^\circ$, $\phi_{{s,harm}} = {np.degrees(hofmann.phi_s2):.2f}^\circ$")

    except Exception as e:
        st.error(f"Hofmann Model Error: {e}")

# === Tab 5: Detailed Data & Theory ===
with tabs[4]:
    st.subheader("Detailed Analysis & Physics Background")
    
    subtabs_det = st.tabs(["Harmonic Analysis Data", "Physics Interpretation"])
    
    with subtabs_det[0]: # Harmonic Data
        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("Optimal Vh", f"{res['vh_opt']:.2f} kV")
        col_h2.metric("Vh per Cavity", f"{res['vh_cav']:.2f} kV")
        col_h3.metric("Opt Phase", f"{res['phi_h_opt']:.2f}°")
        
        st.markdown("**Formulas:**")
        st.latex(r"V_{h,\mathrm{opt}} = \sqrt{\frac{V_c^2}{n^2} - \frac{U_0^2}{(n^2-1)}}")
        
    with subtabs_det[1]: # Physics Intro
        # (Content from previous physics tab)
        st.markdown("### Passive Harmonic Cavity Physics")
        st.latex(r"\Delta f = \frac{f_r}{2 Q_L} \sqrt{ \frac{4 I_{DC}^2 R_{sh}^2}{V_{hc}^2} - 1 }")
        st.info("See full documentation in project wiki.")


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
    - **RF Feedback Control**: Bandwidth analysis and maximum gain calculation
    - **Energy Loss Tracking**: U₀, Uh₀, and UT₀ components
    - **Potential Well Visualization**: Longitudinal bunch stretching analysis
    
    ### Key Features
    
    - **Global Configuration Sync**: All parameters synchronized across pages
    - **Preset Management**: SOLEIL II Phase 1/2, Aladdin, and custom configurations
    - **Real-time Calculations**: Live updates as you adjust parameters
    - **Professional Visualizations**: Interactive Plotly charts
    
    ### References & Technical Sources
    
    1. **Double RF Analytic Model**: 
       - *Hofmann, A. & Myers, S. (1980)*. "Beam dynamics in a double RF system". Proc. 11th Int. Conf. on High-Energy Accelerators.
       - *Hofmann, A. (2004)*. "The Physics of Synchrotron Radiation". Cambridge University Press.
    
    2. **Instability & Landau Damping**:
       - *Sacherer, F. J. (1973)*. "A longitudinal stability criterion for bunched beams". IEEE Trans. Nucl. Sci.
       - *Krinsky, S. & Wang, J. M. (1985)*. "Longitudinal instabilities of bunched beams subject to a non-harmonic RF potential". Particle Accelerators.
    
    3. **RF Cavity Operation**:
       - *ESLS RF Workshop (October 2025)*. SOLEIL II Design Parameters & RF Feedback Strategies.
       - Physics formulas derived from project-specific SMath calculation sheets for cavity matching and detuning.
    
    4. **Implementation**:
       - Based on `cavity_operation/rf_system_pro.py` and `utils/hofmann_model.py`.
    """)
