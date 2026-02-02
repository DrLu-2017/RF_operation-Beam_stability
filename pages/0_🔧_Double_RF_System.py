"""
Double RF System Analytical Dashboard
Professional RF cavity analysis with comprehensive physics calculations.
Integrated from cavity_operation/rf_system_pro.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

st.markdown(f"**Energy:** {energy} GeV | **Frequency:** {f0_mhz} MHz | **Harmonic:** n={nh_harm}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sync Phase φ_s", f"{res['phi_s']:.2f} °")
col2.metric("Optimal β", f"{res['beta_opt']:.2f}")
col3.metric("Beam Power/Cav", f"{res['p_beam']:.2f} kW")
col4.metric("Reflection |ρ|", f"{res['rho']:.3f}")

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

# --- THE COMPREHENSIVE ANALYSIS TABS ---
tabs = st.tabs([
    "Plot 1: Main Cavity Power",
    "Plot 2: Voltage Components",
    "Harmonic Cavity Analysis",
    "Detuning & Phase Analysis",
    "Plot 3: Harmonic Power",
    "Energy Loss Analysis",
    "RF Feedback Control",
    "Plot 4: Potential Well",
    "Plot 5: Reflection Match",
    "Synchronous Phase Analysis"
])

with tabs[0]:  # Plot 1: Main Cavity Power vs Current
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=i_range, y=[s['p_inc'] for s in scans], name="Incident (Pi)", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=i_range, y=[s['p_beam'] for s in scans], name="Beam (Pb)", line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=i_range, y=[s['p_ref'] for s in scans], name="Reflected (Pr)", line=dict(color='red')))
    fig1.update_layout(
        title="Power per Main Cavity (kW)",
        xaxis_title="Current (mA)",
        yaxis_title="Power (kW)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)

with tabs[1]:  # Plot 2: Voltage Components vs Phase
    phi_deg = np.linspace(-180, 180, 1000)
    v_m = vfcav_kv * np.sin(np.radians(phi_deg + res['phi_s']))
    v_h = res['vh_opt'] * np.sin(np.radians(nh_harm * phi_deg))
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=phi_deg, y=v_m, name=f"Main ({vfcav_kv:.0f} kV)", line=dict(dash='dash', color='blue')))
    fig2.add_trace(go.Scatter(x=phi_deg, y=v_h, name="Harmonic (Opt)", line=dict(color='orange')))
    fig2.add_trace(go.Scatter(x=phi_deg, y=v_m+v_h, name="Total Voltage", line=dict(width=3, color='black')))
    fig2.update_layout(
        title="RF Voltage Distribution",
        xaxis_title="Phase (deg)",
        yaxis_title="Voltage (kV)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:  # Harmonic Cavity Analysis
    st.subheader("💠 Harmonic Cavity Analysis (n=4)")
    
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Optimal Voltage", f"{res['vh_opt']:.2f} kV", help="Optimal harmonic voltage")
    col_h2.metric("Voltage per Cavity", f"{res['vh_cav']:.2f} kV", help="Voltage per harmonic cavity")
    col_h3.metric("Harmonic Phase φ_h,opt", f"{res['phi_h_opt']:.2f}°", help="Optimal harmonic phase")
    
    col_h4, col_h5, col_h6 = st.columns(3)
    col_h4.metric("Detuning δfh", f"{h_params['detuning_khz']:.2f} kHz", help="Required detuning at Imax")
    col_h5.metric("Power/Cavity (Imax)", f"{[s['ph_diss'] for s in scans][-1]:.2f} kW")
    col_h6.metric("Total Loss (H)", f"{nhcav * [s['ph_diss'] for s in scans][-1]:.2f} kW")
    
    st.markdown("---")
    st.write("**Calculation Formulas:**")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.latex(r"\phi_s = \pi - \arcsin\left[\frac{n^2}{(n^2-1)} \cdot \frac{U_0}{V_c}\right]")
        st.latex(r"V_{h,\mathrm{opt}} = \sqrt{\frac{V_c^2}{n^2} - \frac{U_0^2}{(n^2-1)}}")
    with col_f2:
        st.latex(r"\phi_{h,\mathrm{opt}} = \frac{1}{n} \arcsin\left[\frac{-U_0}{V_{h,\mathrm{opt}} (n^2-1)}\right]")
        st.latex(r"\delta f_h = -\frac{f_{h0}}{Q_{h0}} \sqrt{\left(\frac{R_{h} \cdot I_{max}}{v_h \cdot n_{h,cav}}\right)^2 - \frac{1}{4}}")

with tabs[3]:  # Detuning & Phase Analysis
    st.subheader("📊 Detuning & Phase Analysis")
    
    col_det1, col_det2, col_det3 = st.columns(3)
    col_det1.metric("Fundamental δf (kHz)", f"{f_params['detuning_khz']:.3f}", 
                   help="Detuning at I_max for fundamental cavities")
    col_det2.metric("Harmonic δfh (kHz)", f"{h_params['detuning_khz']:.3f}",
                   help="Detuning at I_max for harmonic cavities")
    col_det3.metric("Harmonic Phase (°)", f"{h_params['phi_hs']:.2f}",
                   help="Phase of harmonic voltage")
    
    st.markdown("---")
    fig_det = go.Figure()
    fig_det.add_vline(x=f_params['detuning_khz'], line_dash="dash", line_color="blue",
                     annotation_text=f"Fund.: {f_params['detuning_khz']:.2f} kHz", 
                     annotation_position="bottom right")
    fig_det.add_vline(x=h_params['detuning_khz'], line_dash="dot", line_color="red",
                     annotation_text=f"Harm.: {h_params['detuning_khz']:.2f} kHz", 
                     annotation_position="top right")
    fig_det.update_layout(
        title="Operating Detuning Points",
        xaxis_title="Detuning (kHz)",
        yaxis_title="Status",
        template="plotly_white",
        height=300
    )
    st.plotly_chart(fig_det, use_container_width=True)

with tabs[4]:  # Plot 3: Harmonic Cavity Power vs Current
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=i_range, y=[s['ph_diss'] for s in scans], 
                             name="Ph_diss (per cavity)", line=dict(color='purple')))
    fig3.update_layout(
        title="Harmonic Cavity Wall Loss",
        xaxis_title="Current (mA)",
        yaxis_title="Power (kW)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tabs[5]:  # Energy Loss Analysis
    st.subheader("⚡ Energy Loss Analysis")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("U₀ (Base Loss)", f"{u0_kev:.2f} keV", help="Base radiation loss per turn")
    col_e2.metric("Uh₀ (Harmonic)", f"{res['uh0']:.2f} keV", help="Energy loss from harmonic cavities")
    col_e3.metric("UT₀ (Total)", f"{res['ut0']:.2f} keV", help="Total energy loss per turn")
    
    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=i_range, y=[u0_kev]*len(i_range), name="U₀ (Base)", line=dict(dash='dot')))
    fig_en.add_trace(go.Scatter(x=i_range, y=[s['uh0'] for s in scans], name="Uh₀ (Harmonic)"))
    fig_en.add_trace(go.Scatter(x=i_range, y=[s['ut0'] for s in scans], name="UT₀ (Total)", line=dict(width=3)))
    fig_en.update_layout(
        title="Energy Loss vs Beam Current",
        xaxis_title="Current (mA)",
        yaxis_title="Energy Loss (keV)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig_en, use_container_width=True)

with tabs[6]:  # RF Feedback Control
    st.subheader("📡 RF Feedback Control - Cavity Bandwidth Analysis")
    
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    col_rf1.metric("Loaded Q (QL)", f"{rf_params['ql']:.0f}")
    col_rf2.metric("Bandwidth without FB", f"{rf_params['bp_no_fb']:.2f} kHz", help="Natural cavity bandwidth: BP = f₀/QL")
    col_rf3.metric("Bandwidth with FB", f"{rf_params['bp_with_fb']:.2f} kHz", help="With RF Feedback: BP_FB = BP × (1 + Gain)")
    
    st.markdown("---")
    st.write("### Maximum Gain from Feedback Delay")
    
    rf_periods = st.slider("RF Periods in Feedback Delay (N)", 1, 1000, 704, 
                          help="Number of RF periods for 0° phase (Δt = N/f₀)")
    
    # Recalculate with user-specified RF periods
    rf_params_custom = calculate_rf_feedback_params(f0_mhz, q0_fund, beta_fund, rf_gain, rf_periods)
    
    col_gm1, col_gm2, col_gm3, col_gm4 = st.columns(4)
    col_gm1.metric("Feedback Delay (μs)", f"{rf_params_custom['delta_t_us']:.3f}", help=f"{rf_periods} RF periods")
    col_gm2.metric("Max Gain (G_max)", f"{rf_params_custom['g_max']:.3f}", help="Maximum achievable RF feedback gain")
    col_gm3.metric("Max BW (kHz)", f"{rf_params_custom['bp_max']:.2f}", help="Maximum cavity bandwidth with max gain")
    col_gm4.metric("BW Improvement", f"{rf_params_custom['bw_improvement']:.2f}×", help="Bandwidth increase factor")
    
    st.markdown("---")
    # Bandwidth improvement visualization
    gain_range = np.linspace(0, max(3, rf_params_custom['g_max'] * 1.2), 50)
    bp_improved = [rf_params['bp_no_fb'] * (1 + g) for g in gain_range]
    
    fig_bw = go.Figure()
    fig_bw.add_trace(go.Scatter(x=gain_range, y=bp_improved, 
                                name="Cavity Bandwidth", mode='lines', 
                                line=dict(color='blue', width=2)))
    fig_bw.add_vline(x=rf_gain, line_dash="dash", line_color="red", 
                     annotation_text=f"Current: {rf_gain:.3f}", annotation_position="top right")
    fig_bw.add_vline(x=rf_params_custom['g_max'], line_dash="dot", line_color="green",
                     annotation_text=f"Max: {rf_params_custom['g_max']:.3f}", annotation_position="top left")
    fig_bw.update_layout(
        title="RF Feedback Bandwidth vs Gain (with Maximum Limit)",
        xaxis_title="RF Feedback Gain",
        yaxis_title="Cavity Bandwidth (kHz)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig_bw, use_container_width=True)

with tabs[7]:  # Plot 4: Potential Well Representation
    phi_pot, potential = calculate_potential_well(vfcav_kv, res['vh_opt'], res['phi_s'], nh_harm, res['ut0'])
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=phi_pot, y=potential, name="Potential", fill='tozeroy', line=dict(color='darkblue')))
    fig4.update_layout(
        title="Longitudinal Potential Well (Bunch Stretching)",
        xaxis_title="Phase (deg)",
        yaxis_title="U (arb. units)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig4, use_container_width=True)

with tabs[8]:  # Plot 5: Reflection / Match Performance
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=i_range, y=[s['rho'] for s in scans], name="Reflection |ρ|", line=dict(color='red')))
    fig5.update_layout(
        title="Coupler Match over Current Range",
        xaxis_title="Current (mA)",
        yaxis_title="|ρ|",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig5, use_container_width=True)

with tabs[9]:  # Synchronous Phase Analysis
    st.subheader("🔄 Synchronous Phase Analysis")
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Synchronous Phase Φs", f"{res['phi_s']:.2f}°")
    col_p2.metric("Optimal β", f"{res['beta_opt']:.3f}")
    
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=i_range, y=[s['phi_s'] for s in scans], name="Synchronous Phase", 
                                line=dict(color='orange', width=2)))
    fig_ph.update_layout(
        title="Synchronous Phase vs Beam Current",
        xaxis_title="Current (mA)",
        yaxis_title="Phase (degrees)",
        template="plotly_white",
        hovermode='x unified'
    )
    st.plotly_chart(fig_ph, use_container_width=True)

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
        f"{res['ut0']:.2f} kV",
        f"{res['beta_opt']:.2f}",
        f"{res['vh_opt']:.2f} kV",
        f"{res['rho']:.4f}",
        f"{res['phi_s']:.2f}°",
        f"{res['phi_h_opt']:.2f}°"
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
    
    ### References
    
    - Based on `cavity_operation/rf_system_pro.py`
    - SOLEIL II parameters from ESLS RF Workshop (October 2025)
    - Physics formulas from SMath cavity calculation sheets
    """)
