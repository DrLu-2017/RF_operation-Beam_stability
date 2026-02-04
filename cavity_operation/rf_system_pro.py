import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="2.75 GeV RF Analysis Pro", layout="wide")

# --- Machine Constants (From SMath Document) [1, 2] ---
# Note: These are now replaced with sidebar input values below
# E_GEV = 2.75
# F0_MHZ = 352.2
# U0_BASE_KV = 803.0   # Base radiation loss per turn
# N_CAV = 4            # Number of main cavities
# NH_CAV = 2           # Number of harmonic cavities
# NH = 4               # 4th Harmonic system
# R_SHUNT_MQ = 5.0     # 5 MOhm Main shunt impedance
# RH_SHUNT_MQ = 0.92   # 0.92 MOhm Harmonic shunt impedance

def run_physics_engine(curr_ma, v_total_kv, beta_fixed, ncav, nhcav, nh, r_shunt_mohm, rh_shunt_mohm, u0_base_kv, vh_manual_kv=None):
    """
    Implements the full calculation chain for double RF system analysis.
    
    Physical formulas based on:
    - Bosch et al., PRSTAB 4.7 (2001): 074401 [Robinson instabilities]
    - Byrd & Georgsson, PRSTAB 4.3 (2001): 030701 [Harmonic cavities]
    - Wilson, SLAC-PUB-2884 (1981) [RF cavity fundamentals]
    
    Parameters:
    -----------
    curr_ma : float
        Beam current in mA
    v_total_kv : float
        Total main cavity voltage in kV
    beta_fixed : float
        Fixed coupling factor β
    ncav : int
        Number of main cavities
    nhcav : int
        Number of harmonic cavities
    nh : int
        Harmonic number (typically 3 or 4)
    r_shunt_mohm : float
        Main cavity shunt impedance in MΩ
    rh_shunt_mohm : float
        Harmonic cavity shunt impedance in MΩ
    u0_base_kv : float
        Base energy loss per turn in keV
    vh_manual_kv : float, optional
        Manual harmonic voltage. If None, uses optimized formula.
    
    Returns:
    --------
    dict : Dictionary containing all calculated RF parameters
    """
    import warnings
    
    i_beam_a = curr_ma / 1000.0
    v_cav_kv = v_total_kv / ncav
    
    # ========== 1. Harmonic Voltage Calculation ==========
    if vh_manual_kv is None:
        # Standard formula for optimal harmonic voltage (flat potential)
        # V_h,opt = sqrt(V_c^2/n^2 - U_0^2/(n^2-1))
        # Reference: Byrd & Georgsson, PRSTAB 4.3 (2001)
        term1 = (v_total_kv / nh) ** 2
        term2 = (u0_base_kv ** 2) / (nh**2 - 1)
        
        if term1 < term2:
            warnings.warn(
                f"Harmonic voltage calculation: V_c^2/n^2 ({term1:.2f}) < U_0^2/(n^2-1) ({term2:.2f}). "
                f"This indicates insufficient main cavity voltage for optimal harmonic operation. "
                f"Setting V_h,opt to minimum viable value.",
                RuntimeWarning
            )
            vh_opt_kv = 0.01  # Minimum non-zero value
        else:
            vh_opt_kv = np.sqrt(term1 - term2)
    else:
        # Use manually specified voltage
        vh_opt_kv = vh_manual_kv
    
    vh_cav_kv = vh_opt_kv / nhcav if nhcav > 0 else 0
    
    # ========== 2. Harmonic Power Dissipation & Induced Loss ==========
    # Ph_diss = V^2 / (2 * R_shunt)
    # Reference: Wilson, SLAC-PUB-2884 (1981)
    ph_diss_cav_kw = (vh_cav_kv**2) / (2 * rh_shunt_mohm * 1e3)
    p_h_total_diss_kw = nhcav * ph_diss_cav_kw
    
    # Energy loss increase due to harmonic cavities
    uh0_kv = p_h_total_diss_kw / (i_beam_a + 1e-9) if curr_ma > 0 else 0
    ut0_kv = u0_base_kv + uh0_kv
    
    # ========== 3. Synchronous Phase Calculation ==========
    # Two formulas are available:
    # (A) Standard formula: φ_s = arcsin(U_T0 / V_c)
    # (B) Harmonic cavity formula: φ_s = π - arcsin[n²/(n²-1) * U_0 / V_c]
    #
    # Formula (B) is derived for optimal flat-potential condition with harmonic cavity.
    # Reference: Bosch et al., PRSTAB 4.7 (2001), Eq. (3)
    
    # Use formula (B) for harmonic cavity case
    ratio_phi_s = (nh**2 / (nh**2 - 1)) * (u0_base_kv / v_total_kv)
    
    if abs(ratio_phi_s) <= 1:
        # Valid range for arcsin
        phi_s_rad = np.pi - np.arcsin(ratio_phi_s)
    else:
        # Fallback to standard formula when ratio exceeds 1
        warnings.warn(
            f"Synchronous phase calculation: ratio n²/(n²-1) * U_0/V_c = {ratio_phi_s:.3f} exceeds 1. "
            f"Using standard formula φ_s = arcsin(U_T0/V_c) instead. "
            f"Consider increasing main cavity voltage.",
            RuntimeWarning
        )
        ratio_standard = ut0_kv / v_total_kv
        phi_s_rad = np.arcsin(np.clip(ratio_standard, -1, 1))
    
    phi_s_deg = np.degrees(phi_s_rad)
    
    # ========== 4. Harmonic Cavity Optimal Phase ==========
    # φ_h,opt = (1/n) * arcsin[-U_0 / (V_h,opt * (n²-1))]
    # Reference: Bosch et al., PRSTAB 4.7 (2001), Eq. (4)
    if vh_opt_kv > 1e-6:  # Avoid division by very small numbers
        ratio_phi_h = -u0_base_kv / (vh_opt_kv * (nh**2 - 1))
        if abs(ratio_phi_h) <= 1:
            phi_h_opt_rad = (1/nh) * np.arcsin(ratio_phi_h)
        else:
            warnings.warn(
                f"Harmonic phase calculation: ratio -U_0/(V_h*(n²-1)) = {ratio_phi_h:.3f} exceeds 1. "
                f"Setting φ_h,opt to 0. Consider adjusting harmonic voltage.",
                RuntimeWarning
            )
            phi_h_opt_rad = 0
    else:
        phi_h_opt_rad = 0
    
    phi_h_opt_deg = np.degrees(phi_h_opt_rad)
    
    # ========== 5. Main Cavity Power & Coupling (Beta) ==========
    # Reference: Wilson, SLAC-PUB-2884 (1981)
    pf_diss_cav_kw = (v_cav_kv**2) / (2 * r_shunt_mohm * 1e3)
    p_beam_per_cav_kw = (i_beam_a * ut0_kv) / ncav if ncav > 0 else 0
    
    # Optimal Beta: β_opt = 1 + (2 * P_beam / P_diss_wall)
    # This ensures critical coupling for maximum power transfer
    if pf_diss_cav_kw > 1e-9:
        beta_opt = 1 + (2 * p_beam_per_cav_kw) / pf_diss_cav_kw
    else:
        beta_opt = 1.0
        warnings.warn(
            "Main cavity dissipation power is near zero. Setting β_opt = 1.0",
            RuntimeWarning
        )
    
    # Reflection Coefficient (ρ) based on mismatch with fixed Beta
    # ρ = |(β_fixed - β_opt) / (β_fixed + β_opt)|
    denominator = beta_fixed + beta_opt
    if abs(denominator) > 1e-9:
        rho = np.abs((beta_fixed - beta_opt) / denominator)
    else:
        rho = 0.0
        warnings.warn(
            "Beta sum is near zero. Setting ρ = 0.0",
            RuntimeWarning
        )
    
    # Power calculations with reflection
    rho_sq = rho**2
    if rho_sq < 0.999:  # Avoid division by values very close to 1
        p_inc_kw = (pf_diss_cav_kw + p_beam_per_cav_kw) / (1 - rho_sq)
        p_ref_kw = p_inc_kw - p_beam_per_cav_kw - pf_diss_cav_kw
    else:
        warnings.warn(
            f"Reflection coefficient ρ = {rho:.3f} is very high (near total reflection). "
            f"Power calculations may be unreliable. Check coupling factor β.",
            RuntimeWarning
        )
        p_inc_kw = pf_diss_cav_kw + p_beam_per_cav_kw
        p_ref_kw = 0.0
    
    return {
        "phi_s": phi_s_deg, 
        "vh_opt": vh_opt_kv, 
        "beta_opt": beta_opt,
        "p_beam": p_beam_per_cav_kw, 
        "p_inc": p_inc_kw, 
        "p_ref": p_ref_kw,
        "ph_diss": ph_diss_cav_kw, 
        "ut0": ut0_kv, 
        "rho": rho,
        "pf_diss": pf_diss_cav_kw, 
        "vh_cav": vh_cav_kv, 
        "uh0": uh0_kv,
        "phi_h_opt": phi_h_opt_deg
    }


def calculate_harmonic_cavity_params(i_max_ma, vh_opt_kv, nhcav, nh, f0_mhz, rh_shunt_mohm, qh0):
    """
    Calculate harmonic cavity detuning and voltage parameters for passive operation.
    
    Physical basis:
    - Passive harmonic cavities are detuned to Robinson stability point
    - Detuning ensures proper phase relationship for bunch lengthening
    
    Formula: δf_h = -(f_h0 / Q_h0) * sqrt(((R_h_shunt * I_max) / (v_h * n_h_cav))^2 - 1/4)
    
    Parameters:
    -----------
    i_max_ma : float
        Maximum beam current in mA
    vh_opt_kv : float
        Total optimal harmonic voltage in kV
    nhcav : int
        Number of harmonic cavities
    nh : int
        Harmonic number (e.g., 3 or 4)
    f0_mhz : float
        Fundamental RF frequency in MHz
    rh_shunt_mohm : float
        Harmonic cavity shunt impedance in MΩ
    qh0 : float
        Harmonic cavity loaded Q
    
    Returns:
    --------
    dict : Dictionary with detuning_khz, phi_hs, vh_cav
    
    Notes:
    ------
    - The condition ratio^2 >= 0.25 must be satisfied for passive operation
    - If not satisfied, cavity operates in evanescent mode (not recommended)
    """
    import warnings
    
    fh0_mhz = f0_mhz * nh  # Harmonic frequency in MHz
    vh_total_kv = vh_opt_kv  # Total harmonic voltage in kV
    
    # ========== 1. Calculate Detuning Frequency δf_h ==========
    # δf_h = -(f_h0 / Q_h0) * sqrt(((R_h_shunt * I_max) / (v_h * n_h_cav))^2 - 1/4)
    # 
    # Unit analysis:
    # R_h_shunt [MΩ] * I_max [mA] = 10^6 Ω * 10^-3 A = 10^3 V
    # v_h [kV] * n_h_cav = 10^3 V * n_h_cav
    # Ratio: (10^3 V) / (10^3 V * n_h_cav) = 1/n_h_cav [dimensionless] ✓
    
    if i_max_ma > 1e-9 and vh_total_kv > 1e-6 and nhcav > 0:
        # Numerator: R_h_shunt [MΩ] * I_max [mA]
        numerator = rh_shunt_mohm * i_max_ma
        # Denominator: v_h [kV] * n_h_cav
        denominator = vh_total_kv * nhcav
        
        if denominator > 1e-9:
            ratio = numerator / denominator
        else:
            warnings.warn(
                "Harmonic cavity: denominator (v_h * n_h_cav) is near zero. "
                "Cannot calculate detuning. Setting to 0.",
                RuntimeWarning
            )
            return {
                "detuning_khz": 0,
                "phi_hs": 0,
                "vh_cav": vh_opt_kv / nhcav if nhcav > 0 else 0
            }
        
        # Check if the term under the square root is positive
        # Physical requirement: ratio^2 >= 0.25 for passive operation
        ratio_sq = ratio**2
        
        if ratio_sq >= 0.25:  # >= 1/4
            sqrt_term = np.sqrt(ratio_sq - 0.25)
            detuning_mhz = -(fh0_mhz / qh0) * sqrt_term
            detuning_khz = detuning_mhz * 1000  # Convert MHz to kHz
        else:
            # Evanescent mode: cavity cannot sustain passive operation
            warnings.warn(
                f"Harmonic cavity detuning: ratio^2 = {ratio_sq:.4f} < 0.25. "
                f"This indicates evanescent mode operation. "
                f"Passive harmonic cavity may not function properly. "
                f"Consider: (1) Increasing beam current, (2) Reducing harmonic voltage, "
                f"or (3) Using active harmonic cavity mode.",
                RuntimeWarning
            )
            # Return NaN to signal invalid operating point
            detuning_khz = np.nan
    else:
        # Zero current or voltage case
        detuning_khz = 0
    
    # ========== 2. Calculate Harmonic Cavity Phase ==========
    # Harmonic cavity phase (using detuning and frequency)
    # φ_hs = arctan(2 * Q_h0 * δf_h / f_h0)
    if fh0_mhz > 0 and not np.isnan(detuning_khz) and abs(detuning_khz) > 1e-6:
        # Convert detuning back to MHz for phase calculation
        detuning_mhz_for_phase = detuning_khz / 1000
        phi_hs_rad = np.arctan(2 * qh0 * detuning_mhz_for_phase / fh0_mhz)
        phi_hs_deg = np.degrees(phi_hs_rad)
    else:
        phi_hs_deg = 0
    
    return {
        "detuning_khz": detuning_khz,
        "phi_hs": phi_hs_deg,
        "vh_cav": vh_opt_kv / nhcav if nhcav > 0 else 0
    }


def calculate_fundamental_cavity_params(i_max_ma, vf_cav_kv, phi_s_rad, f0_mhz, q0_fund, rshunt_mohm):
    """
    Calculate fundamental cavity detuning and related parameters
    Based on rf_calc_base.py formula:
    
    δf = Vf_cav * sin(φs) / (Q0 * Rshunt * I_max) * f0
    """
    i_beam_max_a = i_max_ma / 1000.0
    vf_volts = vf_cav_kv * 1000
    rshunt_ohms = rshunt_mohm * 1e6
    
    if i_beam_max_a > 1e-9:
        # Detuning in Hz
        detuning_hz = (vf_volts * np.sin(phi_s_rad) / (q0_fund * rshunt_ohms * i_beam_max_a)) * (f0_mhz * 1e6)
        detuning_khz = detuning_hz / 1e3
    else:
        detuning_khz = 0
    
    return {
        "detuning_khz": detuning_khz,
        "detuning_hz": detuning_khz * 1000 if i_beam_max_a > 1e-9 else 0
    }

# --- Sidebar Interaction [5] ---
st.sidebar.header("⚙️ Storage Ring Parameters")
st.sidebar.write("*Configuration Settings*")

# Operating Phase Selection (top level)
st.sidebar.subheader("🎯 Operating Phase Selection")
operation_phase = st.sidebar.radio(
    "Operating Phase:",
    options=["Phase 1", "Phase 2"],
    help="Phase 1: 487 keV (Synch.) + 316 keV (IDs) = 803 keV\nPhase 2: 487 keV (Synch.) + 359 keV (IDs) = 846 keV"
)

# Determine U0 based on phase selection
u0_phase1_kev = 803.0  # 487 + 316
u0_phase2_kev = 846.0  # 487 + 359
u0_default = u0_phase1_kev if operation_phase == "Phase 1" else u0_phase2_kev

# Storage Ring Parameters (expandable section)
with st.sidebar.expander("◇ Storage Ring", expanded=False):
    e_gev = st.number_input("Energy E₀ (GeV)", 1.0, 5.0, 2.75)
    f0_mhz = st.number_input("RF Frequency f₀ (MHz)", 100.0, 500.0, 352.2)
    h_rf = st.number_input("Harmonic Number h", 100, 1000, 416)
    fring_khz = st.number_input("Revolution Frequency f_ring (kHz)", 100.0, 1000.0, 846.63)
    imax_ma = st.number_input("Max Beam Current I_max (mA)", 100.0, 1000.0, 500.0)
    imin_ma = st.number_input("Min Beam Current I_min (mA)", 0.001, 0.1, 0.01)
    u0_kev = st.number_input("Energy Loss U₀ (keV)", 100.0, 1000.0, u0_default)
    
    # Display phase-specific information
    if operation_phase == "Phase 1":
        st.info("**Phase 1:** Synchrotron loss 487 keV + ID loss 316 keV = **803 keV**")
    else:
        st.info("**Phase 2:** Synchrotron loss 487 keV + ID loss 359 keV = **846 keV**")

# Fundamental Cavity Parameters (expandable section)
with st.sidebar.expander("● Fundamental Cavities (n=1)", expanded=True):
    ncav = st.slider("Number of Cavities", 1, 4, 4)
    fdetune = st.number_input("Detuning Frequency (kHz)", -100.0, 100.0, 0.0)
    q0_fund = st.number_input("Loaded Q (Q₀)", 10000, 100000, 35000)
    qext_fund = st.number_input("External Q (Q_ext)", 1000, 50000, 6364)
    beta_fund = st.number_input("Coupling Factor β", 1.0, 20.0, 5.5)
    rshunt_mohm = st.number_input("Shunt Impedance R (MΩ)", 0.5, 20.0, 5.0)
    vfcav_kv = st.number_input("Cavity Voltage V_cav (kV)", 50.0, 1000.0, 425.0)
    tau_cav_us = st.number_input("Cavity Decay Time τ (μs)", 1.0, 20.0, 4.87)
    
    st.markdown("---")
    st.write("**RF Feedback Control:**")
    rf_gain = st.number_input("RF Feedback Gain", 0.0, 5.0, 1.3)
    
    # Calculate bandwidths
    ql_fund = q0_fund / (1 + beta_fund)  # Loaded Q
    bp_noFB = f0_mhz / ql_fund  # Bandwidth without feedback (in kHz)
    bp_withFB = bp_noFB * (1 + rf_gain)  # Bandwidth with RF feedback
    
    col_bp1, col_bp2 = st.columns(2)
    col_bp1.metric("BW without FB (kHz)", f"{bp_noFB:.2f}")
    col_bp2.metric("BW with FB (kHz)", f"{bp_withFB:.2f}")

# Harmonic Cavity Parameters (expandable section)
with st.sidebar.expander("◐ Harmonic Cavities (n=4, 1.409 GHz)", expanded=True):
    nhcav = st.slider("Number of Harmonic Cavities", 1, 3, 2)
    
    # Fixed harmonic number at 4th harmonic (1.409 GHz)
    nh_harm = 4
    fh_calculated = f0_mhz * nh_harm / 1000  # Convert to GHz
    
    st.metric("Harmonic Number (n)", nh_harm, help="Fixed at 4th harmonic")
    st.metric("Harmonic Frequency (GHz)", f"{fh_calculated:.3f}", help=f"{f0_mhz} MHz × {nh_harm}")
    
    st.markdown("---")
    
    # **NEW: Harmonic Voltage Mode Selection**
    vh_mode = st.radio(
        "Harmonic Voltage Mode",
        ["Optimized (V_f,total / 2n_h,cav)", "Manual Fixed Value"],
        help="Choose calculation method for harmonic voltage"
    )
    
    if vh_mode == "Optimized (V_f,total / 2n_h,cav)":
        # Use formula: vh_opt = V_f_total / (2 * nhcav)
        vh_auto_kv = None  # Will be calculated in physics engine
        st.info(f"Formula: V_h,opt = V_f,total / (2 × n_h,cav)")
    else:
        # Allow manual input
        vh_auto_kv = st.number_input("Harmonic Voltage V_h (kV)", 100.0, 600.0, 400.0)
        st.info(f"Manual fixed voltage: {vh_auto_kv} kV")
    
    st.markdown("---")
    qh0 = st.number_input("Loaded Q (Q₀)", 10000, 100000, 31000)
    rhshunt_mohm = st.number_input("Shunt Impedance R (MΩ)", 0.1, 5.0, 0.92)
    beta_harm = st.number_input("Coupling Factor β", 0.0, 10.0, 0.0)
    tau_hcav_us = st.number_input("Cavity Decay Time τ (μs)", 1.0, 20.0, 7.0)

# Beam & RF Settings (expandable section)
with st.sidebar.expander("⚡ Beam & RF Settings", expanded=True):
    current_input = st.slider("Beam Current I (mA)", imin_ma, imax_ma, 250.0)
    v_rf_total = st.number_input("Main RF Voltage (kV)", 800.0, 2000.0, 1700.0)
    beta_fixed = st.number_input("Fixed Coupling Factor β", 1.0, 10.0, 5.5)

# Calculate live results
res = run_physics_engine(current_input, v_rf_total, beta_fixed, ncav, nhcav, nh_harm, rshunt_mohm, rhshunt_mohm, u0_kev, vh_auto_kv)

# --- UI Dashboard ---
st.title("RF System Analytical Dashboard (4th Harmonic)")

# Phase indicator with color coding
if operation_phase == "Phase 1":
    st.markdown(f"### 🟢 **{operation_phase}** - U₀ = {u0_kev:.0f} keV (487 keV + 316 keV IDs)")
else:
    st.markdown(f"### 🟠 **{operation_phase}** - U₀ = {u0_kev:.0f} keV (487 keV + 359 keV IDs)")

st.markdown(f"**Energy:** {e_gev} GeV | **Frequency:** {f0_mhz} MHz | **Harmonic:** n={nh_harm}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sync Phase $\\phi_s$", f"{res['phi_s']:.2f} °")
col2.metric("Optimal $\\beta$", f"{res['beta_opt']:.2f}")
col3.metric("Beam Power/Cav", f"{res['p_beam']:.2f} kW")
col4.metric("Reflection $|\\rho|$", f"{res['rho']:.3f}")

# --- Generate Scans for Plots ---
i_range = np.linspace(0.01, 500, 100)
scans = [run_physics_engine(i, v_rf_total, beta_fixed, ncav, nhcav, nh_harm, rshunt_mohm, rhshunt_mohm, u0_kev, vh_auto_kv) for i in i_range]

# Calculate harmonic cavity parameters using the cavity voltage (not optimal)
h_params = calculate_harmonic_cavity_params(imax_ma, res['vh_opt'], nhcav, nh_harm, f0_mhz, rhshunt_mohm, qh0)

# Calculate fundamental cavity detuning at max current
f_params = calculate_fundamental_cavity_params(imax_ma, v_rf_total/ncav, np.radians(res['phi_s']), 
                                              f0_mhz, q0_fund, rshunt_mohm)

# --- THE 5 CORE PLOTS [1, 6] ---
# Reorganized tabs with harmonic cavity analysis in the middle
tabs = st.tabs([
    "Plot 1: Main Cavity Power", 
    "Plot 2: Voltage Components", 
    "Harmonic Cavity Analysis",    # New: Comprehensive H-cavity analysis
    "Detuning & Phase Analysis",   # New: Detuning calculations  
    "Plot 3: Harmonic Power",
    "Energy Loss Analysis",        # New: U0, Uh0, UT0 analysis
    "RF Feedback Control",         # New: RF feedback bandwidth analysis
    "Plot 4: Potential Well", 
    "Plot 5: Reflection Match",
    "Synchronous Phase Analysis"   # Moved to last tab
])

with tabs[0]: # Plot 1: Main Cavity Power vs Current
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=i_range, y=[s['p_inc'] for s in scans], name="Incident (Pi)"))
    fig1.add_trace(go.Scatter(x=i_range, y=[s['p_beam'] for s in scans], name="Beam (Pb)"))
    fig1.add_trace(go.Scatter(x=i_range, y=[s['p_ref'] for s in scans], name="Reflected (Pr)"))
    fig1.update_layout(title="Power per Main Cavity (kW)", xaxis_title="Current (mA)", yaxis_title="kW", template="plotly_white")
    st.plotly_chart(fig1, width='stretch')

with tabs[1]: # Plot 2: Voltage Components vs Phase
    phi_deg = np.linspace(-180, 180, 1000)
    v_m = v_rf_total * np.sin(np.radians(phi_deg + res['phi_s']))
    v_h = res['vh_opt'] * np.sin(np.radians(nh_harm * phi_deg))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=phi_deg, y=v_m, name="Main (1700kV)", line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=phi_deg, y=v_h, name="Harmonic (Opt)"))
    fig2.add_trace(go.Scatter(x=phi_deg, y=v_m+v_h, name="Total Voltage", line=dict(width=3, color='black')))
    fig2.update_layout(title="RF Voltage Distribution", xaxis_title="Phase (deg)", yaxis_title="kV", template="plotly_white")
    st.plotly_chart(fig2, width='stretch')

with tabs[2]: # Harmonic Cavity Analysis
    st.subheader("💠 Harmonic Cavity Analysis (n=4)")
    
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Optimal Voltage", f"{res['vh_opt']:.2f} kV", help="Optimal harmonic voltage")
    col_h2.metric("Voltage per Cavity", f"{res['vh_cav']:.2f} kV", help="Voltage per harmonic cavity")
    col_h3.metric("Harmonic Phase φ_h,opt", f"{res['phi_h_opt']:.2f}°", help="Optimal harmonic phase")
    
    col_h4, col_h5, col_h6 = st.columns(3)
    col_h4.metric("Detuning δfh", f"{h_params['detuning_khz']:.2f} kHz", help="Required detuning at Imax")
    col_h5.metric("Power/Cavity (Imax)", f"{[s['ph_diss'] for s in scans][-1]:.2f} kW")
    col_h6.metric("Total Loss (H)", f"{nhcav * [s['ph_diss'] for s in scans][-1]:.2f} kW")
    
    # Debug section
    st.markdown("---")
    st.markdown("**Debug Information:**")
    fh0_calc = f0_mhz * nh_harm
    ratio_numerator = rhshunt_mohm * imax_ma
    ratio_denominator = res['vh_opt'] * nhcav
    if ratio_denominator > 0:
        ratio_calc = ratio_numerator / ratio_denominator
    else:
        ratio_calc = 0
    
    col_dbg1, col_dbg2, col_dbg3 = st.columns(3)
    col_dbg1.metric("fh0 (MHz)", f"{fh0_calc:.2f}")
    col_dbg2.metric("Ratio", f"{ratio_calc:.4f}")
    col_dbg3.metric("Ratio² >= 0.25?", "✓" if ratio_calc**2 >= 0.25 else "✗")
    
    st.markdown("---")
    st.write("**Calculation Formulas:**")
    st.latex(r"\phi_s = \pi - \arcsin\left[\frac{n^2}{(n^2-1)} \cdot \frac{U_0}{V_c}\right]")
    st.latex(r"V_{h,\mathrm{opt}} = \sqrt{\frac{V_c^2}{n^2} - \frac{U_0^2}{(n^2-1)}}")
    st.latex(r"\phi_{h,\mathrm{opt}} = \frac{1}{n} \arcsin\left[\frac{-U_0}{V_{h,\mathrm{opt}} (n^2-1)}\right]")

with tabs[3]: # Detuning & Phase Analysis
    st.subheader("📊 Detuning & Phase Analysis")
    
    # Display both fundamental and harmonic detuning
    col_det1, col_det2, col_det3 = st.columns(3)
    col_det1.metric("Fundamental δf (kHz)", f"{f_params['detuning_khz']:.3f}", 
                   help="Detuning at I_max for fundamental cavities")
    col_det2.metric("Harmonic δfh (kHz)", f"{h_params['detuning_khz']:.3f}",
                   help="Detuning at I_max for harmonic cavities")
    col_det3.metric("Harmonic Phase (°)", f"{h_params['phi_hs']:.2f}",
                   help="Phase of harmonic voltage")
    
    st.markdown("---")
    st.write("### Detuning Formulas")
    
    col_df1, col_df2 = st.columns(2)
    with col_df1:
        st.write("**Fundamental Cavity:**")
        st.latex(r"\delta f = \frac{V_{f,\mathrm{cav}} \cdot \sin(\phi_s) \cdot Q_0}{R_{\mathrm{shunt}} \cdot I_{\mathrm{beam,max}} \cdot f_0}")
    
    with col_df2:
        st.write("**Harmonic Cavity:**")
        st.latex(r"\delta f_h = -\frac{f_{h0}}{Q_{h0}} \sqrt{\left(\frac{R_{h,\mathrm{shunt}} \cdot I_{\mathrm{max}}}{v_h \cdot n_{h,\mathrm{cav}}}\right)^2 - \frac{1}{4}}")
    
    # Detuning scan visualization
    st.markdown("---")
    detuning_range = np.linspace(-500, 500, 100)  # kHz
    
    fig_det = go.Figure()
    fig_det.add_vline(x=f_params['detuning_khz'], line_dash="dash", line_color="blue",
                     annotation_text=f"Fund.: {f_params['detuning_khz']:.2f} kHz", 
                     annotation_position="bottom right")
    fig_det.add_vline(x=h_params['detuning_khz'], line_dash="dot", line_color="red",
                     annotation_text=f"Harm.: {h_params['detuning_khz']:.2f} kHz", 
                     annotation_position="top right")
    fig_det.update_layout(title="Operating Detuning Points", 
                          xaxis_title="Detuning (kHz)", 
                          yaxis_title="Status", 
                          template="plotly_white",
                          height=300)
    st.plotly_chart(fig_det, width='stretch')

with tabs[4]: # Plot 3: Harmonic Cavity Power vs Current
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=i_range, y=[s['ph_diss'] for s in scans], name="Ph_diss (per cavity)", line=dict(color='purple')))
    fig3.update_layout(title="Harmonic Cavity Wall Loss", xaxis_title="Current (mA)", yaxis_title="kW", template="plotly_white")
    st.plotly_chart(fig3, width='stretch')

with tabs[5]: # Energy Loss Analysis
    st.subheader("⚡ Energy Loss Analysis")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("U₀ (Base Loss)", f"{u0_kev:.2f} kV", help="Base radiation loss per turn")
    col_e2.metric("Uh₀ (Harmonic)", f"{res['uh0']:.2f} kV", help="Energy loss from harmonic cavities")
    col_e3.metric("UT₀ (Total)", f"{res['ut0']:.2f} kV", help="Total energy loss per turn")
    
    # Energy loss vs current
    fig_en = go.Figure()
    fig_en.add_trace(go.Scatter(x=i_range, y=[u0_kev]*len(i_range), name="U₀ (Base)", line=dict(dash='dot')))
    fig_en.add_trace(go.Scatter(x=i_range, y=[s['uh0'] for s in scans], name="Uh₀ (Harmonic)"))
    fig_en.add_trace(go.Scatter(x=i_range, y=[s['ut0'] for s in scans], name="UT₀ (Total)", line=dict(width=3)))
    fig_en.update_layout(title="Energy Loss vs Beam Current", xaxis_title="Current (mA)", 
                         yaxis_title="Energy Loss (kV)", template="plotly_white")
    st.plotly_chart(fig_en, width='stretch')

with tabs[6]: # RF Feedback Control
    st.subheader("📡 RF Feedback Control - Cavity Bandwidth Analysis")
    
    # Calculate actual bandwidth values
    ql_value = q0_fund / (1 + beta_fund)
    bp_no_feedback = f0_mhz / ql_value  # kHz
    bp_with_feedback = bp_no_feedback * (1 + rf_gain)  # kHz
    
    col_rf1, col_rf2, col_rf3 = st.columns(3)
    col_rf1.metric("Loaded Q (QL)", f"{ql_value:.0f}")
    col_rf2.metric("Bandwidth without FB", f"{bp_no_feedback:.2f} kHz", help="Natural cavity bandwidth: BP = f₀/QL")
    col_rf3.metric("Bandwidth with FB", f"{bp_with_feedback:.2f} kHz", help="With RF Feedback: BP_FB = BP × (1 + Gain)")
    
    st.markdown("---")
    
    # RF Feedback delay calculation
    st.write("### Maximum Gain from Feedback Delay")
    
    # User input for RF periods in delay
    rf_periods = st.slider("RF Periods in Feedback Delay (N)", 1, 1000, 704, 
                           help="Number of RF periods for 0° phase (Δt = N/f₀)")
    
    # Calculate delay and maximum gain
    omega_0 = 2 * np.pi * f0_mhz * 1e6  # rad/s
    delta_t_us = rf_periods / (f0_mhz * 1e6) * 1e6  # microseconds
    delta_t_s = rf_periods / (f0_mhz * 1e6)  # seconds
    
    # G_max = π * QL / (2 * ω₀ * Δt) - 1
    g_max = (np.pi * ql_value) / (2 * omega_0 * delta_t_s) - 1
    bp_max = bp_no_feedback * (1 + g_max)
    
    col_gm1, col_gm2, col_gm3, col_gm4 = st.columns(4)
    col_gm1.metric("Feedback Delay (μs)", f"{delta_t_us:.3f}", help=f"{rf_periods} RF periods")
    col_gm2.metric("Max Gain (G_max)", f"{g_max:.3f}", help="Maximum achievable RF feedback gain")
    col_gm3.metric("Max BW (kHz)", f"{bp_max:.2f}", help="Maximum cavity bandwidth with max gain")
    col_gm4.metric("BW Improvement", f"{bp_max/bp_no_feedback:.2f}×", help="Bandwidth increase factor")
    
    st.markdown("---")
    
    col_rf4, col_rf5 = st.columns(2)
    with col_rf4:
        st.write("**Bandwidth Formulas:**")
        st.latex(r"Q_L = \frac{Q_0}{1 + \beta}")
        st.latex(r"BP_{no-FB} = \frac{f_0}{Q_L}")
        st.latex(r"BP_{with-FB} = BP_{no-FB} \times (1 + G_{feedback})")
        st.latex(r"\Delta t = \frac{N}{f_0}")
        st.latex(r"G_{max} = \frac{\pi Q_L}{2 \omega_0 \Delta t} - 1")
    
    with col_rf5:
        st.write("**Current Parameters:**")
        st.write(f"• RF Feedback Gain: **{rf_gain:.3f}**")
        st.write(f"• Max Possible Gain: **{g_max:.3f}**")
        st.write(f"• RF Frequency: **{f0_mhz} MHz**")
        st.write(f"• Q₀: **{q0_fund}**")
        st.write(f"• β (Coupling): **{beta_fund:.2f}**")
        st.write(f"• Angular Frequency: **{omega_0/(2*np.pi*1e6):.2f} MHz**")
    
    # Bandwidth improvement visualization with max gain limit
    st.markdown("---")
    fig_bw = go.Figure()
    gain_range = np.linspace(0, max(3, g_max * 1.2), 50)
    bp_improved = [bp_no_feedback * (1 + g) for g in gain_range]
    
    fig_bw.add_trace(go.Scatter(x=gain_range, y=bp_improved, 
                                 name="Cavity Bandwidth", mode='lines', 
                                 line=dict(color='blue', width=2)))
    fig_bw.add_vline(x=rf_gain, line_dash="dash", line_color="red", 
                     annotation_text=f"Current: {rf_gain:.3f}", annotation_position="top right")
    fig_bw.add_vline(x=g_max, line_dash="dot", line_color="green",
                     annotation_text=f"Max: {g_max:.3f}", annotation_position="top left")
    fig_bw.update_layout(title="RF Feedback Bandwidth vs Gain (with Maximum Limit)",
                        xaxis_title="RF Feedback Gain",
                        yaxis_title="Cavity Bandwidth (kHz)",
                        template="plotly_white", hovermode='x unified')
    st.plotly_chart(fig_bw, width='stretch')

with tabs[7]: # Plot 4: Potential Well Representation
    phi_rad = np.linspace(-np.pi, np.pi, 1000)
    # Integral of (V_total - UT0) for potential well
    # W(φ) = ∫(V_main(φ) + V_harmonic(φ) - UT0) dφ
    v_main = v_rf_total * np.sin(phi_rad + np.radians(res['phi_s']))
    v_harmonic = res['vh_opt'] * np.sin(nh_harm * phi_rad)
    v_total = v_main + v_harmonic
    # Cumulative potential (numerical integration)
    dphi = phi_rad[1] - phi_rad[0]
    potential = np.cumsum((v_total - res['ut0']) * dphi)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=np.degrees(phi_rad), y=potential - np.min(potential), name="Potential", fill='tozeroy'))
    fig4.update_layout(title="Longitudinal Potential Well (Bunch Stretching)", xaxis_title="Phase (deg)", yaxis_title="U (arb. units)", template="plotly_white")
    st.plotly_chart(fig4, width='stretch')

with tabs[8]: # Plot 5: Reflection / Match Performance
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=i_range, y=[s['rho'] for s in scans], name="Reflection |rho|", line=dict(color='red')))
    fig5.update_layout(title="Coupler Match over Current Range", xaxis_title="Current (mA)", yaxis_title="|rho|", template="plotly_white")
    st.plotly_chart(fig5, width='stretch')

with tabs[9]: # Synchronous Phase Analysis
    st.subheader("🔄 Synchronous Phase Analysis")
    
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Synchronous Phase Φs", f"{res['phi_s']:.2f}°")
    col_p2.metric("Optimal β", f"{res['beta_opt']:.3f}")
    
    # Phase vs current
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=i_range, y=[s['phi_s'] for s in scans], name="Synchronous Phase", 
                                line=dict(color='orange', width=2)))
    fig_ph.update_layout(title="Synchronous Phase vs Beam Current", xaxis_title="Current (mA)", 
                         yaxis_title="Phase (degrees)", template="plotly_white")
    st.plotly_chart(fig_ph, width='stretch')

# Data Table for Reference [1]
st.subheader("Reference Values (at Current Setpoint)")
st.table({
    "Parameter": ["Total Voltage", "Optimal Beta", "Optimal Harmonic Voltage", "Reflection |rho|"],
    "Value": [f"{res['ut0']:.2f} kV", f"{res['beta_opt']:.2f}", f"{res['vh_opt']:.2f} kV", f"{res['rho']:.4f}"]
})