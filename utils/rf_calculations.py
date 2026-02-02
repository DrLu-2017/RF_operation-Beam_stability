"""
RF System Physics Calculations
Implements the full calculation chain for RF cavity systems.
Based on cavity_operation/rf_system_pro.py formulas.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def run_physics_engine(
    curr_ma: float,
    v_total_kv: float,
    beta_fixed: float,
    ncav: int,
    nhcav: int,
    nh: int,
    r_shunt_mohm: float,
    rh_shunt_mohm: float,
    u0_base_kv: float,
    vh_manual_kv: Optional[float] = None
) -> Dict[str, float]:
    """
    Implements the full calculation chain from the SMath sheet.
    
    Parameters:
    - curr_ma: Beam current in mA
    - v_total_kv: Total main cavity voltage in kV
    - beta_fixed: Fixed coupling factor β
    - ncav: Number of main cavities
    - nhcav: Number of harmonic cavities
    - nh: Harmonic number (e.g., 4 for 4th harmonic)
    - r_shunt_mohm: Main cavity shunt impedance in MΩ
    - rh_shunt_mohm: Harmonic cavity shunt impedance in MΩ
    - u0_base_kv: Base radiation loss per turn in kV
    - vh_manual_kv: If None, uses optimized formula vh = v_total_kv / (2 * nhcav)
                    If float, uses this as fixed harmonic voltage per cavity
    
    Returns:
        Dictionary with calculated parameters
    """
    i_beam_a = curr_ma / 1000.0
    v_cav_kv = v_total_kv / ncav
    
    # 1. Harmonic Voltage Calculation
    if vh_manual_kv is None:
        # Formula: V_h,opt = sqrt(V_c^2/n^2 - U_0^2/(n^2-1))
        # where V_c = v_total_kv, n = nh
        term1 = (v_total_kv / nh) ** 2
        term2 = (u0_base_kv ** 2) / (nh**2 - 1)
        vh_opt_kv = np.sqrt(max(0, term1 - term2))
    else:
        # Use manually specified voltage
        vh_opt_kv = vh_manual_kv
    
    vh_cav_kv = vh_opt_kv / nhcav 
    
    # 2. Harmonic Power Dissipation & Induced Loss
    # Ph_diss = V^2 / (2 * R_shunt)
    ph_diss_cav_kw = (vh_cav_kv**2) / (2 * rh_shunt_mohm * 1e3)
    p_h_total_diss_kw = nhcav * ph_diss_cav_kw
    
    # Energy loss increase due to harmonic cavities
    uh0_kv = p_h_total_diss_kw / (i_beam_a + 1e-9) if curr_ma > 0 else 0
    ut0_kv = u0_base_kv + uh0_kv
    
    # 3. Synchronous Phase - NEW FORMULA
    # φ_s = π - arcsin[n²/(n²-1) * U_0 / V_c]
    ratio_phi_s = (nh**2 / (nh**2 - 1)) * (u0_base_kv / v_total_kv)
    if abs(ratio_phi_s) <= 1:
        phi_s_rad = np.pi - np.arcsin(ratio_phi_s)
    else:
        # Fallback if ratio exceeds 1
        phi_s_rad = np.arcsin(np.clip(ut0_kv / v_total_kv, -1, 1))
    
    phi_s_deg = np.degrees(phi_s_rad)
    
    # 4. Harmonic Cavity Optimal Phase - NEW FORMULA
    # φ_h,opt = (1/n) * arcsin[-U_0 / (V_h,opt * (n²-1))]
    if vh_opt_kv > 0:
        ratio_phi_h = -u0_base_kv / (vh_opt_kv * (nh**2 - 1))
        if abs(ratio_phi_h) <= 1:
            phi_h_opt_rad = (1/nh) * np.arcsin(ratio_phi_h)
        else:
            phi_h_opt_rad = 0
    else:
        phi_h_opt_rad = 0
    
    phi_h_opt_deg = np.degrees(phi_h_opt_rad)
    
    # 5. Main Cavity Power & Coupling (Beta)
    pf_diss_cav_kw = (v_cav_kv**2) / (2 * r_shunt_mohm * 1e3)
    p_beam_per_cav_kw = (i_beam_a * ut0_kv) / ncav
    
    # Optimal Beta: Beta_opt = 1 + (2 * P_beam / P_diss_wall)
    beta_opt = 1 + (2 * p_beam_per_cav_kw) / (pf_diss_cav_kw + 1e-9)
    
    # Reflection Coefficient (Rho) based on mismatch with fixed Beta
    rho = np.abs((beta_fixed - beta_opt) / (beta_fixed + beta_opt + 1e-9))
    p_inc_kw = (pf_diss_cav_kw + p_beam_per_cav_kw) / (1 - rho**2 + 1e-9)
    p_ref_kw = p_inc_kw - p_beam_per_cav_kw - pf_diss_cav_kw
    
    return {
        "phi_s": phi_s_deg,
        "phi_s_rad": phi_s_rad,
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


def calculate_harmonic_cavity_params(
    i_max_ma: float,
    vh_opt_kv: float,
    nhcav: int,
    nh: int,
    f0_mhz: float,
    rh_shunt_mohm: float,
    qh0: float
) -> Dict[str, float]:
    """
    Calculate harmonic cavity detuning and voltage parameters
    
    Formula: δf_h = -(f_h0 / Q_h0) * sqrt(((R_h_shunt * I_max) / (v_h * n_h_cav))^2 - 1/4)
    where:
    - f_h0 = harmonic frequency (MHz)
    - Q_h0 = harmonic cavity loaded Q
    - R_h_shunt = harmonic shunt impedance (MOhm)
    - I_max = maximum beam current (mA)
    - v_h = total harmonic voltage (kV)
    - n_h_cav = number of harmonic cavities
    """
    fh0_mhz = f0_mhz * nh  # Harmonic frequency in MHz
    
    # Total harmonic voltage in kV
    vh_total_kv = vh_opt_kv
    
    # Calculate detuning frequency δf_h
    if i_max_ma > 1e-9 and vh_total_kv > 0:
        # Numerator: R_h_shunt [MOhm] * I_max [mA]
        numerator = rh_shunt_mohm * i_max_ma
        # Denominator: v_h [kV] * n_h_cav
        denominator = vh_total_kv * nhcav
        ratio = numerator / denominator
        
        # Check if the term under the square root is positive
        if ratio**2 >= 0.25:  # >= 1/4
            sqrt_term = np.sqrt(ratio**2 - 0.25)
            detuning_mhz = -(fh0_mhz / qh0) * sqrt_term
            detuning_khz = detuning_mhz * 1000  # Convert MHz to kHz
        else:
            # If the condition is not met, detuning is undefined (evanescent mode)
            detuning_khz = 0
    else:
        detuning_khz = 0
    
    # Calculate harmonic cavity phase
    if fh0_mhz > 0 and detuning_khz != 0:
        phi_hs_deg = np.degrees(np.arctan(2 * qh0 * detuning_khz / fh0_mhz))
    else:
        phi_hs_deg = 0
    
    return {
        "detuning_khz": detuning_khz,
        "phi_hs": phi_hs_deg,
        "vh_cav": vh_opt_kv / nhcav
    }


def calculate_fundamental_cavity_params(
    i_max_ma: float,
    vf_cav_kv: float,
    phi_s_rad: float,
    f0_mhz: float,
    q0_fund: float,
    rshunt_mohm: float
) -> Dict[str, float]:
    """
    Calculate fundamental cavity detuning and related parameters
    
    Formula: δf = Vf_cav * sin(φs) / (Q0 * Rshunt * I_max) * f0
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
        detuning_hz = 0
    
    return {
        "detuning_khz": detuning_khz,
        "detuning_hz": detuning_hz
    }


def calculate_rf_feedback_params(
    f0_mhz: float,
    q0_fund: float,
    beta_fund: float,
    rf_gain: float,
    rf_periods: int = 704
) -> Dict[str, float]:
    """
    Calculate RF feedback bandwidth and maximum gain parameters
    
    Parameters:
    - f0_mhz: RF frequency in MHz
    - q0_fund: Fundamental cavity Q0
    - beta_fund: Coupling factor β
    - rf_gain: RF feedback gain
    - rf_periods: Number of RF periods in feedback delay
    
    Returns:
        Dictionary with bandwidth and gain parameters
    """
    # Calculate loaded Q
    ql_value = q0_fund / (1 + beta_fund)
    
    # Bandwidth without feedback (in kHz)
    bp_no_feedback = f0_mhz / ql_value
    
    # Bandwidth with feedback
    bp_with_feedback = bp_no_feedback * (1 + rf_gain)
    
    # Calculate maximum gain from feedback delay
    omega_0 = 2 * np.pi * f0_mhz * 1e6  # rad/s
    delta_t_us = rf_periods / (f0_mhz * 1e6) * 1e6  # microseconds
    delta_t_s = rf_periods / (f0_mhz * 1e6)  # seconds
    
    # G_max = π * QL / (2 * ω₀ * Δt) - 1
    g_max = (np.pi * ql_value) / (2 * omega_0 * delta_t_s) - 1
    bp_max = bp_no_feedback * (1 + g_max)
    
    return {
        "ql": ql_value,
        "bp_no_fb": bp_no_feedback,
        "bp_with_fb": bp_with_feedback,
        "delta_t_us": delta_t_us,
        "g_max": g_max,
        "bp_max": bp_max,
        "bw_improvement": bp_max / bp_no_feedback if bp_no_feedback > 0 else 1.0
    }


def calculate_potential_well(
    v_total_kv: float,
    vh_opt_kv: float,
    phi_s_deg: float,
    nh: int,
    ut0_kv: float,
    n_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the longitudinal potential well for bunch stretching analysis
    
    Returns:
        Tuple of (phase_degrees, potential_normalized)
    """
    phi_rad = np.linspace(-np.pi, np.pi, n_points)
    
    # Calculate voltage components
    v_main = v_total_kv * np.sin(phi_rad + np.radians(phi_s_deg))
    v_harmonic = vh_opt_kv * np.sin(nh * phi_rad)
    v_total = v_main + v_harmonic
    
    # Cumulative potential (numerical integration)
    dphi = phi_rad[1] - phi_rad[0]
    potential = np.cumsum((v_total - ut0_kv) * dphi)
    
    # Normalize (shift minimum to zero)
    potential_normalized = potential - np.min(potential)
    
    return np.degrees(phi_rad), potential_normalized
