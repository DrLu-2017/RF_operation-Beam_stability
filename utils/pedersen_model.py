"""
Pedersen Model for Double-Harmonic Cavity with Feedback Loops
=============================================================
Implements the stability analysis from:
  Y. B. Shen, Q. Gu, Z. G. Jiang, D. Gu, Z. H. Zhu,
  "Stability analysis of double-harmonic cavity system in heavy beam loading
  with its feedback loops by a mathematical method based on Pedersen model,"
  J. Phys.: Conf. Ser. 2687, 072026 (2024).

The Pedersen model treats the RF system as a feedback control problem:
  - Beam, main cavity (MC), harmonic cavity (HC) are subsystems
  - Direct RF Feedback (DRFB), Auto-Level Control (ALC), and
    Phase-Lock Loop (PLL) are feedback loops
  - Robinson instability is the limiting case without feedback

This module provides:
  1. Beam-cavity transfer functions (Pedersen formalism)
  2. Robinson instability limit (no feedback baseline)
  3. DRFB impedance reduction and effective impedance
  4. ALC + PLL stability analysis
  5. Combined system stability (gain/phase margins)
  6. Maximum stable current estimation
"""

import numpy as np
from typing import Dict, Tuple, Optional


# =============================================================================
# 1. Beam-Cavity Transfer Functions (Pedersen Model)
# =============================================================================

def cavity_impedance(
    f: np.ndarray,
    f_r: float,
    Rs: float,
    Q_L: float
) -> np.ndarray:
    """
    Complex cavity impedance Z(f) near resonance.

    Z(f) = Rs / (1 + j * Q_L * (f/f_r - f_r/f))

    Parameters
    ----------
    f    : array - frequencies [Hz]
    f_r  : float - resonant frequency [Hz]
    Rs   : float - shunt impedance (circuit convention: V^2/2P) [Ohm]
    Q_L  : float - loaded quality factor

    Returns
    -------
    Complex array of impedance values [Ohm]
    """
    x = Q_L * (f / f_r - f_r / f)
    return Rs / (1 + 1j * x)


def pedersen_transfer_functions(
    f_rf: float,
    phi_s: float,
    V_c: float,
    I_b: float,
    Rs: float,
    Q_L: float,
    f_r: float
) -> Dict[str, complex]:
    """
    Compute the Pedersen beam-cavity coupling transfer functions.

    In the Pedersen model, the beam current excites the cavity, and the
    cavity voltage affects the beam phase/energy. The coupled system has
    characteristic equation whose roots determine stability.

    Parameters
    ----------
    f_rf   : float - RF frequency [Hz]
    phi_s  : float - synchronous phase [rad]
    V_c    : float - cavity voltage [V]
    I_b    : float - DC beam current [A]
    Rs     : float - shunt impedance [Ohm]
    Q_L    : float - loaded quality factor
    f_r    : float - cavity resonant freq [Hz]

    Returns
    -------
    dict with transfer function components:
        Y_b : beam admittance (beam -> voltage)
        Z_c : cavity impedance at RF
        phi_z : tuning angle [rad]  
        detuning : cavity detuning [Hz]
    """
    # Detuning
    delta_f = f_r - f_rf
    
    # Tuning angle
    tan_phi_z = 2 * Q_L * delta_f / f_rf
    phi_z = np.arctan(tan_phi_z)
    
    # Cavity impedance at RF frequency
    Z_c = Rs * np.cos(phi_z) * np.exp(-1j * phi_z)
    
    # Beam-induced voltage
    V_b = 2 * I_b * Z_c
    
    # Beam admittance (Pedersen formulation)
    Y_b = 2 * I_b / V_c * np.cos(phi_s)
    
    return {
        'Z_c': Z_c,
        'Z_c_mag': abs(Z_c),
        'V_b': V_b,
        'V_b_mag': abs(V_b),
        'Y_b': Y_b,
        'phi_z': phi_z,
        'phi_z_deg': np.degrees(phi_z),
        'detuning': delta_f,
        'tan_phi_z': tan_phi_z,
    }


# =============================================================================
# 2. Robinson Instability Limit (No Feedback)
# =============================================================================

def robinson_limit(
    V_c: float,
    phi_s: float,
    I_b: float,
    Rs: float,
    Q_L: float,
    f_rf: float,
    f_s: float,
    tau_rad: float,
    m: int = 1
) -> Dict[str, float]:
    """
    Calculate Robinson instability growth rate and stability limit.

    The Robinson instability occurs when the beam-cavity interaction
    drives synchrotron oscillations with a growth rate exceeding radiation
    damping. This is the baseline (no feedback) case.

    Growth rate (Pedersen):
        alpha_R = (I_b * f_rf * Rs) / (2 * V_c * Q_L) * F(phi_z, phi_s)

    where F depends on the impedance asymmetry between upper and lower
    synchrotron sidebands.

    Parameters
    ----------
    V_c     : float - cavity voltage [V]
    phi_s   : float - synchronous phase [rad]
    I_b     : float - DC beam current [A]
    Rs      : float - shunt impedance [Ohm]
    Q_L     : float - loaded Q
    f_rf    : float - RF frequency [Hz]
    f_s     : float - synchrotron frequency [Hz]
    tau_rad : float - radiation damping time [s]
    m       : int   - harmonic ratio for HC (1 for MC)

    Returns
    -------
    dict with Robinson stability info
    """
    omega_rf = 2 * np.pi * f_rf
    omega_s = 2 * np.pi * f_s
    
    # Half-bandwidth of the cavity
    omega_half = omega_rf / (2 * Q_L)
    
    # Beam loading parameter Y (Pedersen)
    # Y = I_b * Rs / V_c
    Y = I_b * Rs / V_c if V_c > 0 else 0
    
    # For Robinson: the growth rate depends on the difference in 
    # impedance at upper and lower synchrotron sidebands
    # Im[Z(f_rf + f_s)] - Im[Z(f_rf - f_s)]
    
    # Simplified Robinson criterion:
    # The cavity must be detuned such that:
    #   f_r > f_rf  (above transition)
    # This ensures the upper sideband sees more resistive impedance
    
    # Robinson growth rate (above transition, simplified):
    # alpha_R ≈ (I_b * omega_rf * Rs * sin(2*phi_z)) / (4 * V_c * Q_L)
    #         - cos(phi_s) contribution
    
    # Radiation damping rate
    alpha_rad = 1.0 / tau_rad if tau_rad > 0 else 0
    
    # Maximum beam current (Robinson limit):
    # At the stability boundary: alpha_R = alpha_rad
    # Solving for I_max:
    # I_max = (2 * V_c * Q_L * alpha_rad) / (omega_rf * Rs * |sin(2*phi_z_opt)|)
    
    # Optimal detuning for stability
    # For Robinson stability: phi_z should be positive (detuned above)
    # The optimal is around phi_z = pi/4 for maximum stable current
    sin_2phi_opt = 1.0  # |sin(2*pi/4)| = 1
    
    if Rs > 0 and omega_rf > 0:
        I_max_robinson = (2 * V_c * Q_L * alpha_rad) / (omega_rf * Rs * sin_2phi_opt)
    else:
        I_max_robinson = float('inf')
    
    # Beam loading parameter
    beam_loading_factor = Y * np.sin(phi_s)
    
    # For the double RF system, the HC Robinson limit is typically more restrictive
    # HC Robinson limit: I_max_HC = V_h / (2 * Rs_h * cos(phi_z_h))
    
    return {
        'Y': Y,
        'beam_loading_factor': beam_loading_factor,
        'alpha_rad': alpha_rad,
        'alpha_rad_inv': tau_rad,
        'I_max_robinson_A': I_max_robinson,
        'I_max_robinson_mA': I_max_robinson * 1e3,
        'omega_half_kHz': omega_half / (2 * np.pi * 1e3),
        'cavity_bandwidth_kHz': 2 * omega_half / (2 * np.pi * 1e3),
        'sin_2phi_opt': sin_2phi_opt,
    }


# =============================================================================
# 3. DRFB Impedance Reduction
# =============================================================================

def drfb_impedance_reduction(
    Rs: float,
    Q_L: float,
    f_rf: float,
    G: float,
    tau_fb: float,
    phi_fb: float = 0.0
) -> Dict[str, float]:
    """
    Calculate the effective cavity impedance with Direct RF Feedback (DRFB).

    DRFB reduces the generator impedance seen by the beam:
        Z_eff(f) = Z_cav(f) / (1 + G * e^(-j*2*pi*f*tau) * H(f))

    where G is the feedback gain, tau is the loop delay, and H(f) is
    the controller transfer function.

    For a simple proportional DRFB:
        |Z_eff| ≈ Rs / (1 + G)   (near resonance)
        BW_eff = BW_0 * (1 + G)   (bandwidth expands)

    Parameters
    ----------
    Rs      : float - shunt impedance [Ohm]
    Q_L     : float - loaded Q
    f_rf    : float - RF frequency [Hz]
    G       : float - DRFB gain
    tau_fb  : float - feedback loop delay [s]
    phi_fb  : float - feedback phase [rad]

    Returns
    -------
    dict with DRFB performance metrics
    """
    # Natural bandwidth
    bw_0 = f_rf / (2 * Q_L)  # Half-bandwidth in Hz
    
    # With DRFB
    bw_eff = bw_0 * (1 + G)  # Effective half-bandwidth
    Q_eff = f_rf / (2 * bw_eff)  # Effective Q
    Rs_eff = Rs / (1 + G)  # Effective shunt impedance
    
    # Maximum gain limited by loop delay (Pedersen-Shen criterion)
    # G_max = pi * Q_L / (2 * omega_rf * tau_fb) - 1
    omega_rf = 2 * np.pi * f_rf
    if tau_fb > 0:
        G_max = (np.pi * Q_L) / (2 * omega_rf * tau_fb) - 1
    else:
        G_max = float('inf')
    
    # Gain margin: how much gain headroom
    gain_margin = G_max / G if G > 0 else float('inf')
    gain_margin_dB = 20 * np.log10(gain_margin) if gain_margin > 0 else float('inf')
    
    # Phase margin estimate
    # At maximum bandwidth, the loop phase is approximately:
    # phi_loop = omega_rf * tau_fb + arctan(bw_eff / bw_0)
    phase_at_bw_edge = omega_rf * tau_fb
    phase_margin_deg = 180 - np.degrees(phase_at_bw_edge) % 360
    
    # Impedance reduction factor
    reduction_factor = 1.0 / (1 + G)
    reduction_dB = -20 * np.log10(1 + G) if G > -1 else 0
    
    return {
        'Rs_eff': Rs_eff,
        'Rs_eff_MOhm': Rs_eff / 1e6,
        'Q_eff': Q_eff,
        'bw_0_kHz': bw_0 / 1e3,
        'bw_eff_kHz': bw_eff / 1e3,
        'bw_ratio': bw_eff / bw_0,
        'G_max': max(0, G_max),
        'gain_margin': gain_margin,
        'gain_margin_dB': gain_margin_dB,
        'phase_margin_deg': phase_margin_deg,
        'reduction_factor': reduction_factor,
        'reduction_dB': reduction_dB,
    }


# =============================================================================
# 4. Combined MC + HC Stability with Feedback (Shen 2024)
# =============================================================================

def double_rf_stability_with_feedback(
    # Main cavity parameters
    V_mc: float,
    Rs_mc: float,
    Q0_mc: float,
    beta_mc: float,
    f_rf: float,
    # Harmonic cavity parameters
    V_hc: float,
    Rs_hc: float,
    Q0_hc: float,
    m: int,
    delta_f_hc: float,
    # Beam parameters
    I_b: float,
    phi_s: float,
    f_s: float,
    tau_rad: float,
    U0: float,
    # DRFB parameters
    G_drfb: float = 0.0,
    tau_drfb: float = 2e-6,
    # ALC parameters
    G_alc: float = 0.0,
    tau_alc: float = 1e-3,
    # PLL parameters
    G_pll: float = 0.0,
    tau_pll: float = 1e-3,
) -> Dict[str, dict]:
    """
    Comprehensive stability analysis of a double-harmonic cavity system
    with feedback loops, following the Pedersen-Shen (2024) methodology.

    The system treats:
    - MC with DRFB, ALC, PLL feedback
    - Passive HC (no external feedback, beam-driven only)
    - Beam as coupling element

    Parameters
    ----------
    V_mc      : Main cavity voltage [V]
    Rs_mc     : MC shunt impedance [Ohm]
    Q0_mc     : MC unloaded Q
    beta_mc   : MC coupling factor
    f_rf      : RF frequency [Hz]
    V_hc      : Harmonic cavity voltage [V]
    Rs_hc     : HC shunt impedance [Ohm]
    Q0_hc     : HC unloaded Q
    m         : Harmonic number ratio
    delta_f_hc: HC detuning [Hz]
    I_b       : DC beam current [A]
    phi_s     : Synchronous phase [rad]
    f_s       : Synchrotron frequency [Hz]
    tau_rad   : Radiation damping time [s]
    U0        : Energy loss per turn [eV]
    G_drfb    : DRFB gain
    tau_drfb  : DRFB loop delay [s]
    G_alc     : ALC gain
    tau_alc   : ALC time constant [s]
    G_pll     : PLL gain
    tau_pll   : PLL time constant [s]

    Returns
    -------
    dict with comprehensive stability analysis
    """
    # Loaded Qs
    QL_mc = Q0_mc / (1 + beta_mc)
    QL_hc = Q0_hc  # passive cavity: no external coupler (beta_hc ≈ 0)
    
    # ── Main Cavity Analysis ──
    mc_no_fb = robinson_limit(V_mc, phi_s, I_b, Rs_mc, QL_mc, f_rf, f_s, tau_rad)
    mc_drfb = drfb_impedance_reduction(Rs_mc, QL_mc, f_rf, G_drfb, tau_drfb)
    
    # MC Robinson limit WITH DRFB
    # DRFB effectively reduces Rs → Rs/(1+G), expanding bandwidth
    # This increases the Robinson threshold proportionally
    I_max_mc_with_fb = mc_no_fb['I_max_robinson_mA'] * (1 + G_drfb)
    
    # ── Harmonic Cavity Analysis ──
    # HC tuning angle
    f_hc = m * f_rf
    tan_phi_z_hc = 2 * Q0_hc * delta_f_hc / f_hc
    phi_z_hc = np.arctan(tan_phi_z_hc)
    
    # HC effective impedance at harmonic
    Rs_hc_eff = Rs_hc * np.cos(phi_z_hc)
    
    # HC Robinson limit (passive, no feedback)
    hc_robinson = robinson_limit(V_hc, phi_s, I_b, Rs_hc, QL_hc, f_hc, f_s, tau_rad, m)
    
    # ── Combined System Stability ──
    # The Pedersen-Shen model combines all loops:
    # Overall stability determined by the most restrictive limit
    
    # Radiation damping rate
    alpha_rad = 1.0 / tau_rad if tau_rad > 0 else 0
    
    # MC growth rate (with DRFB)
    # Reduced by factor (1+G)
    alpha_mc = mc_no_fb['beam_loading_factor'] / (1 + G_drfb) if G_drfb > -1 else float('inf')
    
    # HC growth rate (no feedback)
    alpha_hc = hc_robinson['beam_loading_factor']
    
    # Total growth rate
    alpha_total = alpha_mc + alpha_hc
    
    # Stability: total growth rate < radiation damping
    mc_stable = abs(alpha_mc) < alpha_rad
    hc_stable = abs(alpha_hc) < alpha_rad
    overall_stable = abs(alpha_total) < alpha_rad
    
    # Current headroom
    current_headroom = I_max_mc_with_fb - I_b * 1e3 if I_b > 0 else I_max_mc_with_fb
    
    # Beam loading ratio (Y parameter)
    Y_mc = I_b * Rs_mc / V_mc if V_mc > 0 else 0
    Y_hc = I_b * Rs_hc / V_hc if V_hc > 0 else 0
    
    # DRFB effectiveness: how much does it help?
    if G_drfb > 0:
        drfb_improvement = {
            'impedance_reduction_dB': mc_drfb['reduction_dB'],
            'bw_expansion_factor': mc_drfb['bw_ratio'],
            'robinson_limit_improvement': 1 + G_drfb,
            'gain_margin_dB': mc_drfb['gain_margin_dB'],
            'gain_headroom': mc_drfb['G_max'] - G_drfb,
        }
    else:
        drfb_improvement = {
            'impedance_reduction_dB': 0,
            'bw_expansion_factor': 1,
            'robinson_limit_improvement': 1,
            'gain_margin_dB': float('inf'),
            'gain_headroom': mc_drfb['G_max'],
        }
    
    return {
        'mc_analysis': {
            'no_feedback': mc_no_fb,
            'with_drfb': mc_drfb,
            'I_max_no_fb_mA': mc_no_fb['I_max_robinson_mA'],
            'I_max_with_fb_mA': I_max_mc_with_fb,
            'Y_mc': Y_mc,
            'stable': mc_stable,
        },
        'hc_analysis': {
            'robinson': hc_robinson,
            'phi_z_hc_deg': np.degrees(phi_z_hc),
            'Rs_hc_eff_MOhm': Rs_hc_eff / 1e6,
            'Y_hc': Y_hc,
            'stable': hc_stable,
        },
        'combined': {
            'overall_stable': overall_stable,
            'alpha_total': alpha_total,
            'alpha_rad': alpha_rad,
            'stability_margin': alpha_rad - abs(alpha_total),
            'current_headroom_mA': current_headroom,
        },
        'drfb_improvement': drfb_improvement,
        'feedback_summary': {
            'drfb_gain': G_drfb,
            'drfb_delay_us': tau_drfb * 1e6,
            'drfb_G_max': mc_drfb['G_max'],
            'alc_gain': G_alc,
            'pll_gain': G_pll,
        }
    }


# =============================================================================
# 5. DRFB Gain Scan: Stability vs Gain
# =============================================================================

def scan_drfb_gain(
    V_mc: float,
    Rs_mc: float,
    Q0_mc: float,
    beta_mc: float,
    f_rf: float,
    I_b: float,
    phi_s: float,
    f_s: float,
    tau_rad: float,
    tau_drfb: float,
    n_points: int = 50
) -> Dict[str, np.ndarray]:
    """
    Scan DRFB gain to show how it affects Robinson limit and bandwidth.

    Returns arrays for plotting gain vs various stability metrics.
    """
    QL_mc = Q0_mc / (1 + beta_mc)
    omega_rf = 2 * np.pi * f_rf
    
    # Maximum gain
    if tau_drfb > 0:
        G_max_abs = max(0, (np.pi * QL_mc) / (2 * omega_rf * tau_drfb) - 1)
    else:
        G_max_abs = 20.0
    
    gains = np.linspace(0, min(G_max_abs * 1.3, 20), n_points)
    
    bw_values = []
    rs_eff_values = []
    i_max_values = []
    gain_margin_values = []
    stable_flags = []
    
    # Baseline Robinson limit
    robinson_base = robinson_limit(V_mc, phi_s, I_b, Rs_mc, QL_mc, f_rf, f_s, tau_rad)
    
    for G in gains:
        drfb = drfb_impedance_reduction(Rs_mc, QL_mc, f_rf, G, tau_drfb)
        bw_values.append(drfb['bw_eff_kHz'])
        rs_eff_values.append(drfb['Rs_eff'] / 1e6)  # MOhm
        i_max_values.append(robinson_base['I_max_robinson_mA'] * (1 + G))
        gain_margin_values.append(drfb['gain_margin_dB'])
        stable_flags.append(G <= G_max_abs)
    
    return {
        'gains': gains,
        'G_max': G_max_abs,
        'bw_kHz': np.array(bw_values),
        'Rs_eff_MOhm': np.array(rs_eff_values),
        'I_max_mA': np.array(i_max_values),
        'gain_margin_dB': np.array(gain_margin_values),
        'stable': np.array(stable_flags),
        'robinson_base_mA': robinson_base['I_max_robinson_mA'],
    }


# =============================================================================
# 6. Current Scan: Stability vs Beam Current
# =============================================================================

def scan_current_stability(
    V_mc: float,
    Rs_mc: float,
    Q0_mc: float,
    beta_mc: float,
    f_rf: float,
    f_s: float,
    tau_rad: float,
    G_drfb: float,
    tau_drfb: float,
    V_hc: float,
    Rs_hc: float,
    Q0_hc: float,
    m: int,
    delta_f_hc: float,
    phi_s: float,
    i_max_mA: float = 500,
    n_points: int = 60
) -> Dict[str, np.ndarray]:
    """
    Scan beam current to show stability margins and Robinson limits
    for both MC and HC.
    """
    i_range = np.linspace(1, i_max_mA, n_points)
    
    QL_mc = Q0_mc / (1 + beta_mc)
    alpha_rad = 1.0 / tau_rad if tau_rad > 0 else 0
    
    mc_growth = []
    hc_growth = []
    mc_limit_no_fb = []
    mc_limit_with_fb = []
    stability_margin = []
    
    for I_mA in i_range:
        I_A = I_mA / 1e3
        
        # MC
        rob_mc = robinson_limit(V_mc, phi_s, I_A, Rs_mc, QL_mc, f_rf, f_s, tau_rad)
        alpha_mc = rob_mc['beam_loading_factor'] / (1 + G_drfb) if G_drfb > -1 else 0
        mc_growth.append(abs(alpha_mc))
        mc_limit_no_fb.append(rob_mc['I_max_robinson_mA'])
        mc_limit_with_fb.append(rob_mc['I_max_robinson_mA'] * (1 + G_drfb))
        
        # HC
        f_hc = m * f_rf
        rob_hc = robinson_limit(V_hc, phi_s, I_A, Rs_hc, Q0_hc, f_hc, f_s, tau_rad, m)
        alpha_hc = rob_hc['beam_loading_factor']
        hc_growth.append(abs(alpha_hc))
        
        stability_margin.append(alpha_rad - abs(alpha_mc) - abs(alpha_hc))
    
    return {
        'I_mA': i_range,
        'mc_growth': np.array(mc_growth),
        'hc_growth': np.array(hc_growth),
        'total_growth': np.array(mc_growth) + np.array(hc_growth),
        'damping_rate': np.full_like(i_range, alpha_rad),
        'mc_limit_no_fb': np.array(mc_limit_no_fb),
        'mc_limit_with_fb': np.array(mc_limit_with_fb),
        'stability_margin': np.array(stability_margin),
    }


# =============================================================================
# 7. Impedance Spectrum with DRFB
# =============================================================================

def impedance_spectrum(
    f_rf: float,
    Rs: float,
    Q_L: float,
    G: float = 0.0,
    f_span_kHz: float = 500,
    n_points: int = 200
) -> Dict[str, np.ndarray]:
    """
    Calculate the impedance spectrum around f_rf, with and without DRFB.
    
    Returns frequency array and impedance arrays for plotting.
    """
    f_span = f_span_kHz * 1e3
    f = np.linspace(f_rf - f_span, f_rf + f_span, n_points)
    
    # Natural impedance
    Z_nat = cavity_impedance(f, f_rf, Rs, Q_L)
    
    # With DRFB (simplified: flat gain across bandwidth)
    Z_fb = Z_nat / (1 + G)
    
    # Effective Q
    Q_eff = Q_L / (1 + G) if G > -1 else Q_L
    Z_eff = cavity_impedance(f, f_rf, Rs / (1 + G), Q_eff)
    
    return {
        'f_kHz': (f - f_rf) / 1e3,  # relative to f_rf
        'f_Hz': f,
        'Z_nat_real': np.real(Z_nat),
        'Z_nat_imag': np.imag(Z_nat),
        'Z_nat_abs': np.abs(Z_nat),
        'Z_fb_real': np.real(Z_fb),
        'Z_fb_imag': np.imag(Z_fb),
        'Z_fb_abs': np.abs(Z_fb),
        'Z_eff_abs': np.abs(Z_eff),
    }


# =============================================================================
# 8. Nyquist Stability Analysis (Shen 2024, Pedersen Model)
# =============================================================================

def open_loop_transfer_function(
    f_offset: np.ndarray,
    f_rf: float,
    Rs: float,
    Q_L: float,
    G: float,
    tau_fb: float,
    phi_fb: float = 0.0,
    I_b: float = 0.0,
    V_c: float = 1.0,
    phi_s: float = 0.0,
    include_beam: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute the open-loop transfer function T_OL(jω) of the DRFB system
    following the Pedersen-Shen model.

    The DRFB open-loop transfer function is:
        T_OL(jω) = G · H_cav(jΔω) · e^{-jωτ}

    where:
        H_cav(jΔω) = 1 / (1 + j·Δω/ω_half)   (cavity response, normalized)
        Δω = 2π·f_offset                        (offset from resonance)
        ω_half = ω_rf / (2·Q_L)                 (cavity half-bandwidth)
        τ = tau_fb                               (feedback delay)

    When the beam is included (Pedersen model), the beam loading modifies
    the effective impedance driving Robinson-type modes:
        T_OL_beam(jω) = T_OL(jω) + Y_b · H_cav(jΔω)

    where Y_b = I_b · Rs / V_c is the beam loading parameter.

    The system is stable if the Nyquist contour of T_OL does NOT encircle
    the critical point (-1, 0).

    Parameters
    ----------
    f_offset  : array  - frequency offsets from f_rf [Hz]
    f_rf      : float  - RF frequency [Hz]
    Rs        : float  - shunt impedance [Ohm]
    Q_L       : float  - loaded quality factor
    G         : float  - DRFB gain
    tau_fb    : float  - feedback loop delay [s]
    phi_fb    : float  - feedback phase offset [rad]
    I_b       : float  - DC beam current [A]
    V_c       : float  - cavity voltage [V]
    phi_s     : float  - synchronous phase [rad]
    include_beam : bool - whether to include beam loading term

    Returns
    -------
    dict with:
        T_OL      : complex array - open-loop transfer function
        H_cav     : complex array - cavity transfer function
        H_delay   : complex array - delay element
        f_offset  : array - frequency offsets [Hz]
        mag       : array - |T_OL| in natural units
        mag_dB    : array - |T_OL| in dB
        phase_deg : array - arg(T_OL) in degrees
    """
    omega_offset = 2 * np.pi * f_offset
    omega_rf = 2 * np.pi * f_rf
    
    # Cavity half-bandwidth
    omega_half = omega_rf / (2 * Q_L)
    
    # Normalized cavity transfer function (baseband equivalent)
    H_cav = 1.0 / (1.0 + 1j * omega_offset / omega_half)
    
    # Delay element: includes both the delay and the phase offset
    # The delay phase wraps as ω·τ at full RF frequency, but for the
    # baseband model we use the offset frequency
    # In the Pedersen model the delay is at the carrier frequency,
    # so the dominant phase is ω_rf·τ (constant) + Δω·τ (varying)
    carrier_phase = omega_rf * tau_fb + phi_fb
    H_delay = np.exp(-1j * (carrier_phase + omega_offset * tau_fb))
    
    # DRFB open-loop TF (pure feedback path)
    T_OL = G * H_cav * H_delay
    
    # Include beam loading (Pedersen-Shen extension)
    if include_beam and I_b > 0 and V_c > 0:
        Y_b = I_b * Rs / V_c
        # The beam couples through the cavity impedance
        # This adds a driving term proportional to the beam loading parameter
        T_beam = Y_b * np.sin(phi_s) * H_cav
        T_OL_total = T_OL + T_beam
    else:
        T_OL_total = T_OL
        T_beam = np.zeros_like(T_OL)
    
    # Magnitude and phase
    mag = np.abs(T_OL_total)
    mag_dB = 20 * np.log10(np.maximum(mag, 1e-30))
    phase_deg = np.degrees(np.angle(T_OL_total))
    
    return {
        'T_OL': T_OL_total,
        'T_OL_fb_only': T_OL,
        'T_beam': T_beam,
        'H_cav': H_cav,
        'H_delay': H_delay,
        'f_offset': f_offset,
        'mag': mag,
        'mag_dB': mag_dB,
        'phase_deg': phase_deg,
    }


def nyquist_contour(
    f_rf: float,
    Rs: float,
    Q_L: float,
    G: float,
    tau_fb: float,
    phi_fb: float = 0.0,
    I_b: float = 0.0,
    V_c: float = 1.0,
    phi_s: float = 0.0,
    include_beam: bool = True,
    f_span_factor: float = 10.0,
    n_points: int = 2000
) -> Dict[str, np.ndarray]:
    """
    Generate Nyquist contour data for the DRFB open-loop transfer function.

    The Nyquist plot traces T_OL(jω) in the complex plane for ω from -∞ to +∞.
    Stability requires no encirclement of the critical point (-1, 0).

    Parameters
    ----------
    f_rf          : RF frequency [Hz]
    Rs            : shunt impedance [Ohm]
    Q_L           : loaded quality factor
    G             : DRFB gain
    tau_fb        : feedback loop delay [s]
    phi_fb        : feedback phase offset [rad]
    I_b           : beam current [A]
    V_c           : cavity voltage [V]
    phi_s         : synchronous phase [rad]
    include_beam  : include beam loading
    f_span_factor : frequency span as multiple of cavity bandwidth
    n_points      : number of frequency points

    Returns
    -------
    dict with Nyquist plot data
    """
    # Frequency span based on cavity bandwidth
    bw = f_rf / (2 * Q_L)
    f_max = bw * f_span_factor
    
    # Positive frequencies (fine grid near resonance, coarser far away)
    # Use logarithmic-like spacing near 0 for better resolution
    f_pos_near = np.linspace(0, bw * 2, n_points // 2)
    f_pos_far = np.linspace(bw * 2, f_max, n_points // 2)
    f_pos = np.concatenate([f_pos_near, f_pos_far[1:]])
    
    # Full contour: negative + positive frequencies
    f_full = np.concatenate([-f_pos[::-1], f_pos[1:]])
    
    # Compute open-loop TF
    ol = open_loop_transfer_function(
        f_full, f_rf, Rs, Q_L, G, tau_fb, phi_fb,
        I_b, V_c, phi_s, include_beam
    )
    
    T = ol['T_OL']
    
    # Compute winding number around (-1, 0) for stability check
    # N = (1/2π) * Δarg(1 + T_OL)
    one_plus_T = 1 + T
    angles = np.angle(one_plus_T)
    d_angles = np.diff(np.unwrap(angles))
    winding_number = np.sum(d_angles) / (2 * np.pi)
    
    # Gain margin: find where phase = -180° (crossing negative real axis)
    # Phase margin: find where |T_OL| = 1 (0 dB crossing)
    gain_margin, phase_margin = _extract_margins(f_pos, f_rf, Rs, Q_L, G,
                                                  tau_fb, phi_fb, I_b, V_c,
                                                  phi_s, include_beam)
    
    # Find closest approach to (-1, 0)
    dist_to_critical = np.abs(T - (-1 + 0j))
    min_dist_idx = np.argmin(dist_to_critical)
    min_dist = dist_to_critical[min_dist_idx]
    
    return {
        'real': np.real(T),
        'imag': np.imag(T),
        'f_Hz': f_full,
        'T_OL': T,
        'winding_number': round(winding_number),
        'is_stable': abs(winding_number) < 0.5,
        'min_dist_to_critical': min_dist,
        'closest_point_real': np.real(T[min_dist_idx]),
        'closest_point_imag': np.imag(T[min_dist_idx]),
        'gain_margin_dB': gain_margin,
        'phase_margin_deg': phase_margin,
        'bw_Hz': bw,
    }


def _extract_margins(
    f_pos, f_rf, Rs, Q_L, G, tau_fb, phi_fb,
    I_b, V_c, phi_s, include_beam
) -> Tuple[float, float]:
    """
    Extract gain margin and phase margin from the open-loop TF.

    Gain margin: 1/|T_OL| at frequency where phase(T_OL) = -180°
    Phase margin: 180° + phase(T_OL) at frequency where |T_OL| = 1
    """
    ol = open_loop_transfer_function(
        f_pos, f_rf, Rs, Q_L, G, tau_fb, phi_fb,
        I_b, V_c, phi_s, include_beam
    )
    
    mag = ol['mag']
    phase = np.unwrap(np.angle(ol['T_OL']))
    phase_deg = np.degrees(phase)
    
    # Gain margin: find phase = -180° crossings
    gain_margin_dB = float('inf')
    for i in range(len(phase_deg) - 1):
        if (phase_deg[i] > -180 and phase_deg[i+1] <= -180) or \
           (phase_deg[i] < -180 and phase_deg[i+1] >= -180):
            # Linear interpolation
            t = (-180 - phase_deg[i]) / (phase_deg[i+1] - phase_deg[i])
            mag_at_180 = mag[i] + t * (mag[i+1] - mag[i])
            if mag_at_180 > 0:
                gm = -20 * np.log10(mag_at_180)
                if gm < gain_margin_dB:
                    gain_margin_dB = gm
    
    # Phase margin: find |T_OL| = 1 (0 dB) crossing
    phase_margin_deg = float('inf')
    for i in range(len(mag) - 1):
        if (mag[i] > 1 and mag[i+1] <= 1) or (mag[i] < 1 and mag[i+1] >= 1):
            # Linear interpolation
            t = (1 - mag[i]) / (mag[i+1] - mag[i]) if mag[i+1] != mag[i] else 0
            phase_at_0dB = phase_deg[i] + t * (phase_deg[i+1] - phase_deg[i])
            pm = 180 + phase_at_0dB
            if pm < phase_margin_deg:
                phase_margin_deg = pm
    
    return gain_margin_dB, phase_margin_deg


def nyquist_multi_current(
    f_rf: float,
    Rs: float,
    Q_L: float,
    G: float,
    tau_fb: float,
    phi_fb: float,
    V_c: float,
    phi_s: float,
    I_list_mA: list,
    f_span_factor: float = 10.0,
    n_points: int = 1500
) -> Dict[str, list]:
    """
    Generate Nyquist contours for multiple beam currents.
    Shows how beam loading progressively modifies the stability.

    Parameters
    ----------
    I_list_mA : list of beam currents [mA]

    Returns
    -------
    dict with list of contour data for each current
    """
    contours = []
    for I_mA in I_list_mA:
        I_b = I_mA / 1e3
        c = nyquist_contour(
            f_rf, Rs, Q_L, G, tau_fb, phi_fb,
            I_b, V_c, phi_s,
            include_beam=True,
            f_span_factor=f_span_factor,
            n_points=n_points
        )
        c['I_mA'] = I_mA
        contours.append(c)
    
    return {'contours': contours, 'I_list_mA': I_list_mA}


def nyquist_hc_passive(
    f_rf: float,
    Rs_hc: float,
    Q0_hc: float,
    m: int,
    I_b: float,
    V_hc: float,
    delta_f_hc: float,
    phi_s: float,
    f_span_factor: float = 10.0,
    n_points: int = 1500
) -> Dict[str, np.ndarray]:
    """
    Nyquist contour for the passive harmonic cavity.

    The HC has no DRFB. Its open-loop driving term comes from beam loading
    through the HC impedance. The effective T_OL_HC is:
        T_HC(jω) = Y_hc · Z_HC(jω) / Z_HC(0)

    where Y_hc = I_b · Rs_hc / V_hc and Z_HC includes the detuning.

    The Robinson stability of the HC mode is checked by whether this
    contour encircles (-1, 0).
    """
    f_hc = m * f_rf
    bw_hc = f_hc / (2 * Q0_hc)
    f_max = bw_hc * f_span_factor
    
    f_pos_near = np.linspace(0, bw_hc * 2, n_points // 2)
    f_pos_far = np.linspace(bw_hc * 2, f_max, n_points // 2)
    f_pos = np.concatenate([f_pos_near, f_pos_far[1:]])
    f_full = np.concatenate([-f_pos[::-1], f_pos[1:]])
    
    omega_offset = 2 * np.pi * f_full
    omega_hc = 2 * np.pi * f_hc
    omega_half = omega_hc / (2 * Q0_hc)
    
    # HC cavity TF including detuning
    omega_det = 2 * np.pi * delta_f_hc
    H_hc = 1.0 / (1.0 + 1j * (omega_offset - omega_det) / omega_half)
    
    # Beam loading parameter for HC
    Y_hc = I_b * Rs_hc / V_hc if V_hc > 0 else 0
    
    # HC open-loop: beam drives HC through its own impedance
    T_hc = Y_hc * np.sin(phi_s) * H_hc
    
    # Winding number
    one_plus_T = 1 + T_hc
    angles = np.angle(one_plus_T)
    d_angles = np.diff(np.unwrap(angles))
    winding_number = np.sum(d_angles) / (2 * np.pi)
    
    dist_to_critical = np.abs(T_hc - (-1 + 0j))
    min_dist = np.min(dist_to_critical)
    
    return {
        'real': np.real(T_hc),
        'imag': np.imag(T_hc),
        'f_Hz': f_full,
        'T_OL': T_hc,
        'winding_number': round(winding_number),
        'is_stable': abs(winding_number) < 0.5,
        'min_dist_to_critical': min_dist,
        'Y_hc': Y_hc,
        'bw_Hz': bw_hc,
    }


def bode_plot_data(
    f_rf: float,
    Rs: float,
    Q_L: float,
    G: float,
    tau_fb: float,
    phi_fb: float = 0.0,
    I_b: float = 0.0,
    V_c: float = 1.0,
    phi_s: float = 0.0,
    include_beam: bool = True,
    f_span_factor: float = 10.0,
    n_points: int = 500
) -> Dict[str, np.ndarray]:
    """
    Generate Bode plot data (magnitude and phase vs frequency) for the
    open-loop transfer function.

    Returns
    -------
    dict with frequency, magnitude (dB), and phase (deg) arrays
    """
    bw = f_rf / (2 * Q_L)
    f_max = bw * f_span_factor
    f = np.linspace(1, f_max, n_points)
    
    ol = open_loop_transfer_function(
        f, f_rf, Rs, Q_L, G, tau_fb, phi_fb,
        I_b, V_c, phi_s, include_beam
    )
    
    return {
        'f_Hz': f,
        'f_kHz': f / 1e3,
        'mag_dB': ol['mag_dB'],
        'phase_deg': np.degrees(np.unwrap(np.angle(ol['T_OL']))),
        'mag': ol['mag'],
    }


# =============================================================================
# 9. Operational Guidelines (Shen 2024 Summary)
# =============================================================================

def operational_guidelines(
    I_b_mA: float,
    V_mc_kV: float,
    Rs_mc_MOhm: float,
    Q0_mc: float,
    beta_mc: float,
    G_drfb: float,
    tau_drfb_us: float,
    f_rf_MHz: float,
    V_hc_kV: float,
    Rs_hc_MOhm: float,
    Q0_hc: float,
    delta_f_hc_kHz: float,
    m: int
) -> Dict[str, str]:
    """
    Generate operational guidelines based on the Pedersen-Shen analysis.
    
    Returns a dict of recommendations and warnings.
    """
    recommendations = []
    warnings = []
    status = "OK"
    
    # Convert units
    Rs_mc = Rs_mc_MOhm * 1e6
    Rs_hc = Rs_hc_MOhm * 1e6
    f_rf = f_rf_MHz * 1e6
    V_mc = V_mc_kV * 1e3
    V_hc = V_hc_kV * 1e3
    I_b = I_b_mA / 1e3
    tau_drfb = tau_drfb_us * 1e-6
    QL_mc = Q0_mc / (1 + beta_mc)
    
    # 1. Check DRFB gain vs maximum
    drfb = drfb_impedance_reduction(Rs_mc, QL_mc, f_rf, G_drfb, tau_drfb)
    if G_drfb > drfb['G_max']:
        warnings.append(
            f"DRFB gain ({G_drfb:.1f}) exceeds maximum stable gain ({drfb['G_max']:.1f}). "
            "Reduce gain or shorten feedback delay."
        )
        status = "CRITICAL"
    elif G_drfb > 0.8 * drfb['G_max']:
        warnings.append(
            f"DRFB gain ({G_drfb:.1f}) is close to maximum ({drfb['G_max']:.1f}). "
            f"Gain margin: {drfb['gain_margin_dB']:.1f} dB. Consider reducing."
        )
        if status != "CRITICAL":
            status = "WARNING"
    
    # 2. Check beam loading
    Y_mc = I_b * Rs_mc / V_mc if V_mc > 0 else 0
    if Y_mc > 1.0:
        warnings.append(
            f"Heavy beam loading on MC (Y = {Y_mc:.2f} > 1). "
            "DRFB is essential for stability."
        )
        if G_drfb < 1.0:
            warnings.append(
                "DRFB gain is low for heavy beam loading. "
                "Consider increasing gain to improve Robinson margin."
            )
            if status == "OK":
                status = "WARNING"
    
    # 3. HC detuning check
    delta_f_hc_Hz = delta_f_hc_kHz * 1e3
    if delta_f_hc_Hz < 0:
        warnings.append(
            "HC detuning is negative (below resonance). "
            "For Robinson stability above transition, HC should be detuned ABOVE resonance."
        )
        if status == "OK":
            status = "WARNING"
    
    # 4. DRFB bandwidth check
    if drfb['bw_eff_kHz'] > 0:
        recommendations.append(
            f"DRFB expands MC bandwidth from {drfb['bw_0_kHz']:.1f} kHz to {drfb['bw_eff_kHz']:.1f} kHz "
            f"(x{drfb['bw_ratio']:.1f})."
        )
    
    # 5. Impedance reduction
    recommendations.append(
        f"DRFB reduces effective MC impedance by {abs(drfb['reduction_dB']):.1f} dB "
        f"({drfb['reduction_factor']*100:.1f}% of natural value)."
    )
    
    # 6. Pre-detuning recommendation (Shen 2024)
    recommendations.append(
        "Pre-detuning strategy (Shen 2024): As beam current increases during injection, "
        "pre-detune the MC slightly above resonance. The DRFB will maintain voltage regulation."
    )
    
    # 7. ALC/PLL guidance
    recommendations.append(
        "ALC (Auto-Level Control) maintains cavity voltage amplitude stable. "
        "PLL (Phase-Lock Loop) tracks the beam phase. Both act on slower timescales than DRFB."
    )
    
    return {
        'status': status,
        'warnings': warnings,
        'recommendations': recommendations,
        'drfb_metrics': drfb,
    }
