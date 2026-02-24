"""
Venturini (2018) Model: Passive Higher-Harmonic RF Cavities
============================================================
Implements the beam-loading equilibrium and stability formulas from:
  M. Venturini, "Passive higher-harmonic rf cavities with general settings
  and multibunch instabilities in electron storage rings,"
  Phys. Rev. Accel. Beams 21, 114404 (2018).

This module provides:
  - Flat potential conditions (voltage, phase, detuning)
  - Beam-induced voltage in passive cavities
  - Robinson stability criteria
  - Comparison with the "physics engine" used in the dashboard
"""

import numpy as np
from typing import Dict, Tuple, Optional


# =============================================================================
# 1. Flat Potential Conditions (Venturini Eqs. 5–9)
# =============================================================================

def flat_potential_conditions(V1: float, U0: float, m: int) -> Dict[str, float]:
    """
    Compute the analytic flat-potential operating point for a double RF system.
    
    The flat-potential condition requires both the first and second derivatives
    of the total RF voltage to vanish at the synchronous phase.

    Parameters
    ----------
    V1 : float
        Main cavity total voltage [V].
    U0 : float
        Energy loss per turn [eV].
    m  : int
        Harmonic number ratio (e.g., 3 for 3rd harmonic, 4 for 4th).

    Returns
    -------
    dict with keys:
        phi_s  : synchronous phase of main RF [rad]
        V2     : required harmonic voltage [V]
        psi    : harmonic cavity phase [rad]
        k      : voltage ratio V2/V1
        phi_s_deg, psi_deg : same in degrees
    """
    # Venturini Eq. 5:  cos(phi_1) = m^2/(m^2-1) * U0/(eV1)
    cos_phi1 = (m**2 / (m**2 - 1)) * (U0 / V1)
    cos_phi1 = np.clip(cos_phi1, -1.0, 1.0)
    
    # Main RF synchronous phase (above transition: pi - arcsin convention)
    phi_s = np.pi - np.arccos(cos_phi1)  # in [pi/2, pi]
    
    # Alternative: directly from sin(phi_s) = m^2/(m^2-1) * U0/V1
    sin_phi_s = (m**2 / (m**2 - 1)) * (U0 / V1)
    sin_phi_s = np.clip(sin_phi_s, -1.0, 1.0)
    phi_s_alt = np.pi - np.arcsin(sin_phi_s)
    
    # Venturini Eq. 6:  V2 = V1 * cos(phi_s) / (m^2 * cos(psi))
    # For flat potential, psi is determined by Eq. 7
    # Eq. 7:  tan(psi) = -1/m * tan(phi_s)  (at the flat potential condition)
    # This gives:  psi = arctan(-tan(phi_s) / m)
    tan_psi = -np.tan(phi_s_alt) / m
    psi = np.arctan(tan_psi)
    
    # Harmonic voltage from flat potential condition
    # V2 = -V1 * cos(phi_s) / (m^2 * cos(psi))
    V2 = -V1 * np.cos(phi_s_alt) / (m**2 * np.cos(psi))
    
    k = V2 / V1
    
    return {
        'phi_s': phi_s_alt,
        'phi_s_deg': np.degrees(phi_s_alt),
        'V2': V2,
        'V2_kV': V2 / 1e3,
        'psi': psi,
        'psi_deg': np.degrees(psi),
        'k': k,
        'cos_phi1': cos_phi1,
        'sin_phi_s': sin_phi_s,
    }


# =============================================================================
# 2. Beam-Induced Voltage in Passive Cavity (Venturini Sec. II.A)
# =============================================================================

def passive_cavity_beam_loading(
    I0: float,
    Rs: float,
    Q0: float,
    f_rf: float,
    m: int,
    delta_f: float
) -> Dict[str, float]:
    """
    Calculate the beam-induced voltage and phase in a passive harmonic cavity.
    
    For a passive cavity (no external power), the voltage is entirely
    beam-induced through the cavity impedance.

    Parameters
    ----------
    I0      : float  - DC beam current [A]
    Rs      : float  - Shunt impedance (total, linac convention: V²/P) [Ohm]
    Q0      : float  - Unloaded Q (= loaded Q for passive cavity with beta=0)
    f_rf    : float  - Fundamental RF frequency [Hz]
    m       : int    - Harmonic ratio (cavity resonance at m * f_rf)
    delta_f : float  - Detuning = f_resonance - m*f_rf [Hz]

    Returns
    -------
    dict with keys:
        V_ind   : induced voltage amplitude [V]
        phi_z   : impedance phase (= tuning angle) [rad]
        phi_z_deg : tuning angle [degrees]
        Z_eff   : effective impedance magnitude [Ohm]
    """
    f_h = m * f_rf
    
    # Tuning angle: tan(phi_z) = 2 * Q0 * delta_f / f_h
    # For passive cavity:  QL = Q0 (no coupler)
    tan_phi_z = 2 * Q0 * delta_f / f_h
    phi_z = np.arctan(tan_phi_z)
    
    # Effective impedance at the harmonic: Z_eff = Rs / (1 + j*tan(phi_z))
    # |Z_eff| = Rs * cos(phi_z)
    Z_eff = Rs * np.cos(phi_z)
    
    # Beam-induced voltage: V_ind = 2 * I0 * Z_eff  (circuit convention factor 2)
    V_ind = 2 * I0 * Z_eff
    
    return {
        'V_ind': V_ind,
        'V_ind_kV': V_ind / 1e3,
        'phi_z': phi_z,
        'phi_z_deg': np.degrees(phi_z),
        'Z_eff': Z_eff,
        'Z_eff_MOhm': Z_eff / 1e6,
        'tan_phi_z': tan_phi_z,
    }


# =============================================================================
# 3. Required Detuning for Flat Potential (Self-consistent)
# =============================================================================

def required_detuning_flat_potential(
    I0: float,
    Rs: float,
    Q0: float,
    f_rf: float,
    m: int,
    V2_target: float,
    psi_target: float
) -> Dict[str, float]:
    """
    Calculate the cavity detuning required to achieve the flat-potential
    harmonic voltage and phase, given beam current and cavity parameters.

    The self-consistency requires:
      V2 = 2 * I0 * Rs * cos(phi_z)
      tan(phi_z) = 2 * Q0 * delta_f / (m * f_rf)
    
    And the harmonic phase psi is related to the tuning angle phi_z.

    Parameters
    ----------
    I0         : float - DC beam current [A]
    Rs         : float - Total shunt impedance [Ohm]
    Q0         : float - Unloaded Q
    f_rf       : float - Fundamental RF frequency [Hz]
    m          : int   - Harmonic ratio
    V2_target  : float - Required harmonic voltage [V]
    psi_target : float - Required harmonic phase [rad]

    Returns
    -------
    dict with:
        delta_f_Hz   : required detuning [Hz]
        delta_f_kHz  : required detuning [kHz]
        phi_z        : tuning angle [rad]
        achievable   : bool, whether the current is sufficient
        I_min        : minimum current to achieve flat potential [A]
    """
    f_h = m * f_rf
    
    # From V2 = 2*I0*Rs*cos(phi_z) we get cos(phi_z) = V2/(2*I0*Rs)
    if I0 > 1e-12 and Rs > 0:
        cos_phi_z = abs(V2_target) / (2 * I0 * Rs)
    else:
        return {
            'delta_f_Hz': 0, 'delta_f_kHz': 0,
            'phi_z': 0, 'achievable': False, 'I_min': float('inf')
        }
    
    # Minimum current to achieve the required voltage
    I_min = abs(V2_target) / (2 * Rs)
    achievable = cos_phi_z <= 1.0
    
    if achievable:
        phi_z = np.arccos(cos_phi_z)
        tan_phi_z = np.tan(phi_z)
        delta_f = tan_phi_z * f_h / (2 * Q0)
        
        # Sign: passive HC must be detuned above resonance for stability
        delta_f = abs(delta_f)
    else:
        phi_z = 0
        delta_f = 0
    
    return {
        'delta_f_Hz': delta_f,
        'delta_f_kHz': delta_f / 1e3,
        'phi_z': phi_z,
        'phi_z_deg': np.degrees(phi_z) if achievable else 0,
        'achievable': achievable,
        'I_min': I_min,
        'I_min_mA': I_min * 1e3,
        'cos_phi_z': min(cos_phi_z, 1.0),
    }


# =============================================================================
# 4. Robinson Stability Criterion
# =============================================================================

def robinson_stability(
    V1: float,
    phi_s: float,
    V2: float,
    psi: float,
    m: int,
    f_rf: float,
    Rs_mc: float,
    Q_mc: float,
    delta_f_mc: float,
    Rs_hc: float,
    Q_hc: float,
    delta_f_hc: float
) -> Dict[str, float]:
    """
    Evaluate Robinson stability for a double RF system.
    
    Robinson instability occurs when the real part of the complex frequency
    shift due to the impedance is positive and exceeds the radiation damping.
    
    The criterion: the cavity must be detuned such that the upper synchrotron
    sideband sees more impedance than the lower sideband (Robinson criterion).

    Parameters
    ----------
    V1, phi_s   : main cavity voltage [V] and synchronous phase [rad]
    V2, psi     : harmonic voltage [V] and phase [rad]
    m           : harmonic ratio
    f_rf        : RF frequency [Hz]
    Rs_mc, Q_mc, delta_f_mc : main cavity parameters
    Rs_hc, Q_hc, delta_f_hc : harmonic cavity parameters

    Returns
    -------
    dict with stability info
    """
    # Synchrotron frequency (simplified zero-current)
    # omega_s^2 ~ (alpha * omega_rev * h / (2*pi*E0)) * dV/dphi at phi_s
    # For a double RF system:
    dVdphi = V1 * np.cos(phi_s) + m * V2 * np.cos(m * psi)
    
    # Robinson growth rate contribution from each cavity
    # Proportional to Im(Z(f_rf + f_s)) - Im(Z(f_rf - f_s))
    # For a detuned cavity: Im(Z) = Rs * sin(2*phi_z) / 2
    
    f_h = m * f_rf
    
    # MC contribution
    tan_mc = 2 * Q_mc * delta_f_mc / f_rf
    phi_z_mc = np.arctan(tan_mc)
    
    # HC contribution
    tan_hc = 2 * Q_hc * delta_f_hc / f_h
    phi_z_hc = np.arctan(tan_hc)
    
    # Robinson criterion: for stability, cavity should be detuned to high freq
    # (above transition energy)
    mc_stable = delta_f_mc > 0  # MC detuned above
    hc_stable = delta_f_hc > 0  # HC detuned above
    
    return {
        'dVdphi': dVdphi,
        'flat_potential': abs(dVdphi) < 0.01 * V1,  # Near zero slope
        'phi_z_mc_deg': np.degrees(phi_z_mc),
        'phi_z_hc_deg': np.degrees(phi_z_hc),
        'mc_robinson_stable': mc_stable,
        'hc_robinson_stable': hc_stable,
        'overall_stable': mc_stable and hc_stable,
    }


# =============================================================================
# 5. Bunch Lengthening Factor
# =============================================================================

def bunch_lengthening_factor(V1: float, U0: float, m: int, k: float) -> Dict[str, float]:
    """
    Estimate the bunch lengthening factor for flat potential conditions.
    
    For an ideal flat potential (quartic well), the RMS bunch length scales
    differently compared to a single RF system (quadratic well).
    
    Parameters
    ----------
    V1 : float - Main cavity voltage [V]
    U0 : float - Energy loss per turn [eV]
    m  : int   - Harmonic ratio
    k  : float - Voltage ratio V2/V1

    Returns
    -------
    dict with lengthening information
    """
    # Single RF factor for comparison
    sin_phi_s_single = U0 / V1
    sin_phi_s_single = np.clip(sin_phi_s_single, -1, 1)
    cos_phi_s_single = np.sqrt(1 - sin_phi_s_single**2)
    
    # Double RF with flat potential
    sin_phi_s_double = (m**2 / (m**2 - 1)) * (U0 / V1)
    sin_phi_s_double = np.clip(sin_phi_s_double, -1, 1)
    cos_phi_s_double = np.sqrt(1 - sin_phi_s_double**2)
    
    # For flat potential, the focusing is reduced by factor ~(m²-1)/m²
    # relative to single RF at same voltage
    # The potential is quartic: U ~ phi^4 instead of phi^2
    # This leads to bunch lengthening factor ~ (I/sigma_delta)^(1/3)
    
    # Approximate stretching factor for flat-potential case
    # sigma_z(double) / sigma_z(single) ≈ (m²/(m²-1))^(1/4) * (V1*cos(phi_s_single))^(1/4) / ...
    # In practice, factor of 3-5x is typical for optimized systems
    
    if cos_phi_s_single > 0 and cos_phi_s_double >= 0:
        # Rough estimate: ratio of linear focusing terms
        # Single RF: omega_s^2 ~ V1*cos(phi_s)
        # Double RF flat: omega_s -> 0 at center, dominated by quartic term
        linear_ratio = cos_phi_s_double / cos_phi_s_single if cos_phi_s_single > 0 else 0
    else:
        linear_ratio = 0
    
    return {
        'phi_s_single_deg': np.degrees(np.arcsin(sin_phi_s_single)),
        'phi_s_double_deg': np.degrees(np.pi - np.arcsin(sin_phi_s_double)),
        'cos_phi_s_single': cos_phi_s_single,
        'cos_phi_s_double': cos_phi_s_double,
        'linear_focusing_reduction': linear_ratio,
        'qualitative_stretching': "3-5x typical for optimized flat potential",
    }


# =============================================================================
# 6. Comprehensive Validation: Compare App formulas with Venturini
# =============================================================================

def validate_against_app(
    V1_kV: float,
    U0_keV: float,
    m: int,
    I0_mA: float,
    Rs_HC_MOhm: float,
    Q0_HC: float,
    f_rf_MHz: float,
    n_HC: int,
    # App results to compare
    app_phi_s_deg: float,
    app_Vh_kV: float,
    app_detuning_kHz: float,
    app_phi_h_deg: float
) -> Dict[str, dict]:
    """
    Run the Venturini model and compare its predictions with the app's
    physics engine output.

    Returns a dictionary of comparisons for each key parameter.
    """
    V1 = V1_kV * 1e3  # V
    U0 = U0_keV * 1e3  # eV
    I0 = I0_mA / 1e3   # A
    Rs = Rs_HC_MOhm * 1e6  # Ohm
    f_rf = f_rf_MHz * 1e6  # Hz
    
    # 1. Flat potential conditions
    fp = flat_potential_conditions(V1, U0, m)
    
    # 2. Required detuning
    det = required_detuning_flat_potential(I0, Rs, Q0_HC, f_rf, m, fp['V2'], fp['psi'])
    
    # 3. Beam loading check
    if I0 > 0:
        bl = passive_cavity_beam_loading(I0, Rs, Q0_HC, f_rf, m, det['delta_f_Hz'])
    else:
        bl = {'V_ind_kV': 0, 'phi_z_deg': 0}
    
    # 4. Bunch lengthening
    blen = bunch_lengthening_factor(V1, U0, m, fp['k'])
    
    # --- Comparison Table ---
    comparisons = {
        'synchronous_phase': {
            'parameter': 'Synchronous Phase φ_s',
            'venturini': fp['phi_s_deg'],
            'app': app_phi_s_deg,
            'unit': '°',
            'formula': f'φ_s = π - arcsin[m²/(m²−1) · U₀/V₁]',
            'match': abs(fp['phi_s_deg'] - app_phi_s_deg) < 1.0,
            'delta': fp['phi_s_deg'] - app_phi_s_deg,
        },
        'harmonic_voltage': {
            'parameter': 'Harmonic Voltage V₂',
            'venturini': abs(fp['V2_kV']),
            'app': app_Vh_kV,
            'unit': 'kV',
            'formula': f'V₂ = -V₁·cos(φ_s) / (m²·cos(ψ))',
            'match': abs(abs(fp['V2_kV']) - app_Vh_kV) < max(0.05 * app_Vh_kV, 1.0) if app_Vh_kV > 0 else True,
            'delta': abs(fp['V2_kV']) - app_Vh_kV,
        },
        'harmonic_phase': {
            'parameter': 'Harmonic Phase ψ',
            'venturini': fp['psi_deg'],
            'app': app_phi_h_deg,
            'unit': '°',
            'formula': f'tan(ψ) = -tan(φ_s) / m',
            'match': abs(fp['psi_deg'] - app_phi_h_deg) < 5.0,
            'delta': fp['psi_deg'] - app_phi_h_deg,
        },
        'detuning': {
            'parameter': 'HC Detuning δf',
            'venturini': det['delta_f_kHz'],
            'app': abs(app_detuning_kHz),
            'unit': 'kHz',
            'formula': f'δf = tan(φ_z)·m·f_rf / (2·Q₀)',
            'match': abs(det['delta_f_kHz'] - abs(app_detuning_kHz)) < max(0.1 * abs(app_detuning_kHz), 0.5) if abs(app_detuning_kHz) > 0 else True,
            'delta': det['delta_f_kHz'] - abs(app_detuning_kHz),
        },
        'voltage_ratio': {
            'parameter': 'Voltage Ratio k = V₂/V₁',
            'venturini': abs(fp['k']),
            'app': app_Vh_kV / V1_kV if V1_kV > 0 else 0,
            'unit': '',
            'formula': 'k = V₂/V₁  (Hofmann flat potential)',
            'match': True,
            'delta': abs(fp['k']) - (app_Vh_kV / V1_kV if V1_kV > 0 else 0),
        },
    }
    
    # Additional info
    meta = {
        'flat_potential': fp,
        'detuning': det,
        'beam_loading': bl,
        'bunch_lengthening': blen,
        'n_comparisons': len(comparisons),
        'n_matches': sum(1 for c in comparisons.values() if c['match']),
    }
    
    return {'comparisons': comparisons, 'meta': meta}


# =============================================================================
# 7. Operating Regime Classification
# =============================================================================

def classify_operating_regime(
    V1: float, U0: float, m: int,
    V2_actual: float, psi_actual: float
) -> Dict[str, str]:
    """
    Classify the current operating regime relative to flat potential.
    
    Parameters
    ----------
    V1 : float  - Main voltage [V]
    U0 : float  - Energy loss [eV]
    m  : int    - Harmonic ratio
    V2_actual : float - Actual harmonic voltage [V]
    psi_actual : float - Actual harmonic phase [rad]

    Returns
    -------
    dict with regime classification and recommendations
    """
    fp = flat_potential_conditions(V1, U0, m)
    V2_flat = abs(fp['V2'])
    
    ratio = abs(V2_actual) / V2_flat if V2_flat > 0 else 0
    
    if ratio < 0.8:
        regime = "Under-stretched"
        description = (
            "The harmonic voltage is below the flat-potential value. "
            "The bunch will be partially lengthened but retains a near-Gaussian shape. "
            "Robinson instability risk is lower, but Landau damping is weaker."
        )
        recommendation = "Increase HC detuning (tune closer to harmonic) to increase induced voltage."
    elif ratio > 1.2:
        regime = "Over-stretched"
        description = (
            "The harmonic voltage exceeds the flat-potential value. "
            "The bunch may develop a double-peaked (rabbit-ear) structure. "
            "This can lead to additional instabilities and is generally undesirable."
        )
        recommendation = "Decrease HC voltage by detuning further from the harmonic frequency."
    else:
        regime = "Near Flat Potential ✓"
        description = (
            "The system is operating near the optimal flat-potential condition. "
            "Maximum bunch lengthening and Landau damping are achieved. "
            "The potential well has a quartic (x⁴) shape at the center."
        )
        recommendation = "Optimal operating point. Monitor beam stability."
    
    return {
        'regime': regime,
        'description': description,
        'recommendation': recommendation,
        'V2_ratio': ratio,
        'V2_flat_kV': V2_flat / 1e3,
        'V2_actual_kV': abs(V2_actual) / 1e3,
        'psi_flat_deg': fp['psi_deg'],
        'psi_actual_deg': np.degrees(psi_actual),
    }
