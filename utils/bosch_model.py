"""
Bosch Model: Higher-Harmonic Cavity Instability Analysis
=========================================================
Implements the coupled-bunch instability and Robinson stability analysis from:

  [1] R. A. Bosch and C. S. Hsue,
      "Suppression of longitudinal coupled-bunch instabilities by a passive
      higher harmonic cavity,"
      Part. Accel. 42, 81–99 (1993).

  [2] R. A. Bosch,
      "Instability analysis of an active higher-harmonic cavity,"
      Proc. 1997 Particle Accelerator Conference (PAC97), pp. 862–864.

  [3] R. A. Bosch,
      "Instabilities driven by higher-order modes in an RF system with a
      passive higher harmonic cavity."

The Bosch model treats the double RF system (fundamental + harmonic cavity)
in the time domain, using the Sands notation for RF cavities. Key features:

  1. Robinson instability in a two-cavity system
     - Growth/damping rate depends on tuning angles at synchrotron sidebands
     - Both cavities contribute (fundamental & harmonic)

  2. Equilibrium phase stability
     - For passive HHC: constraint on achievable operating points
     - For active HHC: equilibrium phase instability boundary

  3. Coupled-bunch instability growth rate
     - Resonant dipole interaction with parasitic HOM impedance
     - Includes form factors for finite bunch length (Gaussian bunches)

  4. Synchrotron frequency spread & Landau damping
     - Quadratic well: σω_s from potential nonlinearity
     - Quartic well (flat potential): larger spread, stronger damping
     - Threshold: Landau damping overcomes instability when |Δω_CB| < 0.78 σω_s

  5. Active HHC algorithm (PAC97):
     - Compensated condition operation
     - Quartic potential maximizing bunch length
     - Robinson & equilibrium phase instability mapping vs (I, β₂)

This module provides functions for both passive and active harmonic cavities.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.special import gamma as gamma_func


# =============================================================================
# 1. Equilibrium Phases (Bosch 1993 Eqs. 11–13; PAC97 Eqs. 1–6)
# =============================================================================

def equilibrium_phases_active(
    V1: float,
    Vs: float,
    nu: int
) -> Dict[str, float]:
    """
    Calculate equilibrium phases for an active Landau cavity operating
    at maximum bunch length (quartic potential, b=c=0).

    From PAC97 Eqs. (4)–(6):
        ψ₁ = arccos(Vs / ((1 − 1/ν²) V₁))
        ψ₂ = arctan(ν tan ψ₁) − π
        V₂ = −V₁ sin ψ₁ / (ν sin ψ₂)

    Parameters
    ----------
    V1  : Peak voltage in fundamental cavity [V]
    Vs  : Synchronous voltage (= energy loss per turn) [V]
    nu  : Harmonic number of Cavity 2

    Returns
    -------
    dict with equilibrium phases and harmonic voltage
    """
    ratio = Vs / ((1 - 1.0 / nu**2) * V1)

    if abs(ratio) > 1:
        return {
            'psi1_rad': None,
            'psi2_rad': None,
            'V2': None,
            'valid': False,
            'error': 'No equilibrium phase exists (V1 too low or Vs too high)'
        }

    psi1 = np.arccos(ratio)
    psi2 = np.arctan(nu * np.tan(psi1)) - np.pi
    V2 = -V1 * np.sin(psi1) / (nu * np.sin(psi2))

    return {
        'psi1_rad': psi1,
        'psi1_deg': np.degrees(psi1),
        'psi2_rad': psi2,
        'psi2_deg': np.degrees(psi2),
        'V2': V2,
        'V2_kV': V2 / 1e3,
        'valid': True,
        'error': None
    }


def equilibrium_phases_passive(
    V1: float,
    Vs: float,
    I_b: float,
    R2: float,
    Q2: float,
    f_rf: float,
    nu: int,
    phi_z2: float
) -> Dict[str, float]:
    """
    Calculate equilibrium for a passive harmonic cavity.

    For a passive cavity (no generator), the beam-induced voltage is:
        V₂ = 2·I·F₂·R₂·cos(φ_z2)           [Bosch 1993, Eq. 16]

    The equilibrium phase in Cavity 2 is related to the tuning angle:
        ψ₂ = φ_z2 − π/2                      (for passive operation)

    Parameters
    ----------
    V1     : Peak voltage in fundamental cavity [V]
    Vs     : Synchronous voltage [V]
    I_b    : Average beam current [A]
    R2     : Loaded impedance at resonance of HC [Ohm]
    Q2     : Loaded quality factor of HC
    f_rf   : Fundamental RF frequency [Hz]
    nu     : Harmonic number ratio
    phi_z2 : Tuning angle of HC [rad]

    Returns
    -------
    dict with passive cavity equilibrium parameters
    """
    # Form factor for short bunches (F2 ≈ 1)
    F2 = 1.0

    # Beam-induced voltage in passive cavity
    V2 = 2 * I_b * F2 * R2 * np.cos(phi_z2)

    # Equilibrium phase in fundamental cavity
    # Vs = V1·cos(ψ1) + V2·cos(ψ2)
    # For passive HC: ψ2 is related to φ_z2
    psi2 = phi_z2 - np.pi / 2

    # Solve for ψ1
    cos_psi1 = (Vs - V2 * np.cos(psi2)) / V1 if V1 > 0 else 0
    cos_psi1 = np.clip(cos_psi1, -1, 1)
    psi1 = np.arccos(cos_psi1)

    return {
        'psi1_rad': psi1,
        'psi1_deg': np.degrees(psi1),
        'psi2_rad': psi2,
        'psi2_deg': np.degrees(psi2),
        'V2': V2,
        'V2_kV': V2 / 1e3,
        'phi_z2_deg': np.degrees(phi_z2),
        'valid': True
    }


# =============================================================================
# 2. Tuning Angles and Compensated Condition (Bosch 1993 Sec. 2)
# =============================================================================

def tuning_angles(
    I_b: float,
    V1: float,
    R1: float,
    Q1: float,
    psi1: float,
    V2: float,
    R2: float,
    Q2: float,
    psi2: float,
    F1: float = 1.0,
    F2: float = 1.0
) -> Dict[str, float]:
    """
    Calculate tuning angles for compensated condition operation.

    Compensated condition (Sands): generator current is in phase with voltage.
    This gives (PAC97 Eqs. 10–11):
        tan(φ₁) = 2·F₁·I·R₁·sin(ψ₁) / V₁
        tan(φ₂) = 2·F₂·I·R₂·sin(ψ₂) / V₂

    Parameters
    ----------
    I_b   : Average beam current [A]
    V1, V2: Peak voltages [V]
    R1, R2: Loaded impedance at resonance [Ohm]
    Q1, Q2: Loaded quality factors
    psi1, psi2: Equilibrium phase angles [rad]
    F1, F2: Form factors (= 1 for short bunches)
    """
    if V1 > 0:
        tan_phi1 = 2 * F1 * I_b * R1 * np.sin(psi1) / V1
        phi1 = np.arctan(tan_phi1)
    else:
        phi1 = 0.0

    if V2 > 0:
        tan_phi2 = 2 * F2 * I_b * R2 * np.sin(psi2) / V2
        phi2 = np.arctan(tan_phi2)
    else:
        phi2 = 0.0

    return {
        'phi1_rad': phi1,
        'phi1_deg': np.degrees(phi1),
        'phi2_rad': phi2,
        'phi2_deg': np.degrees(phi2),
    }


# =============================================================================
# 3. Form Factors for Finite Bunch Length (Bosch 1993 Eq. 9)
# =============================================================================

def form_factor_gaussian(
    sigma_t: float,
    f: float
) -> float:
    """
    Gaussian bunch form factor at frequency f.

    F(f) = exp(−(2πf)² σ_t² / 2)     [Bosch 1993]

    Parameters
    ----------
    sigma_t : RMS bunch length [s]
    f       : Frequency [Hz]

    Returns
    -------
    Form factor (0 to 1)
    """
    omega = 2 * np.pi * f
    return np.exp(-omega**2 * sigma_t**2 / 2)


def form_factors(
    sigma_t: float,
    f_rf: float,
    nu: int
) -> Dict[str, float]:
    """
    Calculate form factors for both cavities.

    F₁ = exp(−ω_g²·σ_t²/2)     at fundamental frequency
    F₂ = exp(−(ν·ω_g)²·σ_t²/2) at harmonic frequency
    """
    F1 = form_factor_gaussian(sigma_t, f_rf)
    F2 = form_factor_gaussian(sigma_t, nu * f_rf)

    return {
        'F1': F1,
        'F2': F2,
        'sigma_t_ps': sigma_t * 1e12,
    }


# =============================================================================
# 4. Robinson Instability (Bosch 1993 Eqs. 8–10; PAC97 Eqs. 19–21)
# =============================================================================

def robinson_two_cavity(
    alpha: float,
    omega_g: float,
    T0: float,
    E: float,
    I_b: float,
    V1: float,
    R1: float,
    Q1: float,
    psi1: float,
    V2: float,
    R2: float,
    Q2: float,
    psi2: float,
    nu: int,
    tau_rad: float,
    F1: float = 1.0,
    F2: float = 1.0
) -> Dict[str, float]:
    """
    Robinson instability analysis for a two-cavity system.

    Implements Bosch (1993) Eqs. (8)–(10) and PAC97 Eqs. (19)–(21):

    Robinson frequency² (Eq. 9/20):
        Ω² = (eα·ω_g)/(T₀·E) × { F₁V₁sin(ψ₁)
              − R₁F₁²I/2 × [sin(2φ₁₋) + sin(2φ₁₊)]
              + ν·F₂V₂sin(ψ₂)
              − ν·R₂F₂²I/2 × [sin(2φ₂₋) + sin(2φ₂₊)] }

    Robinson damping rate (Eq. 10/21):
        α_R = (4αeI)/(E·T₀) × [F₁²R₁Q₁ tan(φ₁) cos²(φ₁₊) cos²(φ₁₋)
                               + F₂²R₂Q₂ tan(φ₂) cos²(φ₂₊) cos²(φ₂₋)]

    Equilibrium phase stability (Eq. 19):
        F₁V₁sin(ψ₁) + ν·F₂V₂sin(ψ₂) > R₁F₁²I sin(2φ₁) + ν·R₂F₂²I sin(2φ₂)

    Parameters
    ----------
    alpha    : Momentum compaction factor
    omega_g  : Generator angular frequency [rad/s]
    T0       : Revolution period [s]
    E        : Electron energy [eV]
    I_b      : Average beam current [A]
    V1, V2   : Peak voltages [V]
    R1, R2   : Loaded impedances at resonance [Ohm]
    Q1, Q2   : Loaded quality factors
    psi1,psi2: Equilibrium phases [rad]
    nu       : Harmonic number ratio
    tau_rad  : Longitudinal radiation damping time [s]
    F1, F2   : Form factors

    Returns
    -------
    dict with Robinson stability analysis
    """
    e = 1.0  # electron charge magnitude (I already in Amperes, E in eV)

    # Tuning angles (compensated condition)
    ta = tuning_angles(I_b, V1, R1, Q1, psi1, V2, R2, Q2, psi2, F1, F2)
    phi1 = ta['phi1_rad']
    phi2 = ta['phi2_rad']

    # Robinson frequency (iterate: start with Ω=0 for sideband angles)
    # First pass: φ₁± = φ₁, φ₂± = φ₂ (zero-current approximation)
    Omega_sq_0 = (e * alpha * omega_g) / (T0 * E) * (
        F1 * V1 * np.sin(psi1) + nu * F2 * V2 * np.sin(psi2)
    )

    if Omega_sq_0 > 0:
        Omega_0 = np.sqrt(Omega_sq_0)
    else:
        Omega_0 = 0.0

    # Second pass: include beam current terms
    # Sideband tuning angles: tan(φ₁±) = 2Q₁(ω_g ± Ω − ω₁)/ω₁
    # In compensated condition: ω₁ is set so that tan(φ₁) = 2F₁IR₁sin(ψ₁)/V₁
    # We can compute φ₁± from the relation:
    #   tan(φ₁±) = tan(φ₁) ± 2Q₁Ω/ω_g × (1 + tan²(φ₁)) approximately

    # For this implementation, we use the simplified approach:
    # φ₁± ≈ φ₁ ± Q₁Ω/(ω_g/2) ... (small Ω/ω_half approximation)
    omega_half_1 = omega_g / (2 * Q1) if Q1 > 0 else 1e10
    omega_half_2 = (nu * omega_g) / (2 * Q2) if Q2 > 0 else 1e10

    delta_phi1 = np.arctan(Omega_0 / omega_half_1) if omega_half_1 > 0 else 0
    delta_phi2 = np.arctan(Omega_0 / omega_half_2) if omega_half_2 > 0 else 0

    phi1_plus = phi1 + delta_phi1
    phi1_minus = phi1 - delta_phi1
    phi2_plus = phi2 + delta_phi2
    phi2_minus = phi2 - delta_phi2

    # Robinson frequency with beam loading (Eq. 9/20)
    Omega_sq = (e * alpha * omega_g) / (T0 * E) * (
        F1 * V1 * np.sin(psi1)
        - R1 * F1**2 * I_b / 2 * (np.sin(2 * phi1_minus) + np.sin(2 * phi1_plus))
        + nu * F2 * V2 * np.sin(psi2)
        - nu * R2 * F2**2 * I_b / 2 * (np.sin(2 * phi2_minus) + np.sin(2 * phi2_plus))
    )

    # Robinson damping rate (Eq. 10/21)
    alpha_R_no_rad = (4 * alpha * e * I_b) / (E * T0) * (
        F1**2 * R1 * Q1 * np.tan(phi1) * np.cos(phi1_plus)**2 * np.cos(phi1_minus)**2
        + F2**2 * R2 * Q2 * np.tan(phi2) * np.cos(phi2_plus)**2 * np.cos(phi2_minus)**2
    )

    # Include radiation damping
    alpha_R = alpha_R_no_rad + 1.0 / tau_rad if tau_rad > 0 else alpha_R_no_rad

    # Robinson frequency
    if Omega_sq > 0:
        Omega_robinson = np.sqrt(Omega_sq)
    else:
        Omega_robinson = 0.0

    # Equilibrium phase stability (PAC97 Eq. 19)
    lhs_eq19 = F1 * V1 * np.sin(psi1) + nu * F2 * V2 * np.sin(psi2)
    rhs_eq19 = R1 * F1**2 * I_b * np.sin(2 * phi1) + nu * R2 * F2**2 * I_b * np.sin(2 * phi2)
    eq_phase_stable = lhs_eq19 > rhs_eq19

    # Robinson stable if damping rate is positive
    robinson_stable = alpha_R > 0

    # Combined stability
    overall_stable = eq_phase_stable and robinson_stable

    return {
        'Omega_robinson_Hz': Omega_robinson / (2 * np.pi),
        'Omega_robinson_kHz': Omega_robinson / (2 * np.pi * 1e3),
        'Omega_sq': Omega_sq,
        'alpha_R': alpha_R,
        'alpha_R_no_rad': alpha_R_no_rad,
        'alpha_R_inv_s': 1.0 / alpha_R if alpha_R != 0 else float('inf'),
        'robinson_stable': robinson_stable,
        'eq_phase_stable': eq_phase_stable,
        'overall_stable': overall_stable,
        'phi1_deg': np.degrees(phi1),
        'phi2_deg': np.degrees(phi2),
        'lhs_eq19': lhs_eq19,
        'rhs_eq19': rhs_eq19,
    }


# =============================================================================
# 5. Synchrotron Frequency & Frequency Spread (Bosch 1993 Sec. 3)
# =============================================================================

def synchrotron_frequency_spread(
    V1: float,
    psi1: float,
    V2: float,
    psi2: float,
    nu: int,
    alpha: float,
    omega_g: float,
    T0: float,
    E: float,
    sigma_t: float,
    F1: float = 1.0,
    F2: float = 1.0
) -> Dict[str, float]:
    """
    Calculate synchrotron frequency and its spread in a double RF system.

    From Bosch (1993) Eqs. (20)–(25):

    The synchrotron potential U(T) = aT² + bT³ + cT⁴ + ...

    Coefficients:
        a = ω_s²/2 = (αeω_g)/(2ET₀) × (V₁sinψ₁ + νV₂sinψ₂)
        b = −(αeω_g²)/(6ET₀) × (V₁cosψ₁ + ν²V₂cosψ₂)
        c = −(αeω_g³)/(24ET₀) × (V₁sinψ₁ + ν³V₂sinψ₂)

    Synchrotron frequency spread (Eq. 24):
        σω_s = ω_s(ω_g σ_t)² √(8c²/a + 4b²/a²)

    For flat potential (a→0, quartic well):
        σω_s → 1.72 × ω_s(σ_t)   [Eq. 15]
        where ω_s(σ_t) = 1.17 (U₀c)^(1/4)

    Parameters
    ----------
    V1, V2   : Peak voltages [V]
    psi1,psi2: Equilibrium phases [rad]
    nu       : Harmonic number ratio
    alpha    : Momentum compaction
    omega_g  : Generator angular frequency [rad/s]
    T0       : Revolution period [s]
    E        : Electron energy [eV]
    sigma_t  : RMS bunch length [s]
    F1, F2   : Form factors

    Returns
    -------
    dict with synchrotron frequency and spread information
    """
    e = 1.0  # conventions

    # Potential coefficients (Eqs. 20-22)
    a = (alpha * e * omega_g) / (2 * E * T0) * (
        V1 * np.sin(psi1) + nu * V2 * np.sin(psi2)
    )

    b = -(alpha * e * omega_g**2) / (6 * E * T0) * (
        V1 * np.cos(psi1) + nu**2 * V2 * np.cos(psi2)
    )

    c = -(alpha * e * omega_g**3) / (24 * E * T0) * (
        V1 * np.sin(psi1) + nu**3 * V2 * np.sin(psi2)
    )

    # Synchrotron frequency
    if a > 0:
        omega_s = np.sqrt(2 * a)
        f_s = omega_s / (2 * np.pi)
        is_quadratic = True
    else:
        omega_s = 0.0
        f_s = 0.0
        is_quadratic = False

    # Frequency spread for quadratic potential (Eq. 24)
    if is_quadratic and a > 0:
        # σω_s = ω_s · (ω_g·σ_t)² · √(8c²/a + 4b²/a²)    (simplified)
        x = omega_g * sigma_t
        if a > 0:
            spread_term = np.sqrt(abs(8 * c**2 / a + 4 * b**2 / a**2))
            sigma_omega_s = omega_s * x**2 * spread_term
        else:
            sigma_omega_s = 0.0

        # Spread as fraction
        relative_spread = sigma_omega_s / omega_s if omega_s > 0 else 0.0
    else:
        sigma_omega_s = 0.0
        relative_spread = 0.0

    # Quartic potential case (flat potential, a ≈ 0)
    # From PAC97: σ_t = 0.69 × (U₀/c)^(1/4) where U₀ = (α²/2)(σ_E/E)²
    # And ω_s(σ_t) = 1.17 × (U₀·c)^(1/4)
    # Δω_s = 1.72 × ω_s(σ_t)
    if abs(c) > 0:
        # Quartic synchrotron frequency at amplitude σ_t
        Uo_c_product = abs(c)  # U₀·|c| for the quartic regime
        omega_s_quartic = 1.17 * abs(Uo_c_product)**0.25
        delta_omega_s_quartic = 1.72 * omega_s_quartic
    else:
        omega_s_quartic = 0.0
        delta_omega_s_quartic = 0.0

    return {
        'a': a,
        'b': b,
        'c': c,
        'omega_s': omega_s,
        'f_s_Hz': f_s,
        'f_s_kHz': f_s / 1e3,
        'sigma_omega_s': sigma_omega_s,
        'sigma_f_s_Hz': sigma_omega_s / (2 * np.pi),
        'relative_spread': relative_spread,
        'is_quadratic': is_quadratic,
        'omega_s_quartic': omega_s_quartic,
        'delta_omega_s_quartic': delta_omega_s_quartic,
    }


# =============================================================================
# 6. Coupled-Bunch Instability Growth Rate (Bosch 1993 Eq. 26; PAC97 Eqs. 12-18)
# =============================================================================

def coupled_bunch_growth_rate(
    I_b: float,
    alpha: float,
    E: float,
    T0: float,
    omega_s: float,
    Z_cb: float,
    omega_cb: float,
    sigma_t: float,
    tau_rad: float,
    sigma_omega_s: float = 0.0,
    is_quartic: bool = False,
    c_coeff: float = 0.0,
    Uo: float = 0.0
) -> Dict[str, float]:
    """
    Calculate coupled-bunch instability growth rate.

    For a quadratic well (Bosch 1993 Eq. 26):
        Δω_CB = (eI·α·ω_CB·Z(ω_CB)·F²_ωCB) / (2·E·T₀·ω_s)

    For a quartic well (PAC97 Eqs. 12-13):
        Ω²_CB = i·(δΩ₀)²
        (δΩ₀)² = (eI·α)/(E·T₀) × F²_ωCB × ω_CB·Z(ω_CB)

    Landau damping threshold:
        |Δω_CB| < 0.78·σω_s     (quadratic, Bosch 1993)
        |Ω_CB| < 0.6·Δω_s       (quartic, PAC97 Eq. 17)

    Parameters
    ----------
    I_b         : Average beam current [A]
    alpha       : Momentum compaction
    E           : Electron energy [eV]
    T0          : Revolution period [s]
    omega_s     : Synchrotron angular frequency [rad/s]
    Z_cb        : Parasitic impedance at coupled-bunch frequency [Ohm]
    omega_cb    : Coupled-bunch angular frequency [rad/s]
    sigma_t     : RMS bunch length [s]
    tau_rad     : Radiation damping time [s]
    sigma_omega_s: Synchrotron frequency spread [rad/s]
    is_quartic  : Whether the potential is quartic (flat potential)
    c_coeff     : Quartic potential coefficient
    Uo          : Filling height for quartic potential

    Returns
    -------
    dict with growth rate analysis
    """
    e = 1.0  # conventions

    # Form factor at coupled-bunch frequency
    F_cb = form_factor_gaussian(sigma_t, omega_cb / (2 * np.pi))

    if not is_quartic:
        # Quadratic well: resonant dipole interaction (Eq. 26)
        if omega_s > 0:
            delta_omega_cb = (e * I_b * alpha * omega_cb * Z_cb * F_cb**2) / (
                2 * E * T0 * omega_s
            )
        else:
            delta_omega_cb = 0.0

        # Growth rate (imaginary part of frequency shift)
        growth_rate = abs(delta_omega_cb) - 1.0 / tau_rad if tau_rad > 0 else abs(delta_omega_cb)

        # Landau damping criterion (Bosch 1993)
        # Instability occurs when |Δω_CB| > 0.78·σω_s AND > 1/τ_rad
        landau_threshold = 0.78 * sigma_omega_s
        landau_stable = abs(delta_omega_cb) < landau_threshold
        radiation_stable = abs(delta_omega_cb) < (1.0 / tau_rad if tau_rad > 0 else 0)
        stable = landau_stable or radiation_stable

        return {
            'delta_omega_cb': delta_omega_cb,
            'delta_f_cb_Hz': delta_omega_cb / (2 * np.pi),
            'growth_rate': growth_rate,
            'growth_rate_inv_s': 1.0 / abs(growth_rate) if growth_rate != 0 else float('inf'),
            'F_cb': F_cb,
            'landau_threshold': landau_threshold,
            'landau_threshold_Hz': landau_threshold / (2 * np.pi),
            'landau_stable': landau_stable,
            'radiation_stable': radiation_stable,
            'stable': stable,
            'is_quartic': False,
        }
    else:
        # Quartic well (PAC97 Eqs. 12-18)
        delta_Omega_0_sq = (e * I_b * alpha) / (E * T0) * F_cb**2 * omega_cb * Z_cb

        if delta_Omega_0_sq > 0:
            delta_Omega_0 = np.sqrt(delta_Omega_0_sq)
        else:
            delta_Omega_0 = 0.0

        # Ω_CB = (δΩ₀/√2) + i(δΩ₀/√2 − 1/τ_rad)  [PAC97 Eq. 18]
        Omega_cb_real = delta_Omega_0 / np.sqrt(2)
        Omega_cb_imag = delta_Omega_0 / np.sqrt(2) - (1.0 / tau_rad if tau_rad > 0 else 0)

        growth_rate = Omega_cb_imag

        # Quartic Landau damping: ω_s(σ_t) = 1.17 × (U₀c)^(1/4)
        # Δω_s = 1.72 × ω_s(σ_t)
        # Landau damping overcomes instability when (PAC97 Eq. 17):
        #   |Ω_CB| > 0.6 × Δω_s
        if abs(c_coeff) > 0 and Uo > 0:
            omega_s_at_sigma = 1.17 * (Uo * abs(c_coeff))**0.25
            delta_omega_s = 1.72 * omega_s_at_sigma
        else:
            delta_omega_s = sigma_omega_s  # fallback

        landau_threshold = 0.6 * delta_omega_s
        Omega_cb_mag = np.sqrt(Omega_cb_real**2 + Omega_cb_imag**2)
        landau_stable = Omega_cb_mag < landau_threshold

        # Radiation damping check
        radiation_stable = delta_Omega_0 / np.sqrt(2) < (1.0 / tau_rad if tau_rad > 0 else 0)
        stable = landau_stable or radiation_stable

        return {
            'delta_Omega_0': delta_Omega_0,
            'Omega_cb_real': Omega_cb_real,
            'Omega_cb_imag': Omega_cb_imag,
            'growth_rate': growth_rate,
            'growth_rate_inv_s': 1.0 / abs(growth_rate) if growth_rate != 0 else float('inf'),
            'F_cb': F_cb,
            'landau_threshold': landau_threshold,
            'landau_threshold_Hz': landau_threshold / (2 * np.pi),
            'landau_stable': landau_stable,
            'radiation_stable': radiation_stable,
            'stable': stable,
            'is_quartic': True,
        }


# =============================================================================
# 7. Bunch Length Calculation (Bosch 1993 Sec. 3; PAC97 Eq. 8)
# =============================================================================

def bunch_length_double_rf(
    V1: float,
    psi1: float,
    V2: float,
    psi2: float,
    nu: int,
    alpha: float,
    omega_g: float,
    E: float,
    sigma_E_over_E: float,
    T0: float,
    natural_sigma_t: float
) -> Dict[str, float]:
    """
    Calculate the bunch length in a double RF system.

    For a quadratic well:
        σ_t = α·(σ_E/E) / ω_s

    For a quartic well (flat potential, PAC97 Eq. 8):
        σ_t = 0.69 × (U₀/c)^(1/4)
        where U₀ = (α/2)·(σ_E/E)²

    Parameters
    ----------
    V1, V2      : Peak voltages [V]
    psi1, psi2  : Equilibrium phases [rad]
    nu          : Harmonic number ratio
    alpha       : Momentum compaction
    omega_g     : Generator angular frequency [rad/s]
    E           : Electron energy [eV]
    sigma_E_over_E : Relative energy spread
    T0          : Revolution period [s]
    natural_sigma_t: Natural bunch length without HC [s]

    Returns
    -------
    dict with bunch lengths
    """
    e = 1.0

    # Potential coefficients
    a = (alpha * e * omega_g) / (2 * E * T0) * (
        V1 * np.sin(psi1) + nu * V2 * np.sin(psi2)
    )
    c = -(alpha * e * omega_g**3) / (24 * E * T0) * (
        V1 * np.sin(psi1) + nu**3 * V2 * np.sin(psi2)
    )

    # Filling height
    Uo = alpha**2 / 2 * sigma_E_over_E**2

    if a > 0:
        # Quadratic well
        omega_s = np.sqrt(2 * a)
        sigma_t = alpha * sigma_E_over_E / omega_s if omega_s > 0 else natural_sigma_t
        is_quartic = False
    else:
        # Quartic / flat potential
        if abs(c) > 0:
            sigma_t = 0.69 * (Uo / abs(c))**0.25
        else:
            sigma_t = natural_sigma_t
        omega_s = 0.0
        is_quartic = True

    lengthening = sigma_t / natural_sigma_t if natural_sigma_t > 0 else 1.0

    return {
        'sigma_t_s': sigma_t,
        'sigma_t_ps': sigma_t * 1e12,
        'sigma_z_mm': sigma_t * 3e8 * 1e3,  # c·σ_t in mm
        'natural_sigma_t_ps': natural_sigma_t * 1e12,
        'lengthening_factor': lengthening,
        'omega_s': omega_s,
        'f_s_kHz': omega_s / (2 * np.pi * 1e3),
        'is_quartic': is_quartic,
        'Uo': Uo,
        'a': a,
        'c': c,
    }


# =============================================================================
# 8. Complete Bosch Analysis (Combined Algorithm)
# =============================================================================

def bosch_analysis(
    # Ring parameters
    E_MeV: float,
    alpha_c: float,
    T0_s: float,
    sigma_E_over_E: float,
    tau_rad_s: float,
    # Fundamental cavity
    V1_kV: float,
    R1_MOhm: float,
    Q01: float,
    beta1: float,
    f_rf_MHz: float,
    # Harmonic cavity
    R2_MOhm: float,
    Q02: float,
    beta2: float,
    nu: int,
    # Beam
    I_mA: float,
    Vs_kV: float,
    # Mode: 'active' or 'passive'
    mode: str = 'active',
    # For passive: HC tuning angle [deg]
    phi_z2_passive_deg: float = -30.0,
    # Parasitic HOM for coupled-bunch
    Z_hom_kOhm: float = 10.0,
    f_hom_GHz: float = 1.0,
    # Natural bunch length
    natural_sigma_t_ps: float = 30.0
) -> Dict[str, dict]:
    """
    Complete Bosch instability analysis for a double RF system.

    Combines all the analysis steps from Bosch (1993) and PAC97:
    1. Equilibrium phases
    2. Form factors and bunch length
    3. Robinson stability
    4. Coupled-bunch instability
    5. Landau damping assessment

    Parameters
    ----------
    E_MeV           : Electron energy [MeV]
    alpha_c         : Momentum compaction factor
    T0_s            : Revolution period [s]
    sigma_E_over_E  : Relative energy spread
    tau_rad_s       : Longitudinal radiation damping time [s]
    V1_kV           : Fundamental cavity peak voltage [kV]
    R1_MOhm         : Fundamental cavity loaded impedance [MOhm]
    Q01             : Fundamental cavity unloaded Q
    beta1           : Fundamental cavity coupling coefficient
    f_rf_MHz        : RF frequency [MHz]
    R2_MOhm         : Harmonic cavity loaded impedance [MOhm]
    Q02             : Harmonic cavity unloaded Q
    beta2           : Harmonic cavity coupling coefficient (0 for passive)
    nu              : Harmonic number ratio
    I_mA            : Average beam current [mA]
    Vs_kV           : Synchronous voltage [kV]
    mode            : 'active' or 'passive'
    phi_z2_passive_deg: HC tuning angle for passive mode [deg]
    Z_hom_kOhm      : HOM impedance for coupled-bunch [kOhm]
    f_hom_GHz       : HOM frequency [GHz]
    natural_sigma_t_ps: Natural bunch length [ps]

    Returns
    -------
    dict with complete analysis results
    """
    # Unit conversions
    E = E_MeV * 1e6  # eV
    V1 = V1_kV * 1e3  # V
    Vs = Vs_kV * 1e3  # V
    I_b = I_mA * 1e-3  # A
    f_rf = f_rf_MHz * 1e6  # Hz
    omega_g = 2 * np.pi * f_rf
    R1 = R1_MOhm * 1e6 / (1 + beta1)  # loaded impedance
    R2 = R2_MOhm * 1e6 / (1 + beta2)  # loaded impedance
    Q1 = Q01 / (1 + beta1)
    Q2 = Q02 / (1 + beta2)
    sigma_t_nat = natural_sigma_t_ps * 1e-12

    # ── Step 1: Equilibrium Phases ──
    if mode == 'active':
        eq = equilibrium_phases_active(V1, Vs, nu)
        if not eq['valid']:
            return {'error': eq['error'], 'valid': False}
        psi1 = eq['psi1_rad']
        psi2 = eq['psi2_rad']
        V2 = eq['V2']
    else:
        phi_z2 = np.radians(phi_z2_passive_deg)
        eq = equilibrium_phases_passive(V1, Vs, I_b, R2, Q2, f_rf, nu, phi_z2)
        psi1 = eq['psi1_rad']
        psi2 = eq['psi2_rad']
        V2 = eq['V2']

    # ── Step 2: Bunch Length ──
    bl = bunch_length_double_rf(
        V1, psi1, V2, psi2, nu,
        alpha_c, omega_g, E, sigma_E_over_E, T0_s, sigma_t_nat
    )
    sigma_t = bl['sigma_t_s']

    # ── Step 3: Form Factors ──
    ff = form_factors(sigma_t, f_rf, nu)

    # ── Step 4: Robinson Stability ──
    rob = robinson_two_cavity(
        alpha_c, omega_g, T0_s, E, I_b,
        V1, R1, Q1, psi1,
        V2, R2, Q2, psi2,
        nu, tau_rad_s,
        F1=ff['F1'], F2=ff['F2']
    )

    # ── Step 5: Synchrotron Frequency Spread ──
    sfs = synchrotron_frequency_spread(
        V1, psi1, V2, psi2, nu,
        alpha_c, omega_g, T0_s, E, sigma_t,
        F1=ff['F1'], F2=ff['F2']
    )

    # ── Step 6: Coupled-Bunch Instability ──
    Z_cb = Z_hom_kOhm * 1e3  # Ohm
    omega_cb = 2 * np.pi * f_hom_GHz * 1e9  # rad/s

    cb = coupled_bunch_growth_rate(
        I_b, alpha_c, E, T0_s,
        sfs['omega_s'], Z_cb, omega_cb, sigma_t, tau_rad_s,
        sigma_omega_s=sfs['sigma_omega_s'],
        is_quartic=not sfs['is_quadratic'],
        c_coeff=sfs['c'],
        Uo=bl['Uo']
    )

    # ── Summary ──
    return {
        'valid': True,
        'mode': mode,
        'equilibrium': eq,
        'bunch_length': bl,
        'form_factors': ff,
        'robinson': rob,
        'synchrotron': sfs,
        'coupled_bunch': cb,
        'summary': {
            'V2_kV': V2 / 1e3,
            'psi1_deg': np.degrees(psi1),
            'psi2_deg': np.degrees(psi2),
            'sigma_t_ps': sigma_t * 1e12,
            'lengthening': bl['lengthening_factor'],
            'f_s_kHz': sfs['f_s_kHz'],
            'f_spread_Hz': sfs['sigma_f_s_Hz'],
            'robinson_stable': rob['robinson_stable'],
            'eq_phase_stable': rob['eq_phase_stable'],
            'cb_stable': cb['stable'],
            'overall_stable': rob['overall_stable'] and cb['stable'],
        }
    }


# =============================================================================
# 9. Stability Map Scan (PAC97 Fig. 1 style)
# =============================================================================

def scan_stability_map(
    # Ring parameters
    E_MeV: float,
    alpha_c: float,
    T0_s: float,
    sigma_E_over_E: float,
    tau_rad_s: float,
    # Fundamental cavity
    V1_kV: float,
    R1_MOhm: float,
    Q01: float,
    beta1: float,
    f_rf_MHz: float,
    # Harmonic cavity
    R2_MOhm: float,
    Q02: float,
    nu: int,
    Vs_kV: float,
    # Scan ranges
    I_max_mA: float = 500,
    beta2_max: float = 40,
    n_I: int = 30,
    n_beta: int = 30,
    mode: str = 'active'
) -> Dict[str, np.ndarray]:
    """
    Generate a stability map as in PAC97 Fig. 1.

    Scans beam current (I) vs harmonic cavity coupling (β₂) and
    classifies each point as:
      - Robinson unstable
      - Equilibrium phase unstable
      - Coupled-bunch unstable
      - Stable

    Returns arrays suitable for plotting.
    """
    I_range = np.linspace(1, I_max_mA, n_I)
    beta_range = np.linspace(0.1, beta2_max, n_beta)

    robinson_map = np.zeros((n_I, n_beta))
    eq_phase_map = np.zeros((n_I, n_beta))
    cb_map = np.zeros((n_I, n_beta))
    stable_map = np.zeros((n_I, n_beta))

    for i, I_mA in enumerate(I_range):
        for j, beta2 in enumerate(beta_range):
            result = bosch_analysis(
                E_MeV=E_MeV, alpha_c=alpha_c, T0_s=T0_s,
                sigma_E_over_E=sigma_E_over_E, tau_rad_s=tau_rad_s,
                V1_kV=V1_kV, R1_MOhm=R1_MOhm, Q01=Q01, beta1=beta1,
                f_rf_MHz=f_rf_MHz,
                R2_MOhm=R2_MOhm, Q02=Q02, beta2=beta2, nu=nu,
                I_mA=I_mA, Vs_kV=Vs_kV, mode=mode
            )

            if not result.get('valid', False):
                robinson_map[i, j] = 0
                eq_phase_map[i, j] = 0
                cb_map[i, j] = 0
                stable_map[i, j] = 0
                continue

            rob_ok = result['robinson']['robinson_stable']
            eq_ok = result['robinson']['eq_phase_stable']
            cb_ok = result['coupled_bunch']['stable']

            robinson_map[i, j] = 1 if rob_ok else -1
            eq_phase_map[i, j] = 1 if eq_ok else -1
            cb_map[i, j] = 1 if cb_ok else -1
            stable_map[i, j] = 1 if (rob_ok and eq_ok and cb_ok) else -1

    return {
        'I_mA': I_range,
        'beta2': beta_range,
        'robinson_map': robinson_map,
        'eq_phase_map': eq_phase_map,
        'cb_map': cb_map,
        'stable_map': stable_map,
    }


# =============================================================================
# 10. Current Scan: Growth Rates vs Beam Current
# =============================================================================

def scan_current_bosch(
    # Ring parameters
    E_MeV: float,
    alpha_c: float,
    T0_s: float,
    sigma_E_over_E: float,
    tau_rad_s: float,
    # Fundamental cavity
    V1_kV: float,
    R1_MOhm: float,
    Q01: float,
    beta1: float,
    f_rf_MHz: float,
    # Harmonic cavity
    R2_MOhm: float,
    Q02: float,
    beta2: float,
    nu: int,
    Vs_kV: float,
    # Scan
    I_max_mA: float = 500,
    n_points: int = 50,
    mode: str = 'active',
    phi_z2_passive_deg: float = -30.0
) -> Dict[str, np.ndarray]:
    """
    Scan beam current to show Robinson and coupled-bunch growth rates.
    """
    I_range = np.linspace(1, I_max_mA, n_points)

    robinson_rate = []
    cb_growth = []
    eq_phase_val = []
    sigma_t_vals = []
    f_s_vals = []
    lengthening_vals = []
    landau_threshold_vals = []

    for I_mA in I_range:
        result = bosch_analysis(
            E_MeV=E_MeV, alpha_c=alpha_c, T0_s=T0_s,
            sigma_E_over_E=sigma_E_over_E, tau_rad_s=tau_rad_s,
            V1_kV=V1_kV, R1_MOhm=R1_MOhm, Q01=Q01, beta1=beta1,
            f_rf_MHz=f_rf_MHz,
            R2_MOhm=R2_MOhm, Q02=Q02, beta2=beta2, nu=nu,
            I_mA=I_mA, Vs_kV=Vs_kV, mode=mode,
            phi_z2_passive_deg=phi_z2_passive_deg
        )

        if result.get('valid', False):
            robinson_rate.append(result['robinson']['alpha_R'])
            cb_growth.append(result['coupled_bunch']['growth_rate'])
            eq_phase_val.append(
                1 if result['robinson']['eq_phase_stable'] else -1
            )
            sigma_t_vals.append(result['bunch_length']['sigma_t_ps'])
            f_s_vals.append(result['synchrotron']['f_s_kHz'])
            lengthening_vals.append(result['bunch_length']['lengthening_factor'])
            landau_threshold_vals.append(
                result['coupled_bunch']['landau_threshold_Hz']
            )
        else:
            robinson_rate.append(0)
            cb_growth.append(0)
            eq_phase_val.append(0)
            sigma_t_vals.append(0)
            f_s_vals.append(0)
            lengthening_vals.append(1)
            landau_threshold_vals.append(0)

    rad_damping = 1.0 / tau_rad_s if tau_rad_s > 0 else 0

    return {
        'I_mA': I_range,
        'robinson_rate': np.array(robinson_rate),
        'cb_growth': np.array(cb_growth),
        'eq_phase_stable': np.array(eq_phase_val),
        'sigma_t_ps': np.array(sigma_t_vals),
        'f_s_kHz': np.array(f_s_vals),
        'lengthening': np.array(lengthening_vals),
        'landau_threshold_Hz': np.array(landau_threshold_vals),
        'rad_damping_rate': rad_damping,
    }
