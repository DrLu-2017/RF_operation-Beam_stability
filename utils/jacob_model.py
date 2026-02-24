"""
J. Jacob Model: DC Robinson Stability for Double RF Systems
===========================================================
Implements the DC Robinson stability analysis as presented by J. Jacob (ESRF) 
at the HarmonLIP'2022 workshop and other ESLS-RF workshops.

Reference:
  J. Jacob, "Passive vs Active Systems, DC Robinson, DLLRF," 
  HarmonLIP'2022 Workshop, MAX IV, Oct. 2022.

The model focuses on the stability of the longitudinal equilibrium (synchronous phase)
considering beam loading and RF feedback in both fundamental and harmonic cavities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

def jacob_robinson_analysis(
    # Ring parameters
    alpha: float,
    energy_ev: float,
    h1: int,
    u0_ev: float,
    # Main Cavity (MC)
    v1_v: float,
    rs1_ohm: float,
    ql1: float,
    psi1_rad: float,      # Detuning angle
    phi1_rad: float,      # Voltage phase relative to beam (Pedersen notation)
    g1: float,            # DRFB Gain
    # Harmonic Cavity (HC)
    v2_v: float,
    rs2_ohm: float,
    ql2: float,
    psi2_rad: float,
    phi2_rad: float,
    n_h: int,             # Harmonic ratio
    g2: float,            # DRFB Gain for HHC (usually 0 for passive)
    # Beam
    i_beam_a: float,
) -> Dict[str, float]:
    """
    Calculate the Jacob Robinson stability terms for a double RF system.
    
    Stability Criterion (Jacob's notation):
    The Robinson term M must be positive for stability.
    M = M1 + M2
    
    For each cavity k:
    Mk = (n_k / (R_sk * cos^2(psi_k) * (1+G_k))) * [ V_k * cos(phi_k) - I_b * R_sk * sin(2*psi_k) ]
    
    Note: n1 = 1, n2 = n_h.
    Values are normalized by R_sk*cos^2(psi_k) which is the slope of the loading term.
    """
    
    # Robinson term for Main Cavity (k=1)
    # Term = [ V1*cos(phi1) - Ib*R1*sin(2*psi1) ]
    term1 = v1_v * np.cos(phi1_rad) - i_beam_a * rs1_ohm * np.sin(2 * psi1_rad)
    m1 = (1.0 / (rs1_ohm * np.cos(psi1_rad)**2 * (1.0 + g1))) * term1 if rs1_ohm > 0 else 0
    
    # Robinson term for Harmonic Cavity (k=2)
    term2 = v2_v * np.cos(phi2_rad) - i_beam_a * rs2_ohm * np.sin(2 * psi2_rad)
    m2 = (n_h / (rs2_ohm * np.cos(psi2_rad)**2 * (1.0 + g2))) * term2 if rs2_ohm > 0 else 0
    
    m_total = m1 + m2
    
    # Phase stability check (equivalent to m_total > 0)
    is_stable = m_total > 0
    
    # Calculate effective synchrotron frequency from Robinson Term M
    # omega_s^2 = (alpha * omega_rev * e / E) * M * (R_eff...)
    # In J. Jacob's slides, the term inside the bracket is the primary stability indicator.
    
    return {
        'm1': m1,
        'm2': m2,
        'm_total': m_total,
        'term1': term1,
        'term2': term2,
        'is_stable': is_stable,
        'stability_margin': m_total,
        'm1_contrib_pct': (m1 / abs(m_total) * 100) if m_total != 0 else 0,
        'm2_contrib_pct': (m2 / abs(m_total) * 100) if m_total != 0 else 0,
    }

def scan_current_jacob(
    i_range_ma: np.ndarray,
    alpha: float,
    energy_ev: float,
    h1: int,
    u0_ev: float,
    # Cavity params (fixed for scan)
    v1_v: float, rs1_ohm: float, ql1: float, psi1_rad: float, phi1_rad: float, g1: float,
    v2_v: float, rs2_ohm: float, ql2: float, psi2_rad: float, phi2_rad: float, n_h: int, g2: float
) -> Dict[str, np.ndarray]:
    """Scan beam current and return Jacob Robinson terms."""
    m1_list = []
    m2_list = []
    m_total_list = []
    
    for i_ma in i_range_ma:
        res = jacob_robinson_analysis(
            alpha, energy_ev, h1, u0_ev,
            v1_v, rs1_ohm, ql1, psi1_rad, phi1_rad, g1,
            v2_v, rs2_ohm, ql2, psi2_rad, phi2_rad, n_h, g2,
            i_ma / 1000.0
        )
        m1_list.append(res['m1'])
        m2_list.append(res['m2'])
        m_total_list.append(res['m_total'])
        
    return {
        'current_ma': i_range_ma,
        'm1': np.array(m1_list),
        'm2': np.array(m2_list),
        'm_total': np.array(m_total_list)
    }

def calculate_jacob_phases(
    i_a: float,
    v1_v: float,
    phi_s_rad: float,
    rs1_ohm: float,
    ql1: float,
    v2_v: float,
    phi_h_rad: float,  # Phase of V2 relative to V1
    rs2_ohm: float,
    ql2: float,
    n_h: int
) -> Dict[str, float]:
    """
    Calculate the tuning angles and voltage phases for Jacob analysis.
    
    In J. Jacob's 2022 presentation, he assumes the compensated condition
    or a specific detuning.
    
    This helper aligns the physical parameters to the Robinson term inputs.
    """
    # Detuning angles (standard beam loading)
    # tan(psi) = (2*Ib*Rs*sin(phi_v) / V) if we want to cancel reactive power
    # But often we have fixed detuning.
    
    # For now, we assume phi1 is the phase of the voltage relative to the bunch.
    # In Pedersen notation, phi is the angle between V and I_beam (generator current phase relative to bunch).
    # However, common convention in these stability papers:
    # phi is phase of voltage relative to beam. 
    # psi is tuning angle.
    
    return {
        'phi1_rad': phi_s_rad,
        'phi2_rad': phi_h_rad,
    }
