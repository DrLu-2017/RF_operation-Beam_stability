import numpy as np
from typing import Dict, List, Tuple

def olsson_transient_beam_loading(
    I0_A: float,
    f_rf_Hz: float,
    h: int,
    R_sh_MOhm: float,
    Q_L: float,
    detuning_Hz: float,
    V_c_kV: float,
    gap_fraction: float = 0.1,
    n_bunches: int = None
) -> Dict:
    """
    Olsson Model (PRAB 21, 120701, 2018)
    Self-consistent calculation of transient beam loading in electron storage rings 
    with passive harmonic cavities.
    
    This function implements a simplified analytical estimate of the transient 
    beam loading (phase shift across a bunch train with a gap).
    
    Parameters
    ----------
    I0_A : Average beam current [A]
    f_rf_Hz : RF frequency [Hz]
    h : Harmonic number
    R_sh_MOhm : Shunt impedance [MOhm]
    Q_L : Loaded quality factor
    detuning_Hz : Cavity detuning [Hz]
    V_c_kV : Cavity voltage [kV]
    gap_fraction : Fraction of the ring that is empty (0 to 1)
    
    Returns
    -------
    dict with transient beam loading properties
    """
    if n_bunches is None:
        n_bunches = int(h * (1.0 - gap_fraction))
        
    T_rev = h / f_rf_Hz
    omega_rf = 2 * np.pi * f_rf_Hz
    
    Rs = R_sh_MOhm * 1e6
    Vc = V_c_kV * 1e3
    
    # Cavity fill time
    tau_f = 2 * Q_L / omega_rf
    
    # Cavity bandwidth
    bw_Hz = f_rf_Hz / (2 * Q_L)
    
    # Gap duration
    t_gap = gap_fraction * T_rev
    
    # Transient Voltage drop over the gap (simplified Wilson approximation)
    # Delta V = V_br * (t_gap / tau_f)
    # where V_br is the beam-induced steady state voltage
    
    # Approximate beam induced voltage magnitude
    # Using generator convention, the beam current at RF is ~ 2 * I0
    I_rf = 2 * I0_A
    V_br = I_rf * Rs * np.cos(np.arctan(2 * Q_L * detuning_Hz / f_rf_Hz))
    
    # Transient voltage drop
    delta_V = V_br * (t_gap / tau_f) if tau_f > 0 else 0
    
    # Phase shift across the train (Delta theta)
    # The phase shift is related to the voltage drop and total voltage
    if Vc > 0:
        delta_phi_max_rad = delta_V / Vc
    else:
        delta_phi_max_rad = 0
        
    delta_phi_max_deg = np.degrees(delta_phi_max_rad)
    
    return {
        'model': 'Olsson TBL (2018)',
        'reference': 'T. Olsson et al., PRAB 21, 120701 (2018)',
        'gap_fraction': gap_fraction,
        'n_bunches_filled': n_bunches,
        't_gap_ns': t_gap * 1e9,
        'cavity_fill_time_us': tau_f * 1e6,
        'V_beam_induced_kV': V_br / 1e3,
        'max_voltage_drop_kV': delta_V / 1e3,
        'max_phase_shift_deg': delta_phi_max_deg,
        'description': 'A semi-analytical calculation showing the phase modulation '
                       'induced by an empty gap in the bunch train, leading to '
                       'bunch-by-bunch variations in synchronous phase and bunch length.'
    }

def calculate_bunch_profile(
    res_olsson: Dict,
    base_phi_s_deg: float,
    h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an array of synchronous phase across the bunch train.
    """
    phases = []
    bunches = np.arange(h)
    
    gap_start = res_olsson['n_bunches_filled']
    
    # Simple linear relaxation during the gap and beam-loading climb during the train
    current_phi = base_phi_s_deg - res_olsson['max_phase_shift_deg'] / 2.0
    
    for b in bunches:
        if b < gap_start:
            # Beam loading increases phase shift
            current_phi += res_olsson['max_phase_shift_deg'] / res_olsson['n_bunches_filled']
            phases.append(current_phi)
        else:
            # Gap reduces phase shift (cavity relaxes)
            current_phi -= res_olsson['max_phase_shift_deg'] / (h - gap_start)
            phases.append(None) # No bunch here to have a phase
            
    return bunches, np.array(phases, dtype=float)
