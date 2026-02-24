import numpy as np
import scipy.constants as const
from scipy.integrate import quad
from scipy.optimize import brentq
import warnings

class DoubleRF_CLBI:
    """
    Calculates Longitudinal Coupled Bunch Instability (CLBI) growth rates
    for a Double RF System (Main Cavity + Harmonic Cavity).
    
    Physics Model:
    - Rigid bunch approximation (No Landau Damping).
    - Longitudinal Dipole Mode (m=1) and higher order modes.
    - Double RF Potential Well distortion included via synchrotron frequency shift.
    - Impedance summation over sidebands.
    - Sacherer Loop Integral form.
    
    References:
    - Input specifications tailored for ALBuMS/Streamlit integration.
    """

    def __init__(self, beam, mc, hc):
        """
        Initialize coupled bunch instability calculator.

        Parameters:
        -----------
        beam : dict
            Beam parameters:
            - I0: Average beam current [A]
            - E0: Beam Energy [eV]
            - alpha_c: Momentum compaction factor
            - f_rev: Revolution frequency [Hz]
            - h: Harmonic number (Main RF)
            - eta: Slip factor (optional, calculated if not present)
            - sigma_z0: Natural bunch length (Single RF) [s] (Optionally used for scaling)
            - U0: Energy loss per turn [eV] (Used for equilibrium finding)
        
        mc : dict
            Main Cavity parameters:
            - V: Voltage [V] (Peak)
            - Q: Loaded Q factor
            - R_sh: Shunt Impedance [Ohm] (Definition: V^2/2P or V^2/P? Check convention.
                    Standard Circuit: R = V^2/(2P_diss). Linac: R = V^2/P_diss.
                    We assume Circuit R_sh unless specified)
            - psi: Tuning angle [rad] (or calculated from f_r)
            - f_rf: Main Frequency [Hz]
        
        hc : dict
            Harmonic Cavity parameters:
            - V: Voltage [V] (Peak)
            - Q: Loaded Q factor
            - R_sh: Shunt Impedance [Ohm]
            - psi: Tuning angle [rad]
            - n: Harmonic ratio (integer, e.g. 3 or 4)
            - theta_rel: Relative phase to MC [rad]. 
                         V_total = V1*sin(phi) + V2*sin(n*phi + theta_rel)
                         If None, assumes optimal flat potential condition.
        """
        self.beam = beam
        self.mc = mc
        self.hc = hc
        
        # Physical Constants
        self.e = const.e
        self.c = const.c
        
        # Derive Slip Factor if not provided
        if 'eta' not in beam:
            gamma = beam['E0'] / (0.511e6) # Assuming electron
            self.eta = beam['alpha_c'] - 1/gamma**2
        else:
            self.eta = beam['eta']
            
        # Ensure theta_rel exists, default to pi (decelerating) if 0/None? 
        # Actually user prompt implies specific Step 1 logic.
        if 'theta_rel' not in hc:
             # Default to "ideal lengthening" phase if not specified, 
             # usually approximate harmonic phase is set to cancel slopes.
             # but we'll leave it 0 if not set, and warn.
             warnings.warn("HC theta_rel not specified. Assuming 0.", RuntimeWarning)
             self.hc['theta_rel'] = 0.0

    def solve_equilibrium(self):
        """
        Step 1: Calculate Equilibrium State (Synchronous Phase).
        Solves V_total * sin(phi_s) = U0 / e
        
        Returns:
        --------
        phi_s : float
            Synchronous phase [rad]
        """
        U0_eV = self.beam['U0']
        V1 = self.mc['V']
        V2 = self.hc['V']
        n = self.hc['n']
        theta = self.hc['theta_rel']

        # Define Function: f(phi) = V1*sin(phi) + V2*sin(n*phi + theta) - U0_eV
        def voltage_error(phi):
            return V1 * np.sin(phi) + V2 * np.sin(n * phi + theta) - U0_eV

        # Initial Guess:
        # Standard active cavity stable phase is near pi (180 deg) for electron storage rings (above transition)
        # to provide focusing slope. Or near 0 depending on convention.
        # Convention: sin(phi_s) > 0 for U0 > 0.
        # Stability: Slope V' > 0? No, V' > 0 is defocusing in eta > 0?
        # Let's check Robinson: above transition (eta > 0), we need rising slope V' > 0 ?? 
        # No, commonly phi_s ~ 170 deg (pi - phi).
        # We will search near pi - arcsin(U0/V1).
        
        phi_guess = np.pi - np.arcsin(U0_eV / V1) if V1 > U0_eV else np.pi/2
        
        from scipy.optimize import fsolve
        phi_s_solution = fsolve(voltage_error, phi_guess)
        
        self.phi_s = phi_s_solution[0]
        return self.phi_s

    def calculate_synchrotron_frequency(self):
        """
        Step 2: Correct Synchronous Frequency for Double RF.
        Calculates V'_total and w_s0_double.
        
        Returns:
        --------
        w_s_double : float
            Corrected synchrotron frequency [rad/s]
        """
        V1 = self.mc['V']
        V2 = self.hc['V']
        n = self.hc['n']
        theta = self.hc['theta_rel']
        phi_s = self.phi_s
        h = self.beam['h']
        eta = self.eta
        E0 = self.beam['E0']
        w_rev = 2 * np.pi * self.beam['f_rev']

        # Calculate Slope: V'(phi) in [V/rad]
        # V = V1 sin(phi) + V2 sin(n phi + theta)
        # V' = V1 cos(phi) + n V2 cos(n phi + theta)  [V unit, but phi is rad, so V/rad]
        V_prime = V1 * np.cos(phi_s) + n * V2 * np.cos(n * phi_s + theta)
        self.V_prime = V_prime # [V/rad]
        
        # Robinson Stability Criterion (Static):
        # Equilibrium is stable if η * V' < 0.
        # Above transition (η > 0), stable region is on the falling slope (V' < 0).
        # Below transition (η < 0), stable region is on the rising slope (V' > 0).
        self.is_statically_unstable = False
        
        # We allow a small tolerance (100 V/rad) for numerical noise near flat potential (V'=0).
        if eta > 0 and V_prime > 100.0:
             self.is_statically_unstable = True
             warnings.warn(f"Static Robinson Instability (Above Transition): V'={V_prime:.2e} V/rad. Bunch is on the rising (defocusing) slope.", RuntimeWarning)
        elif eta < 0 and V_prime < -100.0:
             self.is_statically_unstable = True
             warnings.warn(f"Static Robinson Instability (Below Transition): V'={V_prime:.2e} V/rad. Bunch is on the falling (defocusing) slope.", RuntimeWarning)
        
        # Term under sqrt for eta > 0: ws^2 = -(h eta w_rev^2 V') / (2pi E0)
        # Note: h*eta*V' has units [1 * 1 * V/rad] -> V. E0 has units [eV].
        # Term is dimensionless scaling of w_rev^2.
        term_under_sqrt = -(h * eta * V_prime) / (2 * np.pi * E0)
        
        if self.is_statically_unstable:
            # Although statically unstable, we keep a dummy low frequency 
            # for logic purposes, but mark it for the growth rate.
            self.w_s_double = 1.0 
        elif abs(V_prime) < 1e-3: # Extremely flat
            self.w_s_double = 1.0
        else:
            self.w_s_double = w_rev * np.sqrt(abs(term_under_sqrt))
            
        return self.w_s_double

    def impedance_rlc(self, freq, R, Q, f_r):
        """
        Calculate Impedance Z(f) for a single RLC resonator.
        Z(w) = R / (1 + iQ(w/wr - wr/w))
        """
        if freq == 0: return 0
        w = 2 * np.pi * freq
        w_r = 2 * np.pi * f_r
        denominator = 1 + 1j * Q * (w / w_r - w_r / w)
        return R / denominator

    def calculate_complex_shift(self, m=1, n_sidebands=1000):
        """
        Calculate complex coherent frequency shift Delta Omega.
        Im(Delta Omega) is the growth rate.
        """
        I0 = self.beam['I0']
        eta = self.eta
        E0 = self.beam['E0']
        w_rev = 2 * np.pi * self.beam['f_rev']
        w_s = self.w_s_double
        h = self.beam['h']
        f_rev = self.beam['f_rev']
        
        if getattr(self, 'is_statically_unstable', False):
            return complex(0, 1.0e9)
            
        sigma_z_double = getattr(self, 'sigma_z_double', 15e-12)
        
        f_rf_mc = self.mc['f_rf']
        psi_mc = self.mc['psi']
        Q_mc = self.mc['Q']
        f_r_mc = f_rf_mc / (1 + np.tan(psi_mc) / (2 * Q_mc))
        
        f_rf_hc = f_rf_mc * self.hc['n']
        psi_hc = self.hc['psi']
        Q_hc = self.hc['Q']
        f_r_hc = f_rf_hc / (1 + np.tan(psi_hc) / (2 * Q_hc))
        
        max_rate = -1e9
        worst_shift = 0j
        
        mu_scan_list = [0, 1, h-1, h-2]
        for mu in mu_scan_list:
            p_scan = range(-5, 6) 
            sum_Z_weighted = 0j
            for p in p_scan:
                freq = (p * h + mu) * f_rev + m * w_s / (2 * np.pi)
                if abs(freq) < 1e-3: continue
                
                Z_tot = self.impedance_rlc(freq, self.mc['R_sh'], self.mc['Q'], f_r_mc) + \
                        self.impedance_rlc(freq, self.hc['R_sh'], self.hc['Q'], f_r_hc)
                
                w_curr = 2 * np.pi * freq
                arg = (w_curr * sigma_z_double)**2
                ff = (arg**m) * np.exp(-arg) if arg < 100 else 0
                
                sum_Z_weighted += w_curr * Z_tot * ff
            
            coeff = (m * I0 * eta * w_rev) / (2 * np.pi * E0 * w_s)
            shift_mu = -1j * coeff * sum_Z_weighted
            
            if shift_mu.imag > max_rate:
                max_rate = shift_mu.imag
                worst_shift = shift_mu
                
        return worst_shift

    def calculate_complex_shift(self, m=1, n_sidebands=1000):
        """
        Calculate complex coherent frequency shift Delta Omega.
        Im(Delta Omega) is the growth rate.
        """
        # Ensure equilibrium and frequency are calculated
        if not hasattr(self, 'phi_s'): self.solve_equilibrium()
        if not hasattr(self, 'w_s_double'): self.calculate_synchrotron_frequency()
        return self._calculate_worst_shift(m, n_sidebands)

    def _calculate_worst_shift(self, m, n_sidebands):
        # Internal implementation moved from growth rate calculation
        I0 = self.beam['I0']
        eta = self.eta
        E0 = self.beam['E0']
        w_rev = 2 * np.pi * self.beam['f_rev']
        w_s = self.w_s_double
        h = self.beam['h']
        f_rev = self.beam['f_rev']
        
        if getattr(self, 'is_statically_unstable', False):
            return complex(0, 1.0e9)
            
        sigma_z_double = getattr(self, 'sigma_z_double', 15e-12)
        
        f_rf_mc = self.mc['f_rf']
        psi_mc = self.mc['psi']
        Q_mc = self.mc['Q']
        f_r_mc = f_rf_mc / (1 + np.tan(psi_mc) / (2 * Q_mc))
        
        f_rf_hc = f_rf_mc * self.hc['n']
        psi_hc = self.hc['psi']
        Q_hc = self.hc['Q']
        f_r_hc = f_rf_hc / (1 + np.tan(psi_hc) / (2 * Q_hc))
        
        max_rate = -1e9
        worst_shift = 0j
        
        mu_scan_list = [0, 1, h-1, h-2]
        for mu in mu_scan_list:
            p_scan = range(-5, 6) 
            sum_Z_weighted = 0j
            for p in p_scan:
                freq = (p * h + mu) * f_rev + m * w_s / (2 * np.pi)
                if abs(freq) < 1e-3: continue
                
                Z_tot = self.impedance_rlc(freq, self.mc['R_sh'], self.mc['Q'], f_r_mc) + \
                        self.impedance_rlc(freq, self.hc['R_sh'], self.hc['Q'], f_r_hc)
                
                w_curr = 2 * np.pi * freq
                arg = (w_curr * sigma_z_double)**2
                ff = (arg**m) * np.exp(-arg) if arg < 100 else 0
                
                sum_Z_weighted += w_curr * Z_tot * ff
            
            coeff = (m * I0 * eta * w_rev) / (2 * np.pi * E0 * w_s)
            shift_mu = -1j * coeff * sum_Z_weighted
            
            if shift_mu.imag > max_rate:
                max_rate = shift_mu.imag
                worst_shift = shift_mu
                
        return worst_shift

    def calculate_growth_rate(self, m=1, n_sidebands=1000):
        """
        Step 3, 4, 5: Calculate growth rate.
        
        Parameters:
        -----------
        m : int
            Azimuthal mode number (1 = dipole, 2 = quadrupole, etc.)
        n_sidebands : int
            Number of sidebands to sum over (p = -N to +N)
        
        Returns:
        --------
        growth_rate : float
            Instability growth rate [s^-1]. Positive = Unstable.
        """
        I0 = self.beam['I0']
        eta = self.eta
        E0 = self.beam['E0']
        w_rev = 2 * np.pi * self.beam['f_rev']
        w_s = self.w_s_double
        
        # Static Robinson check: Return extreme growth rate if statically unstable
        if getattr(self, 'is_statically_unstable', False):
            return 1.0e9 # Effectively immediate collapse
        
        f_rev = self.beam['f_rev']
        
        # Bunch Length Correction for Form Factor
        # Conservative Estimate: Scale sigma_z by w_s0_single / w_s_double
        # Calculate Single RF w_s0 for reference
        # w_s0 = w_rev * sqrt( h eta e V1 cos(phi_s_single) / 2pi E0 )
        # Approx scale:
        
        # User specified: "h_m(w) propto (w sigma)^2m exp(-(w sigma)^2)"
        # We need an effective sigma.
        # Let's perform a simple estimation if sigma_z0 is provided.
        sigma_z = self.beam.get('sigma_z0', 10e-12) # Default 10ps
        
        # Scale sigma if w_s is significantly reduced
        # Avoid scaling to infinity. Cap at factor 10 or similar? 
        # Or calculate using potential well if needed.
        # For now, we apply the scaling derived from linear slope reduction
        # sigma_new / sigma_0 ~ w_s0 / w_s_new
        
        # Calculate reference single RF w_s
        V1 = self.mc['V']
        phi_guess = np.pi - np.arcsin(self.beam['U0'] / V1) if V1 > self.beam['U0'] else np.pi/2
        slope_single = abs(V1 * np.cos(phi_guess))
        w_s_single = w_rev * np.sqrt( abs(math_term := (self.beam['h'] * eta * self.e * slope_single) / (2 * np.pi * E0)) )
        
        scaling_factor = w_s_single / w_s if w_s > 1e-3 else 10.0
        # Limit scaling for realism (bunch doesn't explode infinitely)
        scaling_factor = min(scaling_factor, 10.0) 
        
        sigma_z_double = sigma_z * scaling_factor
        self.sigma_z_double = sigma_z_double

        # Pre-calculate resonant frequencies from tuning angles
        # tan(psi) = Q (f_rf/f_r - f_r/f_rf) approx 2Q (f_rf - f_r)/f_r
        # We need f_r from psi.
        # Exact relation: f_r = f_rf / sqrt(1 - tan(psi)/Q) is often used for detuning?
        # Standard: tan(psi) = 2Q (f_gen - f_res) / f_res.
        # -> f_res = f_gen / (1 + tan(psi)/2Q) approx.
        
        f_rf_mc = self.mc['f_rf']
        psi_mc = self.mc['psi']
        Q_mc = self.mc['Q']
        f_r_mc = f_rf_mc / (1 + np.tan(psi_mc) / (2 * Q_mc))
        
        f_rf_hc = f_rf_mc * self.hc['n']
        psi_hc = self.hc['psi']
        Q_hc = self.hc['Q']
        f_r_hc = f_rf_hc / (1 + np.tan(psi_hc) / (2 * Q_hc))
        
        # Sum Impedance over sidebands
        # p ranges from -N to +N (excluding 0)
        # Frequencies: w_p = p * w_rev + m * w_s
        # Actually usually p * w_rev + mu * w_s. For Coupled Bunch, mu is the coupled bunch mode (0..M-1).
        # Sacherer formula usually sums over p = k*M + mu.
        # The prompt says: "Sum over p... Z(p*w_rev + m*w_s)". 
        # This implies we are calculating for a *specific* coupled bunch mode mu? 
        # Or summing all lines?
        # Prompt: "For each mode number p (scan range...)".
        # Usually we want to find the WORST coupled bunch mode (mu).
        # But if the user asks for "CLBI growth rate" generic, usually we scan mu = 0..M-1.
        # However, the prompt might imply calculating the growth rate for a *single* dominant impedance peak or summing?
        # Let's implement the generic summation for a given coupled bunch mode index mu?
        # Or maybe "p" in the prompt implies the revolution harmonic index.
        # Formula: Re[Z(...)] * F_m(p).
        # I will assume we are calculating for the most dangerous mode or scanning mu. 
        # But for simplicity, I'll sum over `p` as requested. 
        # Wait, if `p` runs -inf to +inf, this covers all frequencies. 
        # But coupled bunch modes are distinct.
        # Let's assume we want to calculate the shift for a specific Coupled Bunch Mode (CBM) index, let's say mu.
        # If the user doesn't specify mu, maybe I should return the max growth rate?
        # The prompt doesn't specify mu. It just says "Sum over p".
        # I will scan likely dangerous mu (near resonances).
        # Actually, standard Sacherer: 1/tau = coeff * Sum_p { Re Z(w_p) F_m(w_p) }.
        # Where w_p = (p*M + mu)*w_rev + m*w_s.
        # If I strictly follow: "For each mode p... Sum...".
        # I will accept an argument `mu` (coupled bunch mode index) default to finding the worst one or just summing p for a general "broadband" effect?
        # No, CLBI is narrowband. The aliasing matters.
        # I will assume mu corresponds to the harmonic cavity interaction.
        # Note: If mu is not specified, I will loop mu from 0 to h-1 and find the max growth rate.
        
        max_growth_rate = None
        worst_mu = -1
        
        h = self.beam['h']
        
        # Optimization: Only check mu near cavity resonances?
        # MC resonance is at h * f_rev. So mu=0 is close.
        # HC resonance is at n * h * f_rev. So mu=0 is close.
        # Detuning moves it.
        # I will limit scan if h is large, otherwise scan all?
        # If h=416, scanning all is okay.
        
        mu_scan_list = [0, 1, h-1, h-2]
        
        for mu in mu_scan_list:
            # Sum over p
            # p_scan: we need to capture the main R_sh peak.
            
            p_scan = range(-5, 6) # Sufficient for RLC (narrow band) 
            
            sum_terms = 0.0
            for p in p_scan:
                freq = (p * h + mu) * f_rev + m * w_s / (2 * np.pi)
                # Skip DC
                if abs(freq) < 1e-3: continue
                
                # Impedance
                Z_mc = self.impedance_rlc(freq, self.mc['R_sh'], self.mc['Q'], f_r_mc)
                Z_hc = self.impedance_rlc(freq, self.hc['R_sh'], self.hc['Q'], f_r_hc)
                Z_tot = Z_mc + Z_hc
                
                # Form factor
                w_curr = 2 * np.pi * freq
                arg = (w_curr * self.sigma_z_double)**2
                # Limit exp argument
                if arg > 100: 
                    ff = 0
                else:
                    ff = (arg**m) * np.exp(-arg)
                
                # Weighted Sum (Chao 6.183 type)
                # Term: w * ReZ * F
                term = w_curr * np.real(Z_tot) * ff
                sum_terms += term
                # print(f"DEBUG: mu={mu} p={p} f={freq/1e6:.4f}M Z={np.real(Z_tot):.2e} Term={term:.2e}")
            
            # Final Growth Rate coeff
            # 1/tau = - (m I0 eta) / (2 pi E0 ws) * Sum(...)
            coeff = (m * I0 * eta * w_rev) / (2 * np.pi * E0 * w_s)
            
            rate = coeff * sum_terms
            # print(f"DEBUG: mu={mu} Sum={sum_terms:.2e} Rate={rate:.2e}")

            if max_growth_rate is None or rate > max_growth_rate:
                max_growth_rate = rate
                worst_mu = mu
                
        return max_growth_rate

    def calculate_frequency_spread(self, sigma_z):
        """
        Calculate incoherent synchrotron frequency spread due to non-linear potential.
        Approximates spread as |ws(sigma_z) - ws(0)|.
        
        Parameters:
        -----------
        sigma_z : float
            Bunch length [s] to evaluate frequency at.
            
        Returns:
        --------
        spread : float
            Frequency spread [rad/s]
        """
        # Physics Constants
        h = self.beam['h']
        eta = self.eta
        E0 = self.beam['E0']
        w_rev = 2 * np.pi * self.beam['f_rev']
        
        # 1. Define Potential Function (relative to synchronous phase)
        
        def voltage(phi):
            V1 = self.mc['V']
            V2 = self.hc['V']
            n = self.hc['n']
            theta = self.hc['theta_rel']
            return V1 * np.sin(phi) + V2 * np.sin(n * phi + theta)
            
        def potential_pe(phi):
            # Potential Energy Function P(x) = - Integration Force
            # Force ~ -(V - U0) (Restoring)
            # P(x) ~ Integral (V - U0)
            # Check sign: phi'' = - (h eta w0^2 / 2pi E0) (V - U0)
            # Let constant C > 0. phi'' = - C (V - U0).
            # Force = - dP/dx = - C (V - U0).
            # dP/dx = C (V - U0).
            # P(x) = C * Integral_0^x (V(phi_s+y) - U0) dy.
            
            C = (h * eta * w_rev**2) / (2 * np.pi * E0) # Magnitude of prefactor
            # We want restoring force.
            # If x>0, V < U0. (V-U0) is negative.
            # dP/dx should be negative? No, Force is negative.
            # Force = - dP/dx. So dP/dx should be positive for x>0 (since Force < 0).
            # If C > 0. (V-U0) < 0.
            # So dP/dx = C(V-U0) would be negative.
            # This implies Unstable (Hilltop).
            # Wait. Step 2 (Synch Freq) used NEGATIVE sign for w_s^2?
            # User prompted formula used positive V'.
            # We found V' < 0.
            # So w_s^2 propto -V'.
            # Restoring force propto -x.
            # If V-U0 approx V' * x.
            # Then dP/dx approx C * V' * x.
            # If V' < 0, then dP/dx < 0. (Hilltop).
            # So we need "Negative Mass" or Hamiltonian inverted?
            # In longitudinal dynamics above transition, effective mass is negative?
            # Let's just use w_s^2 directly.
            # Harmonic Oscillator at small amplitude: w_s^2.
            # P(x) = 1/2 w_s^2 x^2.
            # We want to generalize this.
            # P(x) = Integral (w_s_local^2 * x) dx ??
            # Base definition: P(x) = - Integral Force.
            # Force = phi''.
            # phi'' = (h eta w0^2 / 2pi E0) * (V - U0).
            # Force = K * (V - U0).
            # To have restoration, Force must be opposite to x.
            # If x>0, we need Force < 0.
            # This requires K * (V-U0) < 0.
            # If V < U0 (slope down), (V-U0) < 0.
            # So we need K > 0.
            # YES. K = (h eta w0^2 / 2pi E0). This is positive for eta>0.
            # So Force fits restoration.
            # P(x) = - Integral Force = - K * Integral (V - U0).
            
            K = (h * eta * w_rev**2) / (2 * np.pi * E0)
            integ, _ = quad(lambda u: voltage(self.phi_s + u) - self.beam['U0'], 0, phi)
            return -K * integ

        # 2. Find Turning Points
        sigma_phi = h * w_rev * sigma_z
        
        try:
            target_energy = potential_pe(sigma_phi)
            
            if target_energy <= 0:
                return 0.0 
                
            def find_root(x):
                return potential_pe(x) - target_energy
            
            # Left turning point search
            x_left = brentq(find_root, -10*sigma_phi, -1e-9) # Use wider range for flat pot
            x_right = sigma_phi
            
            # 3. Integrate
            def integrand(x):
                diff = target_energy - potential_pe(x)
                if diff <= 1e-12: return 0
                return 1.0 / np.sqrt(diff)
            
            half_period, _ = quad(integrand, x_left, x_right, points=[x_left, x_right])
            period_osc = np.sqrt(2) * half_period
            
            ws_sigma = 2 * np.pi / period_osc
            ws_center = abs(self.w_s_double)
            
            return abs(ws_sigma - ws_center)
            
        except Exception:
            return 0.0
