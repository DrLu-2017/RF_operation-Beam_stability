import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, quad, trapezoid
from scipy.constants import c, e as qe

class DoubleRFSystem:
    def __init__(self, E0, alpha_c, U0, V1, h, n=3, sigma_delta=1e-3):
        """
        Longitudinal Beam Dynamics in a Double RF System (Main + Harmonic Cavity).
        Based on the work of Hofmann and Myers.
        
        Parameters:
        - E0: Beam energy [eV]
        - alpha_c: Momentum compaction factor
        - U0: Energy loss per turn [eV] (compensated by V_tot(phi_s))
        - V1: Main cavity voltage [V]
        - h: Harmonic number
        - n: Harmonic ratio (typically 3 or 4)
        - sigma_delta: RMS energy spread
        """
        self.E0 = E0
        self.alpha_c = alpha_c
        self.U0 = U0
        self.V1 = V1
        self.h = h
        self.n = n
        self.sigma_delta = sigma_delta
        
        # Derived Internal Parameters
        self.V2 = 0
        self.phi_s1 = 0
        self.phi_s2 = 0
        
    def set_flat_potential(self):
        """
        Solve for V2 and phi_s2 to achieve the 'Flat Potential' condition:
        V'(phi_s) = 0 (Slope cancellation)
        V''(phi_s) = 0 (Curvature cancellation)
        
        Derivation:
        The total voltage is: V(phi) = V1 sin(phi + phi_s1) + V2 sin(n*phi + phi_s2)
        We define phi relative to the 'zero' of the system (synchronous phase at phi=0).
        1. V(0) = V1 sin(phi_s1) + V2 sin(phi_s2) = U0 / e
        2. V'(0) = V1 cos(phi_s1) + n V2 cos(phi_s2) = 0
        3. V''(0) = -V1 sin(phi_s1) - n^2 V2 sin(phi_s2) = 0
        
        From (3): V2 sin(phi_s2) = -(1/n^2) V1 sin(phi_s1)
        Sub into (1): V1 sin(phi_s1) [ 1 - 1/n^2 ] = U0 / e
        => sin(phi_s1) = (U0/e) / (V1 * (1 - 1/n^2))
        
        From (2): n V2 cos(phi_s2) = -V1 cos(phi_s1)
        => V2 cos(phi_s2) = -(V1/n) cos(phi_s1)
        
        Optimal harmonic voltage ratio: k = V2/V1
        At U0=0, k = 1/n. With U0 > 0, k is slightly different.
        """
        # sin(phi_s1) calculation
        # Note: 1 - 1/n^2 corresponds to the factor often denoted as (n^2-1)/n^2
        sin_phi_s1 = (self.U0) / (self.V1 * (1.0 - 1.0/self.n**2))
        
        if abs(sin_phi_s1) > 1:
            raise ValueError(f"Flat potential not possible: sin(phi_s1) = {sin_phi_s1:.3f} > 1. Increase V1.")
            
        # We choose the phase such that V' is 'heading' towards stable (but V'=0 here)
        # Usually phi_s1 is close to pi (back-leg of the sine)
        self.phi_s1 = np.pi - np.arcsin(sin_phi_s1)
        
        # Solve for V2 and phi_s2 from the V2*sin and V2*cos terms
        v2_sin = -(1.0/self.n**2) * self.V1 * np.sin(self.phi_s1)
        v2_cos = -(1.0/self.n) * self.V1 * np.cos(self.phi_s1)
        
        self.V2 = np.sqrt(v2_sin**2 + v2_cos**2)
        self.phi_s2 = np.arctan2(v2_sin, v2_cos)
        
        # The condition xi = 1 usually refers to the bunch lengthening factor k*n/cos(phi_s1)... 
        # specifically if U0=0, V2 = V1/n.
        print(f"Flat Potential Found:")
        print(f"  phi_s1 = {np.degrees(self.phi_s1):.2f} deg")
        print(f"  V2     = {self.V2/1e6:.4f} MV (k={self.V2/self.V1:.4f})")
        print(f"  phi_s2 = {np.degrees(self.phi_s2):.2f} deg")

    def get_voltage(self, phi):
        """Total voltage V(phi) relative to synchronous phase."""
        return self.V1 * np.sin(phi + self.phi_s1) + self.V2 * np.sin(self.n * phi + self.phi_s2)

    def get_potential(self, phi):
        """
        Calculates the effective longitudinal potential U(phi).
        U(phi) = (h*alpha_c / (2*pi*E0)) * integral_0^phi (V(phi') - U0/e) dphi'
        Units: Normalized potential such that it's dimensionless in the distribution exp(-U/sigma^2)
        """
        # We want to return an array of the same shape as phi
        # Integral can be done analytically:
        # Int(V(phi) - U0) dphi = 
        #   -V1 cos(phi + phi_s1) - (V2/n) cos(n*phi + phi_s2) - U0*phi
        # Subtract value at phi=0
        v_int = -self.V1 * np.cos(phi + self.phi_s1) - (self.V2/self.n) * np.cos(self.n * phi + self.phi_s2) - self.U0 * phi
        v0 = -self.V1 * np.cos(self.phi_s1) - (self.V2/self.n) * np.cos(self.phi_s2)
        
        # Factor for distribution: 1 / (2*pi*h*alpha_c*sigma_delta^2*E0) ?
        # Let's define U(phi) as the voltage integral scaled by 1/(2*pi*h)
        return (v_int - v0) / (2.0 * np.pi * self.h)

    def get_distribution(self, phi):
        """
        Bunch density distribution lambda(phi) based on Haissinski-like steady-state.
        lambda(phi) = rho0 * exp( - U_eff(phi) / (alpha_c * E0 * sigma_delta^2) ) ?
        Correct scaling: exponent = - U(phi) / (alpha_c * E0 * sigma_delta^2) 
        where U(phi) is the potential energy.
        """
        U_volts_rad = self.get_potential(phi) * (2 * np.pi * self.h) # back to integral(V-U0)
        # Exponent for longitudinal distribution:
        # - 1 / (alpha_c * E0 * sigma_delta^2) * e * integral(V-U0) * (C / 2pi*h) ???
        # Correct factor for phase distribution:
        # exp( - integral(V-U0) / (2*pi*h * alpha_c * E0 * sigma_delta^2) )
        
        factor = 1.0 / (2.0 * np.pi * self.h * self.alpha_c * self.sigma_delta**2 * self.E0)
        exponent = - (U_volts_rad) * factor
        
        # Normalize
        dist = np.exp(exponent - np.max(exponent))
        dist /= trapezoid(dist, phi)
        return dist

    def get_synchrotron_frequency(self, phi_amp):
        """
        Calculate synchrotron frequency f_s(phi_amp) as function of amplitude.
        Normalized to f_s0 (single RF small amplitude).
        """
        # Formula: Q_s(phi_hat) = pi / (2 * sqrt(2 * pi * h * alpha_c / E0) * Integral)
        # Integral = integral_0^phi_hat ( 1 / sqrt( Pot(phi_hat) - Pot(phi) ) ) dphi
        
        fs = []
        # Precompute voltage integral scaling
        k_factor = np.sqrt(2.0 * np.pi * self.h * self.alpha_c / self.E0)
        
        # Analytic voltage integral (V_pot)
        def v_pot(phi):
            # Not to be confused with self.get_potential
            val = -self.V1 * np.cos(phi + self.phi_s1) - (self.V2/self.n) * np.cos(self.n * phi + self.phi_s2) - self.U0 * phi
            return val
            
        for amp in phi_amp:
            if amp < 1e-5:
                # For flat potential, fs -> 0 at center (U approx phi^4)
                fs.append(0.0)
                continue
                
            v_amp = v_pot(amp)
            
            def integrand(p):
                diff = v_amp - v_pot(p)
                if diff <= 0: return 0
                return 1.0 / np.sqrt(diff)
            
            res, _ = quad(integrand, 0, amp)
            # Qs = pi / (2 * k_factor * res)
            qs = np.pi / (2.0 * k_factor * res)
            fs.append(qs)
            
        return np.array(fs)

def plot_all(system):
    phi_rad = np.linspace(-0.6, 0.6, 1200)
    phi_deg = np.degrees(phi_rad)
    
    # Calculate values
    v_tot = system.get_voltage(phi_rad)
    pot = system.get_potential(phi_rad)
    dist = system.get_distribution(phi_rad)
    
    # Comparison: Single RF
    system_s = DoubleRFSystem(system.E0, system.alpha_c, system.U0, system.V1, system.h, system.n, system.sigma_delta)
    system_s.phi_s1 = np.pi - np.arcsin(system.U0 / system.V1)
    system_s.V2 = 0
    dist_s = system_s.get_distribution(phi_rad)
    pot_s = system_s.get_potential(phi_rad)
    
    # fs distribution
    phi_amps = np.linspace(0.0, 0.4, 40)
    fs_double = system.get_synchrotron_frequency(phi_amps)
    # Small amp Qs for single RF: sqrt( h*alpha_c*V1*abs(cos(phi_s1)) / (2*pi*E0) )
    qs0 = np.sqrt(system.h * system.alpha_c * system.V1 * abs(np.cos(system_s.phi_s1)) / (2.0 * np.pi * system.E0))
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Voltage
    plt.subplot(2, 2, 1)
    plt.plot(phi_deg, v_tot/1e6, 'k', lw=2, label='Total Voltage')
    plt.plot(phi_deg, system.V1*np.sin(phi_rad + system.phi_s1)/1e6, 'b--', alpha=0.6, label='Main Cavity')
    plt.plot(phi_deg, system.V2*np.sin(system.n*phi_rad + system.phi_s2)/1e6, 'r:', alpha=0.6, label='Harmonic Cavity')
    plt.axhline(system.U0/1e6, color='gray', ls='-', lw=1, label='Energy Loss $U_0$')
    plt.axvline(0, color='gray', lw=1)
    plt.ylabel('Voltage [MV]')
    plt.title('1. Total RF Voltage $V(\\phi)$')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Potential Well
    plt.subplot(2, 2, 2)
    plt.plot(phi_deg, pot, 'r', lw=2, label='Double RF (Flat)')
    plt.plot(phi_deg, pot_s, 'b--', label='Single RF')
    plt.ylabel('Potential $U(\\phi)$ [V*rad / 2πh]')
    plt.title('2. Potential Well Comparison')
    plt.axvline(0, color='gray', lw=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Bunch Profile
    plt.subplot(2, 2, 3)
    plt.plot(phi_deg, dist, 'r', lw=2, label='Flat-topped (Double RF)')
    plt.plot(phi_deg, dist_s, 'b--', label='Gaussian (Single RF)')
    plt.fill_between(phi_deg, dist, alpha=0.2, color='red')
    plt.xlabel('Phase $\\phi$ [deg]')
    plt.ylabel('Density [arb. units]')
    plt.title('3. Bunch Distribution $\\lambda(\\phi)$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Synchrotron Frequency
    plt.subplot(2, 2, 4)
    plt.plot(np.degrees(phi_amps), fs_double/qs0, 'ro-', ms=4, label='Double RF (Flat Potential)')
    plt.axhline(1.0, color='blue', ls='--', label='Single RF (Linear)')
    plt.xlabel('Amplitude $\\phi_{amp}$ [deg]')
    plt.ylabel('$f_s / f_{s0}$')
    plt.title('4. Synchrotron Frequency Distribution')
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.annotate('$f_s \\to 0$ at center', xy=(0, 0), xytext=(5, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.tight_layout()
    plt.savefig('hofmann_analytic_model.png', dpi=150)
    print("Plots saved to hofmann_analytic_model.png")

if __name__ == "__main__":
    # Test Parameters
    params = {
        'E0': 2.75e9,        # 2.75 GeV
        'alpha_c': 1.0e-4, 
        'U0': 0.5e6,         # 0.5 MeV
        'V1': 1.5e6,         # 1.5 MV
        'h': 400,
        'n': 3,
        'sigma_delta': 1.0e-3
    }
    
    sys = DoubleRFSystem(**params)
    sys.set_flat_potential()
    plot_all(sys)
