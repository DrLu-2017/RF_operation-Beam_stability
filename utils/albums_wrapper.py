"""
Wrapper functions for ALBuMS operations.
Provides simplified interfaces for Streamlit UI.
"""
import numpy as np
import sys
import platform
from pathlib import Path

# Add local modules to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add mbtrack2 and ALBuMS paths
mbtrack2_path = project_root / "mbtrack2-stable"
if mbtrack2_path.exists():
    sys.path.insert(0, str(mbtrack2_path))

pycolleff_path = project_root / "collective_effects-master" / "pycolleff"
if pycolleff_path.exists():
    sys.path.insert(0, str(pycolleff_path))

# Note: 'sh' module is only available on Linux/macOS
# On Windows, ALBuMS will work without it
if platform.system() != "Windows":
    try:
        import sh
        SH_AVAILABLE = True
    except ImportError:
        print("Warning: 'sh' module not available (only needed on Linux/macOS)")
        SH_AVAILABLE = False
else:
    # 'sh' is not available on Windows
    SH_AVAILABLE = False
    print("Note: Running on Windows. 'sh' module not needed.")

# Try to import mbtrack2 objects first (needed to create real objects)
try:
    from mbtrack2.tracking.synchrotron import Synchrotron
    from mbtrack2.tracking.rf import CavityResonator
    from mbtrack2.tracking.particles import Particle
    MBTRACK2_OBJECTS_AVAILABLE = True
except ImportError as e:
    print(f"Info: mbtrack2 objects not available: {e}")
    MBTRACK2_OBJECTS_AVAILABLE = False
    Synchrotron = None
    CavityResonator = None
    Particle = None

# Try to import ALBuMS modules
try:
    from albums.robinson import RobinsonModes
    from albums.scan import scan_psi_I0, scan_psi_RoQ, scan_psi_QL, scan_modes
    from albums.optimiser import maximize_R, maximize_R_equilibrium
    ALBUMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ALBuMS modules not available: {e}")
    # This is expected if full installation is not done
    # The wrapper will still work with parameter dictionaries
    ALBUMS_AVAILABLE = False
    RobinsonModes = None
    scan_psi_I0 = None
    scan_psi_RoQ = None
    scan_psi_QL = None
    scan_modes = None
    maximize_R = None
    maximize_R_equilibrium = None


class DictObject:
    """
    Wrapper class that allows dict-like access with attribute access.
    
    This allows ALBuMS functions to treat parameter dicts as objects
    with settable attributes (e.g., cavity.psi = value) and methods.
    
    Includes stub implementations of ALBuMS cavity methods:
    - set_optimal_detune(I0, F=None): Set optimal detuning
    - set_generator(I0): Set generator parameters
    - Vb(I0): Calculate beam loading voltage
    
    Examples:
        cavity = DictObject({"voltage": 1.0, "frequency": 500})
        cavity.psi = 0.5  # Can set attributes
        print(cavity.psi)  # Returns 0.5
        cavity.set_optimal_detune(0.1)  # Can call methods
        vb = cavity.Vb(0.1)  # Can call Vb method
    """
    def __init__(self, data):
        if isinstance(data, dict):
            self.__dict__.update(data)
        else:
            # If already an object, copy its attributes
            self.__dict__.update(vars(data))
    
    def __getitem__(self, key):
        return self.__dict__.get(key)
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def __getattribute__(self, name):
        """Override attribute access to prioritize methods over dict values."""
        # Avoid recursion on __dict__ and related special attributes
        if name in ('__dict__', '__class__', '__doc__'):
            return object.__getattribute__(self, name)
        
        # Check if this is a method/property on the class first
        # This ensures methods and properties are used instead of dict values with the same name
        try:
            for cls in type(self).__mro__:
                if name in cls.__dict__:
                    attr = cls.__dict__[name]
                    # Handle descriptors (methods, properties, etc.)
                    if hasattr(attr, '__get__'):
                        return attr.__get__(self, type(self))
                    # Handle direct callables/attributes
                    elif callable(attr):
                        return attr
                    return attr
        except (AttributeError, TypeError):
            pass
        
        # Check instance __dict__ for regular attributes
        try:
            instance_dict = object.__getattribute__(self, '__dict__')
            if name in instance_dict:
                return instance_dict[name]
        except AttributeError:
            pass
        
        # If not found anywhere, raise AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Allow attribute setting via dot notation"""
        # Special handling for private/internal attributes
        if name in ('__dict__', '__class__'):
            object.__setattr__(self, name, value)
        else:
            # Store in internal dict for regular attributes
            object.__getattribute__(self, '__dict__')[name] = value
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)
    
    def __deepcopy__(self, memo):
        """Support deepcopy for DictObject"""
        from copy import deepcopy as _deepcopy
        # Create a new DictObject with a deep copy of the internal dict
        new_dict = {}
        for key, value in self.__dict__.items():
            new_dict[key] = _deepcopy(value, memo)
        result = DictObject(new_dict)
        memo[id(self)] = result
        return result
    
    def __copy__(self):
        """Support copy for DictObject"""
        new_dict = dict(self.__dict__)
        return DictObject(new_dict)
    
    def set_optimal_detune(self, I0, F=None):
        """
        Stub method for setting optimal detuning.
        In a real cavity object, this would calculate optimal detuning.
        For dict wrapper, we just store the parameters.
        """
        # Store parameters that might be used
        self.__dict__['_I0_detune'] = I0
        if F is not None:
            self.__dict__['_F_detune'] = F
        # Detuning is already initialized to 0.0 in create_cavity_from_params
        return None
    
    def set_generator(self, I0):
        """
        Stub method for setting generator parameters.
        In a real cavity object, this would calculate generator voltage/angle.
        For dict wrapper, we just store the current.
        """
        # Store current parameter
        self.__dict__['_I0_generator'] = I0
        # Generator parameters (Vg, theta_g) are already initialized in create_cavity_from_params
        return None
    
    def Vb(self, I0):
        """
        Calculate beam loading voltage.
        
        For a thin cavity with series resistance Rs and quality factor Q:
        Vb = I0 * Rs = I0 * R/Q * Q = I0 * R
        
        Parameters
        ----------
        I0 : float
            Beam current
            
        Returns
        -------
        Vb : float
            Beam loading voltage
        """
        # Simple calculation: Vb = I0 * Rs
        # Rs is already in __dict__ from cavity params
        Rs = self.__dict__.get('Rs', self.__dict__.get('resistance', 1000))
        return I0 * Rs
    
    def __repr__(self):
        return f"DictObject({self.__dict__})"


class RingObject(DictObject):
    """
    Ring-specific wrapper that extends DictObject with required methods.
    
    ALBuMS RobinsonModes expects ring objects to have an eta() method that
    returns the slip factor (momentum compaction factor).
    
    This class provides that method while maintaining dict-like behavior.
    """
    
    def eta(self):
        """
        Return the slip factor (momentum compaction factor).
        
        The slip factor is used in ALBuMS calculations to determine the
        longitudinal dynamics of the beam. For synchrotron radiation dominated
        systems, it's typically close to the momentum compaction factor.
        
        Returns
        -------
        eta : float
            Slip factor / momentum compaction factor
        """
        # Look for eta value in dict using different possible key names
        # to avoid circular reference with this method
        eta_value = None
        for key in ['eta_value', '_eta', 'momentum_compaction', 'ac']:
            val = self.__dict__.get(key)
            if val is not None and not callable(val):
                eta_value = val
                break
        
        if eta_value is not None:
            try:
                return float(eta_value)
            except (TypeError, ValueError):
                pass
        
        # Fallback: compute from momentum_compaction or ac
        ac = self.__dict__.get('momentum_compaction') or self.__dict__.get('ac')
        if ac is not None:
            try:
                ac_val = float(ac)
                return float(abs(ac_val)) if ac_val != 0 else 0.01
            except (TypeError, ValueError):
                pass
        
        # Last resort: return default small value
        print("Warning: eta not found in ring object, using default 0.01")
        return 0.01
    
    @property
    def T1(self):
        """
        Return the RF period (seconds).
        
        T1 is accessed as an attribute in ALBuMS code (e.g., ring.T1 * 0.1),
        so we define it as a property.
        """
        T1_value = self.__dict__.get('T1')
        if T1_value is not None and not callable(T1_value):
            try:
                return float(T1_value)
            except (TypeError, ValueError):
                pass
        
        # Compute from T0 and h (harmonic number)
        T0 = self.__dict__.get('T0')
        h = self.__dict__.get('h')
        if T0 is not None and h is not None:
            try:
                return float(T0) * float(h)
            except (TypeError, ValueError):
                pass
        
        print("Warning: T1 not found in ring object")
        return None
    
    def synchrotron_tune(self, Vrf):
        """
        Calculate the synchrotron tune from RF voltage.
        
        The synchrotron tune (fractional) is given by:
        ν_s = sqrt(|η * h * Vrf / (2π * E0)|)
        
        where:
        - η is the slip factor (momentum compaction)
        - h is the harmonic number
        - Vrf is the RF voltage in [V]
        - E0 is the beam energy in [eV]
        
        Parameters
        ----------
        Vrf : float
            RF cavity voltage in volts
            
        Returns
        -------
        sync_tune : float
            Synchrotron tune (fractional)
        """
        try:
            eta = self.eta()
            h = float(self.__dict__.get('h', 1))
            E0 = float(self.__dict__.get('E0', 1e9))
            
            if h <= 0 or E0 <= 0:
                print("Warning: Invalid h or E0 in synchrotron_tune calculation")
                return 0.01
            
            # Calculate synchrotron tune
            sync_tune_sq = abs(eta * h * Vrf / (2 * np.pi * E0))
            sync_tune = np.sqrt(sync_tune_sq)
            
            return sync_tune
        except Exception as e:
            print(f"Warning: Error calculating synchrotron_tune: {e}")
            return 0.01



class CavityWrapper(DictObject):
    """
    Wrapper for Cavity objects that implements necessary physics methods for ALBuMS
    without requiring the full mbtrack2 library.
    """
    def __init__(self, data):
        super().__init__(data)
        # Ensure critical attributes exist and convert types
        if not hasattr(self, 'theta'): self.theta = 0.0
        if not hasattr(self, 'psi'): self.psi = 0.0
        if not hasattr(self, 'Vc'): self.Vc = 0.0
        
        # Ensure numerical types for critical parameters
        self.Rs = float(self.Rs) if hasattr(self, 'Rs') else float(self.get('resistance', 0))
        self.Q = float(self.Q) if hasattr(self, 'Q') else float(self.get('Q', 1))
        self.QL = float(self.QL) if hasattr(self, 'QL') else float(self.get('QL', self.Q))
        self.Vc = float(self.Vc) if hasattr(self, 'Vc') else float(self.get('voltage_V', 0))
        
        if not hasattr(self, 'beta'): 
            if self.QL != 0:
                self.beta = self.Q / self.QL - 1
            else:
                self.beta = 0.0
        else:
            self.beta = float(self.beta)
    
    def Vbr(self, I0):
        """Beam voltage at resonance [V]"""
        # Rs is total shunt impedance if Ncav=1 (which we assume for the wrapper)
        return 2 * I0 * self.Rs / (1 + self.beta)

    def Vb(self, I0):
        """Beam voltage [V]"""
        return self.Vbr(I0) * np.cos(float(self.psi))

    def set_optimal_detune(self, I0, F=1.0):
        """
        Set detuning to optimal conditions.
        Updates self.psi based on current and beam loading.
        """
        try:
            # Check for required values
            if self.Vc == 0:
                print("Warning: Vc is 0, cannot set optimal detune")
                return 
            
            # Recalculate Vbr with current I0
            vbr = self.Vbr(I0)
            
            # Determine theta (synchronous phase)
            # If we have access to the ring, calculate it from U0 to ensure accuracy
            # Otherwise use stored theta
            theta_val = float(getattr(self, 'theta', 0.0))
            
            if hasattr(self, 'ring'):
                try:
                    # Calculate synchronous phase: cos(phi_s) = U0 / Vc
                    # U0 should be in eV, Vc in V
                    U0 = float(getattr(self.ring, 'U0', 0))
                    if U0 > 0 and self.Vc > U0:
                        theta_val = np.arccos(U0 / self.Vc)
                except Exception:
                    pass
            
            # Calculate optimal psi
            # Formula: tan(psi) = - (Vbr * F / Vc) * sin(theta)
            arg = - (vbr * F / self.Vc) * np.sin(theta_val)
            self.psi = np.arctan(arg)
            self.theta = theta_val # Update theta too if we calculated it
            
        except Exception as e:
            print(f"Warning: Failed to set optimal detune: {e}")

    def set_generator(self, I0):
        """Mock set_generator to avoid errors if called"""
        pass

def dict_to_synchrotron(ring_params):
    """
    Convert parameter dictionary to mbtrack2 Synchrotron object.
    
    This is the cleanest approach - converts parameters to real ALBuMS objects.
    
    Parameters
    ----------
    ring_params : dict
        Ring parameters from create_ring_from_params()
        
    Returns
    -------
    ring : Synchrotron object or dict
        Real Synchrotron object if mbtrack2 available, otherwise returns dict
    """
    if not MBTRACK2_OBJECTS_AVAILABLE:
        # Fallback: return dict as-is if objects not available
        print("Warning: mbtrack2 objects not available. Using parameter dict.")
        return ring_params
    
    try:
        # Create a Synchrotron object from parameters
        # This requires energy in eV, circumference in m, etc.
        ring = Synchrotron(
            h=ring_params["harmonic_number"],
            # Note: Synchrotron requires Optics and Particle objects
            # For now, we use a simplified approach with dict
            # In production, would need full setup with lattice/optics
            L=ring_params["circumference"],
            E0=ring_params.get("energy_eV", ring_params["energy"] * 1e9),
            ac=ring_params["momentum_compaction"],
            U0=ring_params.get("energy_loss_eV", ring_params["energy_loss_per_turn"] * 1e9)
        )
        print(f"✅ Created Synchrotron object: {type(ring)}")
        return ring
    except TypeError as e:
        # Synchrotron might require more parameters
        print(f"Note: Full Synchrotron object creation failed ({e})")
        print("      Using parameter dict instead (limited functionality)")
        return ring_params
    except Exception as e:
        print(f"Warning: Error creating Synchrotron: {e}")
        return ring_params


def dict_to_cavity_resonator(cavity_params, ring):
    """
    Convert parameter dictionary to mbtrack2 CavityResonator object.
    
    Parameters
    ----------
    cavity_params : dict
        Cavity parameters from create_cavity_from_params()
    ring : Synchrotron object or dict
        Ring object or parameters
        
    Returns
    -------
    cavity : CavityResonator object or dict
        Real CavityResonator object if mbtrack2 available, otherwise returns dict
    """
    if not MBTRACK2_OBJECTS_AVAILABLE:
        # Fallback: return dict as-is if objects not available
        return cavity_params
    
    try:
        # For CavityResonator, we need the ring object
        # If ring is a dict, we can't create a real CavityResonator
        if isinstance(ring, dict):
            print("Note: Ring is dict, using cavity dict (limited functionality)")
            return cavity_params
        
        # Create cavity object
        cavity = CavityResonator(
            ring=ring,
            m=cavity_params["harmonic"],
            Rs=cavity_params.get("resistance", cavity_params["R_over_Q"] * cavity_params.get("Q", 14500)),
            Q=cavity_params.get("Q", 14500),
            Vc=cavity_params.get("voltage_V", cavity_params["voltage"] * 1e6)
        )
        print(f"✅ Created CavityResonator object: {type(cavity)}")
        return cavity
    except Exception as e:
        print(f"Note: CavityResonator creation failed ({e}), using dict")
        return cavity_params


def validate_parameters(ring_dict, mc_dict, hc_dict, current):
    """
    Validate and sanitize parameters to prevent numerical issues.
    
    Checks:
    - All voltages > 0 (prevents arccos out of range)
    - Current > 0
    - Quality factors reasonable (Q > 0, QL > 0)
    - Frequencies > 0
    - Critical beam dynamics parameters non-zero
    
    Parameters
    ----------
    ring_dict : dict
        Ring parameters
    mc_dict : dict
        Main cavity parameters
    hc_dict : dict
        Harmonic cavity parameters
    current : float
        Beam current
        
    Returns
    -------
    valid : bool
        Whether parameters are valid
    issues : list
        List of issues found (empty if valid)
    """
    issues = []
    
    # Check ring parameters
    if ring_dict.get('energy', 0) <= 0:
        issues.append(f"Ring energy must be > 0, got {ring_dict.get('energy')}")
    
    if ring_dict.get('harmonic_number', 0) <= 0:
        issues.append(f"Harmonic number must be > 0, got {ring_dict.get('harmonic_number')}")
    
    if ring_dict.get('circumference', 0) <= 0:
        issues.append(f"Ring circumference must be > 0, got {ring_dict.get('circumference')}")
    
    # Check critical beam dynamics parameters (cannot be zero - causes division)
    sigma_delta = ring_dict.get('sigma_delta', 0)
    if sigma_delta <= 0:
        issues.append(f"Ring sigma_delta must be > 0 (momentum spread), got {sigma_delta}. Minimum is 0.0001.")
    
    ac = ring_dict.get('ac', None)
    if ac is None or ac == 0:
        issues.append(f"Ring momentum compaction 'ac' cannot be 0, got {ac}")
    
    # Check main cavity
    mc_voltage = mc_dict.get('voltage', 0)
    if mc_voltage <= 0:
        issues.append(f"Main cavity voltage must be > 0, got {mc_voltage} MV. Minimum is 0.001 MV.")
    
    mc_vg = mc_dict.get('Vg', 0)
    if mc_vg <= 0:
        issues.append(f"Main cavity generator voltage Vg must be > 0, got {mc_vg} V")
    
    if mc_dict.get('Q', 0) <= 0:
        issues.append(f"Main cavity Q must be > 0, got {mc_dict.get('Q')}")
    
    if mc_dict.get('R_over_Q', 0) <= 0:
        issues.append(f"Main cavity R/Q must be > 0, got {mc_dict.get('R_over_Q')}")
    
    # Check harmonic cavity
    hc_voltage = hc_dict.get('voltage', 0)
    if hc_voltage <= 0:
        issues.append(f"Harmonic cavity voltage must be > 0, got {hc_voltage} MV. Minimum is 0.001 MV.")
    
    hc_vg = hc_dict.get('Vg', 0)
    if hc_vg <= 0:
        issues.append(f"Harmonic cavity generator voltage Vg must be > 0, got {hc_vg} V")
    
    if hc_dict.get('Q', 0) <= 0:
        issues.append(f"Harmonic cavity Q must be > 0, got {hc_dict.get('Q')}")
    
    if hc_dict.get('R_over_Q', 0) <= 0:
        issues.append(f"Harmonic cavity R/Q must be > 0, got {hc_dict.get('R_over_Q')}")
    
    # Check current
    if current <= 0:
        issues.append(f"Beam current must be > 0, got {current} A")
    
    return len(issues) == 0, issues


def create_ring_from_params(circumference, energy, momentum_compaction, 
                            energy_loss_per_turn, harmonic_number, damping_time):
    """
    Create a ring parameters dictionary (compatible with ALBuMS).
    
    Calculates all required ALBuMS ring attributes including:
    - E0, T0, T1, U0: Energy and timing parameters
    - ac, h, sigma_0, sigma_delta, eta: Beam dynamics parameters
    - omega0, omega1, f0, f1: Frequency parameters
    - tau, k1: Damping and coupling parameters
    
    Parameters
    ----------
    circumference : float
        Ring circumference in meters
    energy : float
        Beam energy in GeV
    momentum_compaction : float
        Momentum compaction factor
    energy_loss_per_turn : float
        Energy loss per turn in GeV
    harmonic_number : int
        Harmonic number
    damping_time : float
        Damping time in seconds
        
    Returns
    -------
    ring : dict
        Ring parameter dictionary with all ALBuMS-required attributes
    """
    try:
        c_light = 299792458  # m/s
        
        # Validate inputs
        if circumference <= 0:
            raise ValueError(f"Circumference must be > 0, got {circumference}")
        if energy <= 0:
            raise ValueError(f"Energy must be > 0, got {energy}")
        if momentum_compaction == 0:
            raise ValueError(f"Momentum compaction cannot be 0 (would cause division errors)")
        if harmonic_number <= 0:
            raise ValueError(f"Harmonic number must be > 0, got {harmonic_number}")
        if damping_time <= 0:
            raise ValueError(f"Damping time must be > 0, got {damping_time}")
        
        # Convert energy from GeV to eV for consistency
        energy_eV = energy * 1e9
        energy_loss_eV = max(energy_loss_per_turn * 1e9, 1e3)  # At least 1 keV loss
        
        # Calculate basic timing parameters
        circumference_c = circumference / c_light  # seconds
        revolution_time = circumference_c
        revolution_frequency = c_light / circumference
        
        # Create ring parameters dict
        ring = {
            # Basic parameters
            "circumference": circumference,
            "energy": energy,
            "energy_GeV": energy,
            "energy_eV": energy_eV,
            "momentum_compaction": momentum_compaction,
            "energy_loss_per_turn": energy_loss_per_turn,
            "energy_loss_per_turn_eV": energy_loss_eV,
            "harmonic_number": harmonic_number,
            "damping_time": damping_time,
            "revolution_time": revolution_time,
            "revolution_frequency": revolution_frequency,
            
            # ALBuMS-required attributes (E0, T0, T1, U0)
            "E0": energy_eV,                        # Beam energy in eV
            "T0": revolution_time,                  # Revolution period in seconds
            "T1": revolution_time / harmonic_number,  # RF period (T0 / h) - CRITICAL for tau_boundary calculation
            "U0": energy_loss_eV,                   # Energy loss per turn in eV
            "L": circumference,                      # Circumference in meters (required by BeamLoadingEquilibrium)
            
            # ALBuMS beam dynamics parameters
            "ac": momentum_compaction,              # Momentum compaction factor
            "h": harmonic_number,                   # Harmonic number
            
            # Frequency parameters
            "omega0": 2 * np.pi * revolution_frequency,  # Revolution angular frequency
            "f0": revolution_frequency,             # Revolution frequency
            "f1": harmonic_number * revolution_frequency,  # RF frequency
            "omega1": harmonic_number * 2 * np.pi * revolution_frequency,  # RF angular frequency
            
            # Beam dynamics - calculate from physics parameters
            # sigma_delta (relative momentum spread) - typical value for light source
            "sigma_delta": 0.001,  # ~0.1% momentum spread (typical for 2.75 GeV light source)
            
            # Synchrotron tune and frequency (rough estimate for beam loading initialization)
            # omega_s^2 ≈ alpha_c * h * e * Vc / (2*pi*E0) * omega0
            # For order-of-magnitude, assume Vc ~ 1 MV
            "omega0": 2 * np.pi * revolution_frequency,  # Revolution angular frequency
            "omega1": harmonic_number * 2 * np.pi * revolution_frequency,  # RF angular frequency
            
            # Natural bunch length: sigma_z = c * alpha_c * sigma_delta / omega_s
            # For SOLEIL II: sigma_z ~ 5 mm, sigma_tau ~ 16 ps
            # sigma_0 is RMS bunch length in SECONDS (used in gaussian_bunch)
            "sigma_0": 20e-12,  # ~20 ps RMS bunch length (typical for light source, will be refined by solver)
            
            "eta": abs(momentum_compaction) if momentum_compaction != 0 else 0.01,  # Slip factor
            "tau": [damping_time, damping_time, damping_time],  # Damping times for [x, y, z] planes
            "tune": [0.15, 0.15],  # Tunes [horizontal, vertical]
            "chro": [1.0, 1.0],  # Chromaticity
            "k1": harmonic_number * 2 * np.pi * revolution_frequency / c_light,  # RF wave number k1 = omega1 / c
            
            # Additional attributes for beam loading (BeamLoadingEquilibrium)
            "bunlen": 5e-3,  # Bunch length in meters (~5 mm RMS)
            "sync_tune": 0.003,  # Synchrotron tune (typical ~0.002-0.005 for storage ring)
        }
        
        return ring
    except Exception as e:
        print(f"Error creating ring: {e}")
        raise


def create_cavity_from_params(voltage, frequency, harmonic, Q, R_over_Q, QL=None, Rs=None):
    """
    Create a cavity parameters dictionary (compatible with ALBuMS).
    
    Includes all ALBuMS-required cavity attributes:
    - Vc: Cavity voltage (V)
    - Vg: Generator voltage (estimate)
    - Q: Quality factor (unloaded)
    - QL: Loaded quality factor
    - Rs: Series resistance (Ohms)
    - m: Harmonic number (for harmonic cavity)
    - psi: Cavity phase (will be set by ALBuMS)
    - theta: Cavity tuning angle (will be calculated)
    
    Parameters
    ----------
    voltage : float
        Cavity voltage in MV
    frequency : float
        Cavity frequency in MHz
    harmonic : int
        Harmonic number
    Q : float
        Unloaded quality factor (Q0)
    R_over_Q : float
        R/Q in Ohms
    QL : float, optional
        Loaded quality factor. If None, estimated as 0.9*Q
    Rs : float, optional
        Shunt impedance in Ohms. If None, calculated as R/Q * Q
        
    Returns
    -------
    cavity : dict
        Cavity parameter dictionary with all ALBuMS-required attributes
    """
    try:
        # Ensure voltage is not zero to avoid numerical issues
        # If voltage is 0, set it to a small value (1 kV = 0.001 MV)
        if voltage == 0 or voltage < 0.001:
            voltage = 0.001
            print(f"⚠️  Cavity voltage was {voltage}, set to minimum 0.001 MV (1 kV)")
        
        voltage_V = voltage * 1e6
        frequency_Hz = frequency * 1e6
        
        # Ensure frequency is valid
        if frequency <= 0:
            raise ValueError(f"Cavity frequency must be > 0, got {frequency}")
        
        # Ensure Q is valid
        if Q <= 0:
            raise ValueError(f"Quality factor Q must be > 0, got {Q}")
        
        # Ensure R/Q is valid
        if R_over_Q <= 0:
            raise ValueError(f"R/Q must be > 0, got {R_over_Q}")
        
        # Ensure harmonic is valid
        if harmonic <= 0:
            raise ValueError(f"Harmonic number must be > 0, got {harmonic}")
        
        # Use provided Rs if available, otherwise calculate from R/Q * Q
        if Rs is not None and Rs > 0:
            resistance = Rs * 1e6  # Convert from MΩ to Ω
        else:
            resistance = R_over_Q * Q
        
        # Use provided QL if available, otherwise estimate from Q
        # For SOLEIL II main cavity: Q0=35700, QL=6000 (heavily loaded)
        if QL is not None and QL > 0:
            loaded_QL = QL
        else:
            # Estimate QL as 0.9*Q for a slightly loaded cavity
            loaded_QL = max(0.5 * Q, 0.9 * Q)
        
        # Calculate coupling coefficient beta
        # beta = Q/QL - 1
        # IMPORTANT: Ensure beta > -1 to prevent division by zero in RL calculation
        beta = max(-0.99, Q / loaded_QL - 1)
        
        # Calculate loaded shunt impedance RL (Property from CavityResonator)
        # RL = Rs / (1 + beta)
        RL = resistance / (1.0 + beta)
        
        # Generator voltage must be positive and non-zero
        Vg = max(voltage_V * 1.1, 1000)  # At least 1000 V above cavity voltage or 1 kV minimum
        
        # Create cavity parameters dict
        cavity = {
            # Basic parameters
            "voltage": voltage,
            "voltage_MV": voltage,
            "voltage_V": voltage_V,
            "frequency": frequency,
            "frequency_MHz": frequency,
            "frequency_Hz": frequency_Hz,
            "harmonic": harmonic,
            "Q": Q,
            "QL": loaded_QL,                    # Loaded quality factor
            "beta": beta,                       # Coupling coefficient (required by BeamLoadingEquilibrium)
            "R_over_Q": R_over_Q,
            "resistance": resistance,
            "Rs": resistance,                   # Series resistance (same as resistance)
            "RL": RL,                           # Loaded shunt impedance (CRITICAL for Bosch method)
            
            # ALBuMS-required attributes
            "Vc": voltage_V,                    # Cavity voltage in V (required by ALBuMS)
            "Vg": Vg,                           # Generator voltage (safe positive value)
            "m": harmonic,                      # Harmonic number (for harmonic cavity)
            
            # Phase and angle parameters (will be set/calculated by ALBuMS)
            "psi": np.deg2rad(1.0),             # Cavity phase (initialize to 1 degree, NOT 0, to avoid numerical issues)
            "theta": 0.1,                       # Tuning angle (initialize to small angle, ~0.1 rad ≈ 5.7°, NOT π/2 to avoid cos=0)
            "theta_g": 0.0,                     # Generator angle
            "wr": frequency_Hz * 2 * np.pi,    # Angular frequency
            "detune": 0.0,                      # Detuning (initialize to 0)
        }
        return cavity
    except Exception as e:
        print(f"Error creating cavity: {e}")
        raise


def run_psi_current_scan(ring, main_cavity, harmonic_cavity, 
                         psi_range, current_range, 
                         method="Venturini", passive_hc=True,
                         progress_callback=None):
    """
    Run a psi vs current parameter scan.
    
    Supports both:
    1. Dict parameters (limited functionality)
    2. Real ALBuMS objects (full functionality) - RECOMMENDED
    
    This function will automatically try to convert dicts to real objects
    if mbtrack2 is available.
    
    Parameters
    ----------
    ring : dict or Synchrotron
        Ring configuration (dict or actual object)
    main_cavity : dict or CavityResonator
        Main cavity configuration (dict or actual object)
    harmonic_cavity : dict or CavityResonator
        Harmonic cavity configuration (dict or actual object)
    psi_range : tuple
        (min, max, num_points) for psi in degrees
    current_range : tuple
        (min, max, num_points) for current in A
    method : str
        Solution method ("Venturini", "Bosch", "Alves")
    passive_hc : bool
        Whether harmonic cavity is passive
    progress_callback : callable, optional
        Function to call with progress updates
        
    Returns
    -------
    results : dict
        Dictionary containing scan results
    """
    if not ALBUMS_AVAILABLE:
        return {
            "success": False,
            "error": "ALBuMS module not available. Core scientific computation requires full ALBuMS installation.",
            "status": "NOT_AVAILABLE",
            "hint": "Try: cd mbtrack2-stable && pip install -e ."
        }
    
    if scan_psi_I0 is None:
        return {
            "success": False,
            "error": "scan_psi_I0 function not available. This requires the full ALBuMS package installation.",
            "status": "FUNCTION_NOT_AVAILABLE"
        }
    
    try:
        # Normalize method parameter - ensure consistency with ALBuMS expectations
        method_map = {
            "Bosch": "Bosch",
            "bosch": "Bosch",
            "BOSCH": "Bosch",
            "Venturini": "Venturini",
            "venturini": "Venturini",
            "VENTURINI": "Venturini",
            "Alves": "Alves",
            "alves": "Alves",
            "ALVES": "Alves"
        }
        normalized_method = method_map.get(method, "Venturini")
        
        # Convert dicts to appropriate wrapper objects
        if isinstance(ring, dict):
            print("📦 Wrapping ring dict for ALBuMS compatibility...")
            ring = RingObject(ring)  # Use RingObject for ring with eta() method
        
        if isinstance(main_cavity, dict):
            print("📦 Wrapping main cavity dict for ALBuMS compatibility (CavityWrapper)...")
            main_cavity = CavityWrapper(main_cavity)
        
        if isinstance(harmonic_cavity, dict):
            print("📦 Wrapping harmonic cavity dict for ALBuMS compatibility (CavityWrapper)...")
            harmonic_cavity = CavityWrapper(harmonic_cavity)
        
        psi_vals = np.linspace(psi_range[0], psi_range[1], int(psi_range[2]))
        current_vals = np.linspace(current_range[0], current_range[1], int(current_range[2]))
        
        # psi_vals is in degrees. scan_psi_I0 expects DEGREES (it converts to rad internally)
        psi_vals_deg_for_scan = psi_vals 
        
        print(f"🔍 Running scan with ALBuMS (method={normalized_method})...")
        print(f"   Ring: {type(ring).__name__}")
        print(f"   MC: {type(main_cavity).__name__}")
        print(f"   HC: {type(harmonic_cavity).__name__}")
        print(f"   MC voltage: {main_cavity.get('voltage', 'N/A')} MV")
        print(f"   MC Vg: {main_cavity.get('Vg', 'N/A')} V")
        print(f"   HC voltage: {harmonic_cavity.get('voltage', 'N/A')} MV")
        print(f"   HC Vg: {harmonic_cavity.get('Vg', 'N/A')} V")
        print(f"   Ring ac: {ring.get('ac', 'N/A')}")
        print(f"   Ring sigma_delta: {ring.get('sigma_delta', 'N/A')}")
        print(f"   Psi range: {psi_vals[0]:.1f}° - {psi_vals[-1]:.1f}°")
        print(f"   Current range: {current_vals[0]:.4f} A - {current_vals[-1]:.4f} A")
        
        # Validate parameters before passing to ALBuMS
        is_valid, validation_issues = validate_parameters(
            ring if isinstance(ring, dict) else ring.__dict__,
            main_cavity if isinstance(main_cavity, dict) else main_cavity.__dict__,
            harmonic_cavity if isinstance(harmonic_cavity, dict) else harmonic_cavity.__dict__,
            current_vals[0] if len(current_vals) > 0 else 0.01
        )
        
        if not is_valid:
            error_list = "\n".join([f"  • {issue}" for issue in validation_issues])
            return {
                "success": False,
                "error": f"Parameter validation failed:\n{error_list}",
                "hint": "Check that all voltages are > 0, Q factors are reasonable, etc."
            }
        
        # Run the scan
        results = scan_psi_I0(
            name="streamlit_scan",
            MC=main_cavity,
            HC=harmonic_cavity,
            ring=ring,
            psi_HC_vals=psi_vals_deg_for_scan,
            currents=current_vals,
            method=normalized_method,
            save=False,
            passive_harmonic_cavity=passive_hc
        )
        
        # Debug: Check result structure
        print(f"\n📊 ALBuMS Scan Results:")
        if isinstance(results, tuple):
            print(f"   Result type: tuple with {len(results)} elements")
            for i, elem in enumerate(results):
                if hasattr(elem, 'shape'):
                    valid_count = np.sum(np.isfinite(elem.flatten()))
                    total_count = elem.size
                    print(f"   [{i}] shape={elem.shape}, dtype={elem.dtype}, valid={valid_count}/{total_count}")
                else:
                    print(f"   [{i}] type={type(elem).__name__}")
        
        # Convert tuple results to dict for easier access
        # results tuple: (zero_freq_coup, robinson_coup, modes_coup, HOM_coup, 
        #                 converged_coup, PTBL_coup, bl, xi, R)
        fallback_used = False
        if isinstance(results, tuple) and len(results) == 9:
            results_dict = {
                'zero_freq_coup': results[0],
                'robinson_coup': results[1],
                'modes_coup': results[2],
                'HOM_coup': results[3],
                'converged_coup': results[4],
                'PTBL_coup': results[5],
                'bl': results[6],
                'xi': results[7],
                'R': results[8]
            }
            
            # Check convergence rate - if Venturini failed for most points, try Bosch
            if normalized_method == "Venturini":
                converged = results[4]  # converged_coup
                if hasattr(converged, 'size'):
                    convergence_rate = np.sum(converged) / converged.size if converged.size > 0 else 0
                    print(f"   Convergence rate: {convergence_rate*100:.1f}%")
                    
                    # If less than 20% converged, try Bosch as fallback
                    if convergence_rate < 0.2:
                        print(f"\n⚠️ Low Venturini convergence ({convergence_rate*100:.1f}%). Trying Bosch fallback...")
                        try:
                            fallback_results = scan_psi_I0(
                                name="streamlit_scan_fallback",
                                MC=main_cavity,
                                HC=harmonic_cavity,
                                ring=ring,
                                psi_HC_vals=psi_vals_deg_for_scan,
                                currents=current_vals,
                                method="Bosch",
                                save=False,
                                passive_harmonic_cavity=passive_hc
                            )
                            
                            if isinstance(fallback_results, tuple) and len(fallback_results) == 9:
                                fallback_converged = fallback_results[4]
                                if hasattr(fallback_converged, 'size'):
                                    fallback_rate = np.sum(fallback_converged) / fallback_converged.size if fallback_converged.size > 0 else 0
                                    print(f"   Bosch convergence rate: {fallback_rate*100:.1f}%")
                                    
                                    # Use Bosch results if they have better convergence
                                    if fallback_rate > convergence_rate:
                                        print(f"   ✅ Using Bosch results (better convergence)")
                                        results_dict = {
                                            'zero_freq_coup': fallback_results[0],
                                            'robinson_coup': fallback_results[1],
                                            'modes_coup': fallback_results[2],
                                            'HOM_coup': fallback_results[3],
                                            'converged_coup': fallback_results[4],
                                            'PTBL_coup': fallback_results[5],
                                            'bl': fallback_results[6],
                                            'xi': fallback_results[7],
                                            'R': fallback_results[8]
                                        }
                                        fallback_used = True
                        except Exception as fallback_error:
                            print(f"   ❌ Bosch fallback failed: {fallback_error}")
        else:
            results_dict = results if isinstance(results, dict) else {}
        
        return {
            "psi_vals": psi_vals,
            "current_vals": current_vals,
            "results": results_dict,
            "success": True,
            "fallback_used": fallback_used,
            "method_note": "Fallback to Bosch method was used due to Venturini convergence issues" if fallback_used else None
        }
    except ZeroDivisionError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Division by zero error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        # Try to extract more info
        try:
            print(f"\n📋 Debug Info:")
            print(f"   Main cavity type: {type(main_cavity).__name__}")
            print(f"   HC type: {type(harmonic_cavity).__name__}")
            print(f"   Ring type: {type(ring).__name__}")
            if hasattr(main_cavity, 'get'):
                print(f"   MC Vg: {main_cavity.get('Vg', 'N/A')}")
            if hasattr(ring, 'get'):
                print(f"   Ring ac: {ring.get('ac', 'N/A')}")
                print(f"   Ring sigma_delta: {ring.get('sigma_delta', 'N/A')}")
        except:
            pass
        
        return {
            "success": False,
            "error": f"Division by zero during scan: {error_str}",
            "traceback": tb_str,
            "hint": "This usually happens when Vg, ac, or sigma_delta become 0. Check cavity and ring parameters."
        }
    except AttributeError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Attribute error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        if "'dict' object has no attribute" in error_str:
            # This was the original error - dict compatibility issue
            return {
                "success": False,
                "error": f"Dict compatibility error: {error_str}. Parameters must be wrapped for ALBuMS.",
                "hint": "Check that DictObject wrapper is properly initialized",
                "traceback": tb_str
            }
        else:
            return {
                "success": False,
                "error": f"Scan error: {error_str}",
                "traceback": tb_str
            }
    except ValueError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Value error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        return {
            "success": False,
            "error": f"Value error during scan: {error_str}",
            "traceback": tb_str
        }
    except NotImplementedError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Not implemented error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        # Provide helpful hint for Alves method incompatibility
        if "Alves method with active harmonic cavities" in error_str:
            hint = "The Alves method has compatibility issues with this cavity configuration. Try using 'Venturini' or 'Bosch' method instead."
        else:
            hint = "This feature is not yet implemented."
        
        return {
            "success": False,
            "error": f"Not implemented: {error_str}",
            "hint": hint,
            "traceback": tb_str
        }
    except Exception as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Unexpected error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        return {
            "success": False,
            "error": f"Scan error: {error_str}",
            "traceback": tb_str
        }


def run_psi_roq_scan(ring, main_cavity, harmonic_cavity, current,
                     psi_range, roq_range, 
                     method="Venturini", passive_hc=True):
    """
    Run a psi vs R/Q parameter scan.
    
    Supports both:
    1. Dict parameters (limited functionality)
    2. Real ALBuMS objects (full functionality) - RECOMMENDED
    
    Parameters
    ----------
    ring : dict or Synchrotron
        Ring configuration (dict or actual object)
    main_cavity : dict or CavityResonator
        Main cavity configuration (dict or actual object)
    harmonic_cavity : dict or CavityResonator
        Harmonic cavity configuration (dict or actual object)
    current : float
        Beam current in A
    psi_range : tuple
        (min, max, num_points) for psi in degrees
    roq_range : tuple
        (min, max, num_points) for R/Q in Ohms
    method : str
        Solution method
    passive_hc : bool
        Whether harmonic cavity is passive
        
    Returns
    -------
    results : dict
        Dictionary containing scan results
    """
    if not ALBUMS_AVAILABLE:
        return {
            "success": False,
            "error": "ALBuMS module not available. Core scientific computation requires full ALBuMS installation.",
            "status": "NOT_AVAILABLE",
            "hint": "Try: cd mbtrack2-stable && pip install -e ."
        }
    
    if scan_psi_RoQ is None:
        return {
            "success": False,
            "error": "scan_psi_RoQ function not available. This requires the full ALBuMS package installation.",
            "status": "FUNCTION_NOT_AVAILABLE"
        }
    
    try:
        # Normalize method parameter - ensure consistency with ALBuMS expectations
        method_map = {
            "Bosch": "Bosch",
            "bosch": "Bosch",
            "BOSCH": "Bosch",
            "Venturini": "Venturini",
            "venturini": "Venturini",
            "VENTURINI": "Venturini",
            "Alves": "Alves",
            "alves": "Alves",
            "ALVES": "Alves"
        }
        normalized_method = method_map.get(method, "Venturini")
        
        # Convert dicts to appropriate wrapper objects
        if isinstance(ring, dict):
            print("📦 Wrapping ring dict for ALBuMS compatibility...")
            ring = RingObject(ring)  # Use RingObject for ring with eta() method
        
        if isinstance(main_cavity, dict):
            print("📦 Wrapping main cavity dict for ALBuMS compatibility (CavityWrapper)...")
            main_cavity = CavityWrapper(main_cavity)
        
        if isinstance(harmonic_cavity, dict):
            print("📦 Wrapping harmonic cavity dict for ALBuMS compatibility (CavityWrapper)...")
            harmonic_cavity = CavityWrapper(harmonic_cavity)
        
        psi_vals = np.linspace(psi_range[0], psi_range[1], int(psi_range[2]))
        roq_vals = np.linspace(roq_range[0], roq_range[1], int(roq_range[2]))
        
        psi_vals_deg_for_scan = psi_vals # scan_psi_RoQ expects degrees

        
        print(f"🔍 Running R/Q scan with ALBuMS (method={normalized_method})...")
        
        results = scan_psi_RoQ(
            name="streamlit_roq_scan",
            MC=main_cavity,
            HC=harmonic_cavity,
            ring=ring,
            psi_HC_vals=psi_vals_deg_for_scan,
            RoQ_vals=roq_vals,
            I0=current,
            method=normalized_method,
            save=False,
            passive_harmonic_cavity=passive_hc
        )
        
        # Convert tuple results to dict for easier access
        # results tuple: (zero_freq_coup, robinson_coup, modes_coup, HOM_coup, 
        #                 converged_coup, PTBL_coup, bl, xi, R)
        if isinstance(results, tuple) and len(results) == 9:
            results_dict = {
                'zero_freq_coup': results[0],
                'robinson_coup': results[1],
                'modes_coup': results[2],
                'HOM_coup': results[3],
                'converged_coup': results[4],
                'PTBL_coup': results[5],
                'bl': results[6],
                'xi': results[7],
                'R': results[8]
            }
        else:
            results_dict = results if isinstance(results, dict) else {}
        
        return {
            "psi_vals": psi_vals,
            "roq_vals": roq_vals,
            "results": results_dict,
            "success": True
        }
    except ZeroDivisionError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Division by zero error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        return {
            "success": False,
            "error": f"Division by zero during scan: {error_str}",
            "traceback": tb_str
        }
    except ValueError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Value error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        return {
            "success": False,
            "error": f"Value error during scan: {error_str}",
            "traceback": tb_str
        }
    except NotImplementedError as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Not implemented error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        # Provide helpful hint for Alves method incompatibility
        if "Alves method with active harmonic cavities" in error_str:
            hint = "The Alves method has compatibility issues with this cavity configuration. Try using 'Venturini' or 'Bosch' method instead."
        else:
            hint = "This feature is not yet implemented."
        
        return {
            "success": False,
            "error": f"Not implemented: {error_str}",
            "hint": hint,
            "traceback": tb_str
        }
    except Exception as e:
        import traceback
        error_str = str(e)
        tb_str = traceback.format_exc()
        print(f"❌ Unexpected error: {error_str}")
        print(f"Full traceback:\n{tb_str}")
        
        return {
            "success": False,
            "error": f"Scan error: {error_str}",
            "traceback": tb_str
        }


def run_optimization(ring, main_cavity, harmonic_cavity, current,
                    psi0, bounds, method="Venturini", 
                    equilibrium_only=False):
    """
    Optimize R-factor by finding optimal psi.
    
    Parameters
    ----------
    ring : Synchrotron
        Ring configuration
    main_cavity : CavityResonator
        Main cavity configuration
    harmonic_cavity : CavityResonator
        Harmonic cavity configuration
    current : float
        Beam current in A
    psi0 : float
        Initial guess for psi in degrees
    bounds : tuple
        (min, max) bounds for psi in degrees
    method : str
        Solution method
    equilibrium_only : bool
        If True, only solve equilibrium (faster)
        
    Returns
    -------
    results : dict
        Optimization results
    """
    if not ALBUMS_AVAILABLE:
        return {
            "success": False,
            "error": "ALBuMS module not available. Please check installation."
        }
    
    if maximize_R is None or maximize_R_equilibrium is None:
        return {
            "success": False,
            "error": "Optimization functions not available"
        }
    
    try:
        # Convert dicts to specialize wrappers for ALBuMS compatibility
        if isinstance(ring, dict):
            ring = RingObject(ring)
        
        if isinstance(main_cavity, dict):
            main_cavity = CavityWrapper(main_cavity)
        
        if isinstance(harmonic_cavity, dict):
            harmonic_cavity = CavityWrapper(harmonic_cavity)
        
        psi0_rad = np.deg2rad(psi0)
        bounds_rad = (np.deg2rad(bounds[0]), np.deg2rad(bounds[1]))
        
        if equilibrium_only:
            optimal_psi_rad = maximize_R_equilibrium(
                ring=ring,
                MC=main_cavity,
                HC=harmonic_cavity,
                I0=current,
                psi0_HC=psi0_rad,
                tau_boundary=None,
                method=method,
                bounds=bounds_rad
            )
        else:
            optimal_psi_rad = maximize_R(
                ring=ring,
                MC=main_cavity,
                HC=harmonic_cavity,
                I0=current,
                psi0_HC=psi0_rad,
                tau_boundary=None,
                method=method,
                bounds=bounds_rad
            )
        
        optimal_psi = np.rad2deg(optimal_psi_rad)
        
        # Calculate R-factor at optimal psi
        if RobinsonModes is not None:
            B = RobinsonModes(ring, [main_cavity, harmonic_cavity], current)
            B.solve(passive_harmonic_cavity=True, method=method)
            r_factor = B.R_factor(method)
        else:
            r_factor = None
        
        return {
            "success": True,
            "optimal_psi": optimal_psi,
            "r_factor": r_factor,
            "psi0": psi0
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Optimization error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        }


def analyze_robinson_modes(ring, main_cavity, harmonic_cavity, current,
                          psi_range, method="Venturini", passive_hc=True):
    """
    Analyze Robinson modes over a range of psi values.
    
    Parameters
    ----------
    ring : Synchrotron
        Ring configuration
    main_cavity : CavityResonator
        Main cavity configuration
    harmonic_cavity : CavityResonator
        Harmonic cavity configuration
    current : float
        Beam current in A
    psi_range : tuple
        (min, max, num_points) for psi in degrees
    method : str
        Solution method ("Venturini", "Bosch", or "Alves")
    passive_hc : bool
        Whether harmonic cavity is passive
        
    Returns
    -------
    results : dict
        Mode analysis results
    """
    if not ALBUMS_AVAILABLE:
        return {
            "success": False,
            "error": "ALBuMS module not available. Please check installation."
        }
    
    if scan_modes is None:
        return {
            "success": False,
            "error": "scan_modes function not available"
        }
    
    try:
        # Normalize method parameter - ensure consistency with ALBuMS expectations
        method_map = {
            "Bosch": "Bosch",
            "bosch": "Bosch",
            "BOSCH": "Bosch",
            "Venturini": "Venturini",
            "venturini": "Venturini",
            "VENTURINI": "Venturini",
            "Alves": "Alves",
            "alves": "Alves",
            "ALVES": "Alves"
        }
        normalized_method = method_map.get(method, "Venturini")
        print(f"📌 Normalized method: {method} → {normalized_method}")
        
        # Convert dicts to appropriate wrapper objects
        if isinstance(ring, dict):
            print("📦 Wrapping ring dict for ALBuMS compatibility...")
            ring = RingObject(ring)  # Use RingObject for ring with eta() method
        
        if isinstance(main_cavity, dict):
            print("📦 Wrapping main cavity dict for ALBuMS compatibility...")
            main_cavity = DictObject(main_cavity)
        
        if isinstance(harmonic_cavity, dict):
            print("📦 Wrapping harmonic cavity dict for ALBuMS compatibility...")
            harmonic_cavity = DictObject(harmonic_cavity)
        
        psi_vals = np.linspace(psi_range[0], psi_range[1], int(psi_range[2]))
        psi_vals_rad = np.deg2rad(psi_vals)
        
        print(f"🔍 Running mode analysis with ALBuMS using {normalized_method} method...")
        
        results = scan_modes(
            MC=main_cavity,
            HC=harmonic_cavity,
            ring=ring,
            psi_HC_vals=psi_vals_rad,
            current=current,
            mode_coupling=True,
            tau_boundary=None,
            method=normalized_method,
            passive_harmonic_cavity=passive_hc
        )
        
        return {
            "success": True,
            "psi_vals": psi_vals,
            "results": results
        }
    except Exception as e:
        print(f"❌ Mode analysis error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "success": False,
            "error": f"Mode analysis error: {str(e)}"
        }
