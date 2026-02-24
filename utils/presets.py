"""
Predefined configurations for common accelerator lattices and cavity setups.
"""

from typing import List, Dict, Any

# Aladdin lattice parameters (from benchmark examples)
ALADDIN_RING = {
    "name": "Aladdin",
    "circumference": 96.0,  # meters
    "energy": 1.0,  # GeV
    "momentum_compaction": 0.0165,
    "energy_loss_per_turn": 0.0001,  # GeV
    "harmonic_number": 32,
    "damping_time": 0.01,  # seconds
}

ALADDIN_MAIN_CAVITY = {
    "voltage": 1.7,  # MV (increased to 1.7 MV as requested)
    "frequency": 499.654,  # MHz
    "harmonic": 32,
    "Q0": 40000,  # Unloaded Q
    "Q": 40000,  # Q0 (for backward compatibility)
    "QL": 6667,  # Loaded Q (estimated)
    "Q_ext": 10000,  # External Q (estimated)
    "R_over_Q": 100,  # Ohms
    "Rs": 4.0,  # Shunt impedance Rs = 4 MΩ / cav (estimated)
    "Ncav": 1,  # Number of cavities
    "beta": 5.0,  # Coupling factor β (estimated)
    "detuning_khz": 0.0,  # Detuning frequency (kHz)
    "rf_feedback_gain": 1.0,  # RF feedback gain (estimated)
    "tau_us": 4.0,  # Cavity decay time (μs, estimated)
}

ALADDIN_HARMONIC_CAVITY_PASSIVE = {
    "voltage": 0.15,  # MV (150 kV - reasonable for passive cavity)
    "frequency": 1498.962,  # MHz (3rd harmonic)
    "harmonic": 96,
    "harmonic_number": 3,  # 3rd harmonic of main RF
    "Q0": 20000,  # Unloaded Q
    "Q": 20000,  # Q0 (for backward compatibility)
    "R_over_Q": 50,  # Ohms
    "Rs": 1.0,  # Shunt impedance Rs = 1 MΩ / cav (estimated)
    "Ncav": 1,  # Number of cavities
    "beta": 0.0,  # Coupling factor β (passive cavity)
    "tau_us": 5.0,  # Cavity decay time (μs, estimated)
}

ALADDIN_HARMONIC_CAVITY_ACTIVE = {
    "voltage": 0.3,  # MV
    "frequency": 1498.962,  # MHz (3rd harmonic)
    "harmonic": 96,
    "harmonic_number": 3,  # 3rd harmonic of main RF
    "Q0": 20000,  # Unloaded Q
    "Q": 20000,  # Q0 (for backward compatibility)
    "R_over_Q": 50,  # Ohms
    "Rs": 1.0,  # Shunt impedance Rs = 1 MΩ / cav (estimated)
    "Ncav": 1,  # Number of cavities
    "beta": 1.0,  # Coupling factor β (active cavity)
    "tau_us": 5.0,  # Cavity decay time (μs, estimated)
}

# SOLEIL II parameters (from SOLEIL II Lattice Review, January 2026)
# Reference: P. Borowiec & al., 28th ESLS RF Workshops – ALBA Synchrotron, Spain, October 2025
# SOLEIL II parameters
SOLEIL_II_RF_FREQ = 352.0  # MHz (Main cavity frequency as requested)
SOLEIL_II_HC_FREQ = 1408.0  # MHz (4th harmonic = 1.408 GHz)

# Ring parameters for different ID configurations
SOLEIL_II_RING_OPEN = {
    "name": "SOLEIL II (Open IDs)",
    "circumference": 353.98,
    "frequency": SOLEIL_II_RF_FREQ,
    "energy": 2.75,  # GeV
    "momentum_compaction": 1.06e-4, 
    "energy_loss_per_turn": 0.000471,  # 471 keV (V3633 bare lattice)
    "harmonic_number": 416,
    "damping_time": 0.0115,  # 11.5 ms
    "energy_spread": 9.3e-4,    # 0.093%
    "sigma_z0": 9.0e-12,        # 9.0 ps (V3633 bare lattice)
}

SOLEIL_II_RING_CLOSED = {
    "name": "SOLEIL II (Closed IDs)",
    "circumference": 353.98,
    "frequency": SOLEIL_II_RF_FREQ,
    "energy": 2.75,  # GeV
    "momentum_compaction": 1.06e-4, 
    "energy_loss_per_turn": 0.000787,  # 787 keV (471 keV bare + 316 keV IDs)
    "harmonic_number": 416,
    "damping_time": 0.0059,  # 5.9 ms
    "energy_spread": 8.55e-4,
    "sigma_z0": 9.0e-12,
}

# Diamond parameters
DIAMOND_RING = {
    "name": "Diamond",
    "circumference": 561.6,
    "frequency": 499.680,
    "energy": 3.0,
    "momentum_compaction": 1.7e-4,
    "energy_loss_per_turn": 0.0010, # 1.0 MeV (without IDs)
    "harmonic_number": 936,
}

# ALBA parameters
ALBA_RING = {
    "name": "ALBA",
    "circumference": 268.8,
    "frequency": 499.654,
    "energy": 3.0,
    "momentum_compaction": 8.8e-4,
    "energy_loss_per_turn": 0.0013, # 1.3 MeV
    "harmonic_number": 448,
}

# PETRA III parameters
PETRA_III_RING = {
    "name": "PETRA III",
    "circumference": 2304.0,
    "frequency": 499.564,
    "energy": 6.0,
    "momentum_compaction": 1.2e-4,
    "energy_loss_per_turn": 0.004660, # 4.66 MeV
    "harmonic_number": 3840,
}

# ESRF-EBS parameters
ESRF_EBS_RING = {
    "name": "ESRF-EBS",
    "circumference": 843.977,
    "frequency": 352.202,
    "energy": 6.0,
    "momentum_compaction": 8.5e-5,
    "energy_loss_per_turn": 0.0025, # 2.5 MeV
    "harmonic_number": 992,
}

# SLS 2.0 parameters
SLS_2_0_RING = {
    "name": "SLS 2.0",
    "circumference": 290.4,
    "frequency": 499.654,
    "energy": 2.7,
    "momentum_compaction": 1.33e-4, # Nominal magnitude
    "energy_loss_per_turn": 0.00085, # ~850 keV
    "harmonic_number": 484,
}

# BESSY II parameters
BESSY_II_RING = {
    "name": "BESSY II",
    "circumference": 240.0,
    "frequency": 499.654,
    "energy": 1.7,
    "momentum_compaction": 1.6e-3,
    "energy_loss_per_turn": 0.000178, # 178 keV
    "harmonic_number": 400,
}

# SOLEIL (Current) parameters
SOLEIL_NOMINAL_RING = {
    "name": "SOLEIL (Nominal)",
    "circumference": 354.1,
    "frequency": 352.197,
    "energy": 2.75,
    "momentum_compaction": 4.38e-4,
    "energy_loss_per_turn": 0.000944, # 944 keV (with IDs)
    "harmonic_number": 416,
    "damping_time": 0.0066,           # 6.6 ms (Present)
    "sigma_z0": 15.2e-12,             # 15.2 ps (SOLEIL 1)
}

# MAX IV 3 GeV parameters
MAX_IV_3GEV_RING_BARE = {
    "name": "MAX IV 3 GeV (Bare)",
    "circumference": 528.0,
    "frequency": 99.931,
    "energy": 3.0,
    "momentum_compaction": 3.06e-4,
    "energy_loss_per_turn": 0.0003638, # 363.8 keV
    "harmonic_number": 176,
    "damping_time": 0.025194,
    "energy_spread": 7.69e-4,
    "sigma_z0": 3.37e-11, # 10.1 mm / c
}

MAX_IV_3GEV_RING_LOADED = {
    "name": "MAX IV 3 GeV (Updated)",
    "circumference": 528.0,
    "frequency": 99.931,
    "energy": 3.0,
    "momentum_compaction": 3.06e-4,
    "energy_loss_per_turn": 0.0003638, # 363.8 keV
    "harmonic_number": 176,
    "damping_time": 0.025194,
    "energy_spread": 7.69e-4,
    "sigma_z0": 3.37e-11, # 10.1 mm / c
}

# Elettra (2.0 GeV)
ELETTRA_RING = {
    "name": "Elettra (2.0 GeV)",
    "circumference": 259.2,
    "frequency": 499.654,
    "energy": 2.0,
    "momentum_compaction": 1.6e-3,
    "energy_loss_per_turn": 0.0002557, # 255.7 keV
    "harmonic_number": 432,
}

# Default RING (pointing to Closed IDs as it was before, but with fixed values)
SOLEIL_II_RING = SOLEIL_II_RING_CLOSED

# ESRF-EBS Normal Conducting (NC) 352 MHz main cavities (MC)
# Parameters for the TOTAL system of 4 cavities
SOLEIL_II_MAIN_CAVITY = {
    "voltage": 1.7,  # MV (Vrf = 1.7 MV total)
    "frequency": SOLEIL_II_RF_FREQ,
    "harmonic": 416,
    "Q0": 35700,
    "Q": 35700,
    "QL": 6000,
    "Q_ext": 6364,
    "Rs": 20.0,  # Total Rs = 5 MΩ/cav * 4 cav = 20 MΩ
    "R_over_Q": 560.0,  # Total R/Q = 140 Ω/cav * 4 cav = 560 Ω
    "Ncav": 4,
    "beta": 5.5,
    "detuning_khz": 0.0,
    "rf_feedback_gain": 1.3,
    "tau_us": 4.87,
}

# Passive Normal Conducting (NC) harmonic cavities (HC)
# Parameters for the TOTAL system of 2 cavities
SOLEIL_II_HARMONIC_CAVITY_2HC = {
    "voltage": 0.7,  # 350 kV/cav * 2 cav = 0.7 MV
    "frequency": SOLEIL_II_HC_FREQ,
    "harmonic": 1664,
    "harmonic_number": 4,
    "Q0": 31000,
    "Q": 31000,
    "R_over_Q": 59.2,  # 29.6 Ω/cav * 2 cav = 59.2 Ω
    "Rs": 1.84,  # 0.92 MΩ/cav * 2 cav = 1.84 MΩ
    "Ncav": 2,
    "beta": 0.0,
    "tau_us": 7.0,
}

# Parameters for the TOTAL system of 3 cavities
SOLEIL_II_HARMONIC_CAVITY_3HC = {
    "voltage": 1.05,  # 350 kV/cav * 3 cav = 1.05 MV
    "frequency": SOLEIL_II_HC_FREQ,
    "harmonic": 1664,
    "harmonic_number": 4,
    "Q0": 31000,
    "Q": 31000,
    "R_over_Q": 88.8,  # 29.6 Ω/cav * 3 cav = 88.8 Ω
    "Rs": 2.76,  # 0.92 MΩ/cav * 3 cav = 2.76 MΩ
    "Ncav": 3,
    "beta": 0.0,
    "tau_us": 7.0,
}

# Default HC
SOLEIL_II_HARMONIC_CAVITY = SOLEIL_II_HARMONIC_CAVITY_2HC

# Preset configurations
PRESETS = {
    "Aladdin (Passive HC)": {
        "ring": ALADDIN_RING,
        "main_cavity": ALADDIN_MAIN_CAVITY,
        "harmonic_cavity": ALADDIN_HARMONIC_CAVITY_PASSIVE,
        "current": 0.2,  # A
        "passive_hc": True,
        # Scan parameter defaults
        "scan_params": {
            "psi_min": 60.0,  # degrees
            "psi_max": 90.0,  # degrees
            "psi_points": 30,
        }
    },
    "Aladdin (Active HC)": {
        "ring": ALADDIN_RING,
        "main_cavity": ALADDIN_MAIN_CAVITY,
        "harmonic_cavity": ALADDIN_HARMONIC_CAVITY_ACTIVE,
        "current": 0.2,  # A
        "passive_hc": False,
        # Scan parameter defaults
        "scan_params": {
            "psi_min": 60.0,  # degrees
            "psi_max": 90.0,  # degrees
            "psi_points": 30,
        }
    },
    "SOLEIL II (Open IDs, 2 HC)": {
        "ring": SOLEIL_II_RING_OPEN,
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY_2HC,
        "current": 0.5,
        "passive_hc": True,
        "scan_params": {
            "psi_min": 60.0,
            "psi_max": 100.0,
            "psi_points": 60,
        }
    },
    "SOLEIL II (Open IDs, 3 HC)": {
        "ring": SOLEIL_II_RING_OPEN,
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY_3HC,
        "current": 0.5,
        "passive_hc": True,
        "scan_params": {
            "psi_min": 60.0,
            "psi_max": 100.0,
            "psi_points": 60,
        }
    },
    "SOLEIL II (Closed IDs, 2 HC)": {
        "ring": SOLEIL_II_RING_CLOSED,
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY_2HC,
        "current": 0.5,
        "passive_hc": True,
        "scan_params": {
            "psi_min": 60.0,
            "psi_max": 100.0,
            "psi_points": 60,
        }
    },
    "SOLEIL II (Closed IDs, 3 HC)": {
        "ring": SOLEIL_II_RING_CLOSED,
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY_3HC,
        "current": 0.5,
        "passive_hc": True,
        "scan_params": {
            "psi_min": 60.0,
            "psi_max": 100.0,
            "psi_points": 60,
        }
    },
    "SOLEIL II": {
        "ring": SOLEIL_II_RING_OPEN,
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY_3HC,
        "current": 0.5,
        "passive_hc": True,
        "scan_params": {
            "psi_min": 60.0,
            "psi_max": 100.0,
            "psi_points": 60,
        }
    },
    "SOLEIL (Nominal)": {
        "ring": SOLEIL_NOMINAL_RING,
        "main_cavity": {
            "voltage": 1.8,
            "frequency": 352.197,
            "harmonic": 416,
            "Q0": 35000,
            "Rs": 5.0,
            "Ncav": 2,
            "beta": 5.0,
        },
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY_2HC,
        "current": 0.5,
        "passive_hc": True,
        "scan_params": {
            "psi_min": 60.0,
            "psi_max": 100.0,
            "psi_points": 60,
        }
    },
    "ESRF-EBS": {
        "ring": ESRF_EBS_RING,
        "main_cavity": {
            "voltage": 6.0,
            "frequency": 352.202,
            "harmonic": 992,
            "Q0": 35000,
            "Rs": 1.5,
            "Ncav": 13,
            "beta": 5.0,
        },
        "harmonic_cavity": {
            "voltage": 1.0,
            "frequency": 1408.8,
            "harmonic_number": 4,
            "Q0": 30000,
            "Rs": 1.0,
            "Ncav": 3,
        },
        "current": 0.2,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 100.0, "psi_points": 60}
    },
    "SLS 2.0": {
        "ring": SLS_2_0_RING,
        "main_cavity": {
            "voltage": 2.6,
            "frequency": 499.654,
            "harmonic": 484,
            "Q0": 40000,
            "Rs": 1.57,
            "Ncav": 4,
            "beta": 3.0,
        },
        "harmonic_cavity": {
            "voltage": 0.5,
            "frequency": 1498.962,
            "harmonic_number": 3,
            "Q0": 1e9, # Superconducting
            "Rs": 1000.0,
            "Ncav": 1,
            "beta": 0.0,
        },
        "current": 0.4,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 110.0, "psi_points": 100}
    },
    "BESSY II": {
        "ring": BESSY_II_RING,
        "main_cavity": {
            "voltage": 2.0,
            "frequency": 499.654,
            "harmonic": 400,
            "Q0": 30000,
            "Rs": 3.0,
            "Ncav": 4,
            "beta": 2.0,
        },
        "harmonic_cavity": {
            "voltage": 0.4,
            "frequency": 1498.962,
            "harmonic_number": 3,
            "Q0": 30000,
            "Rs": 1.0,
            "Ncav": 1,
        },
        "current": 0.3,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 100.0, "psi_points": 60}
    },
    "PETRA III": {
        "ring": PETRA_III_RING,
        "main_cavity": {
            "voltage": 20.0,
            "frequency": 499.564,
            "harmonic": 3840,
            "Q0": 30000,
            "Rs": 3.0,
            "Ncav": 12,
            "beta": 2.0,
        },
        "harmonic_cavity": {
            "voltage": 4.0,
            "frequency": 1498.692,
            "harmonic": 11520,
            "harmonic_number": 3,
            "Q0": 30000,
            "Rs": 1.0,
            "Ncav": 4,
        },
        "current": 0.1,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 100.0, "psi_points": 60}
    },
    "MAX IV 3 GeV": {
        "ring": MAX_IV_3GEV_RING_LOADED,
        "main_cavity": {
            "voltage": 1.397, # MV
            "frequency": 99.931,
            "harmonic": 176,
            "Q0": 40000,
            "Rs": 3.14, # Est for 2 cavities
            "Ncav": 2,
            "beta": 3.0,
        },
        "harmonic_cavity": {
            "voltage": 0.448, # MV
            "frequency": 299.793,
            "harmonic": 528,
            "harmonic_number": 3,
            "Q0": 20800,
            "Rs": 8.25,
            "R_over_Q": 396.0,
            "Ncav": 3,
            "beta": 0.0,
            "detuning_khz": 75.0,
        },
        "current": 0.3, # 300 mA
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 110.0, "psi_points": 100}
    },
    "ALBA": {
        "ring": ALBA_RING,
        "main_cavity": {
            "voltage": 3.6,
            "frequency": 499.654,
            "harmonic": 448,
            "Q0": 30000,
            "Rs": 30.0, # 5 MOhms * 6
            "Ncav": 6,
            "beta": 3.0,
        },
        "harmonic_cavity": {
            "voltage": 0.3,
            "frequency": 1498.962,
            "harmonic_number": 3,
            "Q0": 25000,
            "Rs": 1.0,
            "Ncav": 1,
        },
        "current": 0.4,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 100.0, "psi_points": 60}
    },
    "Diamond": {
        "ring": DIAMOND_RING,
        "main_cavity": {
            "voltage": 3.0,
            "frequency": 499.680,
            "harmonic": 936,
            "Q0": 1e9, # Superconducting
            "Rs": 500.0, 
            "Ncav": 2,
            "beta": 10.0,
        },
        "harmonic_cavity": {
            "voltage": 0.8,
            "frequency": 1499.04,
            "harmonic_number": 3,
            "Q0": 30000,
            "Rs": 1.0,
            "Ncav": 2,
        },
        "current": 0.3,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 100.0, "psi_points": 60}
    },
    "Elettra (2.0 GeV)": {
        "ring": ELETTRA_RING,
        "main_cavity": {
            "voltage": 1.6,
            "frequency": 499.654,
            "harmonic": 432,
            "Q0": 30000,
            "Rs": 4.0,
            "Ncav": 4,
            "beta": 3.0,
        },
        "harmonic_cavity": {
            "voltage": 0.2,
            "frequency": 1498.962,
            "harmonic_number": 3,
            "Q0": 25000,
            "Rs": 1.0,
            "Ncav": 1,
        },
        "current": 0.3,
        "passive_hc": True,
        "scan_params": {"psi_min": 60.0, "psi_max": 100.0, "psi_points": 60}
    },
    "Custom": {
        "ring": {
            "name": "Custom",
            "circumference": 100.0,
            "energy": 1.0,
            "momentum_compaction": 0.01,
            "energy_loss_per_turn": 0.0001,
            "harmonic_number": 100,
            "damping_time": 0.01,
        },
        "main_cavity": {
            "voltage": 1.7,  # MV
            "frequency": 500.0,
            "harmonic": 100,
            "Q": 40000,
            "R_over_Q": 100,
        },
        "harmonic_cavity": {
            "voltage": 0.1,  # kV (0.0001 MV would be too small; 0.1 is reasonable minimum)
            "frequency": 1500.0,
            "harmonic": 300,
            "Q": 20000,
            "R_over_Q": 50,
        },
        "current": 0.2,
        "passive_hc": False,
        # Scan parameter defaults
        "scan_params": {
            "psi_min": 60.0,  # degrees
            "psi_max": 90.0,  # degrees
            "psi_points": 30,
        }
    }
}

def get_preset(name):
    """Get a preset configuration by name."""
    return PRESETS.get(name, PRESETS["Custom"])

def get_preset_names():
    """Get list of available preset names."""
    return list(PRESETS.keys())

def get_all_configs() -> Dict[str, List[str]]:
    """
    Get all available configurations including built-in presets and saved configs.
    
    Returns:
        Dictionary mapping accelerator names to configuration names.
    """
    from utils.config_manager import ConfigManager
    
    manager = ConfigManager()
    
    # Start with built-in presets
    result = {}
    for preset_name in PRESETS.keys():
        # Extract accelerator name from preset
        accel_name = preset_name.split(" (")[0] if "(" in preset_name else preset_name
        if accel_name not in result:
            result[accel_name] = {"built_in": [], "custom": []}
        result[accel_name]["built_in"].append(preset_name)
    
    # Add saved configurations
    accelerators = manager.get_all_accelerators()
    for accel in accelerators:
        if accel not in result:
            result[accel] = {"built_in": [], "custom": []}
        saved_configs = manager.get_accelerator_configs(accel)
        result[accel]["custom"].extend(saved_configs)
    
    return result

def load_config_with_source(name: str) -> tuple:
    """
    Load configuration and track its source.
    
    Args:
        name: Configuration name
    
    Returns:
        Tuple of (config_data, source_config_name or None)
    """
    from utils.config_manager import ConfigManager
    
    # Check if it's a built-in preset
    if name in PRESETS:
        return PRESETS[name], None
    
    # Otherwise load from saved configs
    manager = ConfigManager()
    full_config = manager.load_config_with_metadata(name)
    
    if full_config is None:
        return None, None
    
    config_data = full_config.get("config", full_config)
    metadata = full_config.get("metadata", {})
    source = metadata.get("source_config")
    
    return config_data, source
