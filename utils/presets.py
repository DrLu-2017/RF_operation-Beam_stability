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
SOLEIL_II_RF_FREQ = 352.2  # MHz (Main cavity frequency)
SOLEIL_II_HC_FREQ = 1408.8  # MHz (4th harmonic = 1.4088 GHz)

SOLEIL_II_RING = {
    "name": "SOLEIL II",
    "circumference": 354.0,  # meters
    "frequency": SOLEIL_II_RF_FREQ,  # MHz
    "energy": 2.75,  # GeV
    "momentum_compaction": 1.06e-4,  # Closed IDs
    "energy_loss_per_turn": 0.000743,  # GeV (Closed IDs: 743 keV)
    "harmonic_number": 416,
    "damping_time": 0.0122,  # seconds (12.2 ms)
    "energy_spread": 8.55e-4,  # Closed IDs energy spread
}

# ESRF-EBS Normal Conducting (NC) 352 MHz main cavities (MC)
# Parameters per cavity, Ncav = 4
SOLEIL_II_MAIN_CAVITY = {
    "voltage": 1.7,  # MV (Vrf = 1.7 MV total)
    "frequency": SOLEIL_II_RF_FREQ,  # MHz (352 MHz)
    "harmonic": 416,
    "Q0": 35700,  # Unloaded Q = 35700
    "Q": 35700,  # Q0 = 35700 (for backward compatibility)
    "QL": 6000,  # Loaded Q
    "Q_ext": 6364,  # External Q
    "Rs": 5.0,  # Shunt impedance Rs = 5 MΩ / cav
    "R_over_Q": 140.0,  # Rs/Q0 ≈ 5e6/35700 ≈ 140 Ohms
    "Ncav": 4,  # Number of cavities
    "beta": 5.5,  # Coupling factor β
    "detuning_khz": 0.0,  # Detuning frequency (kHz)
    "rf_feedback_gain": 1.3,  # RF feedback gain
    "tau_us": 4.87,  # Cavity decay time (μs)
}

# Passive Normal Conducting (NC) harmonic cavities (HC)
# Custom design by ESRF, Ncav = 2 or 3
SOLEIL_II_HARMONIC_CAVITY = {
    "voltage": 0.35,  # MV (Vmax = 350 kV / cav)
    "frequency": SOLEIL_II_HC_FREQ,  # MHz (1.41 GHz, 4th harmonic)
    "harmonic": 1664,  # 416 * 4 = 1664
    "harmonic_number": 4,  # 4th harmonic of main RF
    "Q0": 31000,  # Unloaded Q = 31000
    "Q": 31000,  # Q0 = 31000 (for backward compatibility)
    "R_over_Q": 29.6,  # R/Q = 29.6 Ω / cav
    "Rs": 0.92,  # Shunt impedance Rs = 0.92 MΩ / cav
    "Ncav": 3,  # Number of cavities (2 or 3)
    "beta": 0.0,  # Coupling factor β (passive cavity)
    "tau_us": 7.0,  # Cavity decay time (μs)
}

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
    "SOLEIL II": {
        "ring": SOLEIL_II_RING,
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY,
        "current": 0.5,  # A
        "passive_hc": True,
        # Scan parameter defaults for SOLEIL II
        "scan_params": {
            "psi_min": 1.0,   # degrees - wider range for SOLEIL II
            "psi_max": 180.0, # degrees
            "psi_points": 50,
        }
    },
    "SOLEIL II (Phase 1)": {
        "ring": {**SOLEIL_II_RING, "energy_loss_per_turn": 0.000803},  # 803 keV (487 + 316)
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY,
        "current": 0.5,  # A
        "passive_hc": True,
        "operation_phase": "Phase 1",
        "scan_params": {
            "psi_min": 1.0,
            "psi_max": 180.0,
            "psi_points": 50,
        }
    },
    "SOLEIL II (Phase 2)": {
        "ring": {**SOLEIL_II_RING, "energy_loss_per_turn": 0.000846},  # 846 keV (487 + 359)
        "main_cavity": SOLEIL_II_MAIN_CAVITY,
        "harmonic_cavity": SOLEIL_II_HARMONIC_CAVITY,
        "current": 0.5,  # A
        "passive_hc": True,
        "operation_phase": "Phase 2",
        "scan_params": {
            "psi_min": 1.0,
            "psi_max": 180.0,
            "psi_points": 50,
        }
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
