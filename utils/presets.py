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
    "Q": 40000,
    "R_over_Q": 100,  # Ohms
}

ALADDIN_HARMONIC_CAVITY_PASSIVE = {
    "voltage": 0.15,  # MV (150 kV - reasonable for passive cavity)
    "frequency": 1498.962,  # MHz (3rd harmonic)
    "harmonic": 96,
    "Q": 20000,
    "R_over_Q": 50,  # Ohms
}

ALADDIN_HARMONIC_CAVITY_ACTIVE = {
    "voltage": 0.3,  # MV
    "frequency": 1498.962,  # MHz (3rd harmonic)
    "harmonic": 96,
    "Q": 20000,
    "R_over_Q": 50,  # Ohms
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
    "Q": 35700,  # Q0 = 35700
    "QL": 6000,  # Loaded Q
    "Rs": 5.0,  # Shunt impedance Rs = 5 MΩ / cav
    "R_over_Q": 140.0,  # Rs/Q0 ≈ 5e6/35700 ≈ 140 Ohms
    "Ncav": 4,  # Number of cavities
}

# Passive Normal Conducting (NC) harmonic cavities (HC)
# Custom design by ESRF, Ncav = 2 or 3
SOLEIL_II_HARMONIC_CAVITY = {
    "voltage": 0.35,  # MV (Vmax = 350 kV / cav)
    "frequency": SOLEIL_II_HC_FREQ,  # MHz (1.41 GHz, 4th harmonic)
    "harmonic": 1664,  # 416 * 4 = 1664
    "harmonic_number": 4,  # 4th harmonic of main RF
    "Q": 31000,  # Q0 = 31000
    "R_over_Q": 29.6,  # R/Q = 29.6 Ω / cav
    "Ncav": 3,  # Number of cavities (2 or 3)
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
