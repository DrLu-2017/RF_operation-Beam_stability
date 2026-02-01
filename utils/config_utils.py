"""
Configuration utilities for Streamlit app.
Handles saving and loading configurations with Streamlit session state integration.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from utils.config_manager import ConfigManager
from utils.presets import PRESETS

def get_config_manager() -> ConfigManager:
    """Get or create a ConfigManager instance in session state."""
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    return st.session_state.config_manager

def save_current_config(config_name: str, accelerator_name: str, config_data: Dict[str, Any], 
                       source_config: Optional[str] = None) -> bool:
    """
    Save current configuration and update session state.
    
    Args:
        config_name: Name of the configuration
        accelerator_name: Name of the accelerator
        config_data: Configuration data to save
        source_config: Source config if this is a modification
    
    Returns:
        True if saved successfully
    """
    try:
        manager = get_config_manager()
        manager.save_config(config_name, accelerator_name, config_data, source_config)
        manager.save_session_config(config_name, accelerator_name)
        
        # Update session state
        st.session_state.current_config = config_name
        st.session_state.current_accelerator = accelerator_name
        
        return True
    except Exception as e:
        st.error(f"Failed to save config: {str(e)}")
        return False

def load_current_config(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Load a configuration from disk.
    
    Args:
        config_name: Name of the configuration to load
    
    Returns:
        Configuration data or None if not found
    """
    manager = get_config_manager()
    return manager.load_config(config_name)

def get_saved_configs_for_accelerator(accelerator_name: str) -> List[str]:
    """
    Get all saved configurations for an accelerator.
    
    Args:
        accelerator_name: Name of the accelerator
    
    Returns:
        List of configuration names
    """
    manager = get_config_manager()
    return manager.get_accelerator_configs(accelerator_name)

def build_config_from_ui(ring_params: Dict, main_cavity: Dict, harmonic_cavity: Dict,
                        current: float, passive_hc: bool) -> Dict[str, Any]:
    """
    Build a configuration dictionary from UI inputs.
    
    Args:
        ring_params: Ring parameters
        main_cavity: Main cavity parameters
        harmonic_cavity: Harmonic cavity parameters
        current: Current value
        passive_hc: Whether harmonic cavity is passive
    
    Returns:
        Complete configuration dictionary
    """
    return {
        "ring": ring_params,
        "main_cavity": main_cavity,
        "harmonic_cavity": harmonic_cavity,
        "current": current,
        "passive_hc": passive_hc,
    }

def extract_ui_params_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters from a configuration for UI display.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary with extracted parameters for each section
    """
    return {
        "ring": config.get("ring", {}),
        "main_cavity": config.get("main_cavity", {}),
        "harmonic_cavity": config.get("harmonic_cavity", {}),
        "current": config.get("current", 0.2),
        "passive_hc": config.get("passive_hc", False),
    }

def get_all_available_configs() -> Dict[str, List[str]]:
    """
    Get all available configurations organized by accelerator.
    
    Returns:
        Dictionary mapping accelerator names to lists of configuration names
    """
    manager = get_config_manager()
    
    # Start with built-in presets
    result = {}
    for preset_name in PRESETS.keys():
        # Extract accelerator name from preset
        accel_name = preset_name.split(" (")[0] if "(" in preset_name else preset_name
        if accel_name not in result:
            result[accel_name] = []
        result[accel_name].append(f"[Built-in] {preset_name}")
    
    # Add saved configurations
    accelerators = manager.get_all_accelerators()
    for accel in accelerators:
        if accel not in result:
            result[accel] = []
        saved_configs = manager.get_accelerator_configs(accel)
        for config_name in saved_configs:
            result[accel].append(config_name)
    
    return result

def initialize_session_config():
    """Initialize configuration in session state."""
    if 'current_config' not in st.session_state:
        manager = get_config_manager()
        session_data = manager.load_session_config()
        
        if session_data:
            st.session_state.current_config = session_data.get("last_config", "SOLEIL II")
            st.session_state.current_accelerator = session_data.get("last_accelerator", "SOLEIL II")
        else:
            st.session_state.current_config = "SOLEIL II"
            st.session_state.current_accelerator = "SOLEIL II"

