"""
Parameter Scans Page
Perform 2D parameter scans for stability analysis.
"""
import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add local mbtrack2 if available
mbtrack2_local = project_root / "mbtrack2-stable"
if mbtrack2_local.exists():
    sys.path.insert(0, str(mbtrack2_local))

# Add local pycolleff if available
pycolleff_local = project_root / "collective_effects-master" / "pycolleff"
if pycolleff_local.exists():
    sys.path.insert(0, str(pycolleff_local))

from utils.presets import get_preset, get_preset_names, load_config_with_source
from utils.albums_wrapper import (
    create_ring_from_params, 
    create_cavity_from_params,
    run_psi_current_scan,
    run_psi_roq_scan
)
from utils.visualization import plot_2d_heatmap, plot_stability_map, plot_stability_regions
from utils.config_manager import ConfigManager
from utils.config_utils import (
    save_current_config, 
    load_current_config,
    build_config_from_ui,
    initialize_session_config,
    get_saved_configs_for_accelerator
)

st.set_page_config(page_title="Parameter Scans", page_icon="📊", layout="wide")

st.title("📊 Parameter Scans")
st.markdown("Explore stability regions across 2D parameter spaces.")

# Initialize session configuration
initialize_session_config()

# Sidebar for preset selection
with st.sidebar:
    st.header("Configuration Management")
    
    # Initialize preset with default
    preset_name = get_preset_names()[0]
    preset, source_config = load_config_with_source(preset_name)
    
    # Configuration selection mode
    config_mode = st.radio(
        "Config Selection",
        options=["Load Preset", "Manage Saved"],
        horizontal=True
    )
    
    if config_mode == "Load Preset":
        # Callback to update session state when preset changes
        def update_inputs_from_preset():
            name = st.session_state.preset_select
            new_preset, _ = load_config_with_source(name)
            if new_preset:
                # Update Main Cavity keys
                mc = new_preset.get("main_cavity", {})
                st.session_state.mc_voltage = float(mc.get("voltage", 0))
                st.session_state.mc_freq = float(mc.get("frequency", 0))
                st.session_state.mc_harm = int(mc.get("harmonic", 0))
                st.session_state.mc_q = float(mc.get("Q", 0))
                st.session_state.mc_roq = float(mc.get("R_over_Q", 0))
                
                # Update Harmonic Cavity keys
                hc = new_preset.get("harmonic_cavity", {})
                st.session_state.hc_voltage = float(hc.get("voltage", 0))
                st.session_state.hc_q = float(hc.get("Q", 0))
                st.session_state.hc_roq = float(hc.get("R_over_Q", 0))
                
                # Calculate and update ratio
                mc_f = float(mc.get("frequency", 1))
                hc_f = float(hc.get("frequency", 0))
                if mc_f > 0:
                    st.session_state.hc_ratio = float(round(hc_f / mc_f))
                else:
                    st.session_state.hc_ratio = 4.0

        # Load from presets or saved configs
        preset_names = get_preset_names()
        default_index = preset_names.index("SOLEIL II") if "SOLEIL II" in preset_names else 0
        preset_name = st.selectbox(
            "Select Configuration",
            options=preset_names,
            index=default_index,
            key="preset_select",
            on_change=update_inputs_from_preset
        )
        
        preset, source_config = load_config_with_source(preset_name)
        
        # Show source config if this is a modified version
        if source_config:
            st.info(f"📝 Based on: **{source_config}**")
    
    else:
        # Manage saved configurations
        st.subheader("Saved Configurations")
        
        # Get current accelerator
        accelerators = ["Aladdin", "SOLEIL II", "Custom"]
        accelerator = st.selectbox(
            "Select Accelerator",
            options=accelerators,
            index=0
        )
        
        # Get saved configs for this accelerator
        saved_configs = get_saved_configs_for_accelerator(accelerator)
        
        if saved_configs:
            saved_config_name = st.selectbox(
                "Saved Configurations",
                options=saved_configs
            )
            
            # Load buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📂 Load", width='stretch'):
                    preset = load_current_config(saved_config_name)
                    if preset:
                        st.session_state.current_config = saved_config_name
                        st.session_state.current_accelerator = accelerator
                        st.success("✅ Config loaded!")
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Delete", width='stretch'):
                    manager = ConfigManager()
                    if manager.delete_config(f"{accelerator}_{saved_config_name}"):
                        st.success("✅ Config deleted!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete config")
            
            with col3:
                if st.button("📤 Export", width='stretch'):
                    manager = ConfigManager()
                    export_path = f"/tmp/{accelerator}_{saved_config_name}.json"
                    if manager.export_config(f"{accelerator}_{saved_config_name}", export_path):
                        st.success(f"✅ Exported to {export_path}")
        else:
            st.info(f"No saved configurations for {accelerator} yet.")
    
    st.markdown("---")
    st.markdown("### Scan Settings")
    
    scan_type = st.selectbox(
        "Scan Type",
        options=["Psi vs Current", "Psi vs R/Q"],
        help="Choose which parameters to scan"
    )
    
    # Recommend method based on configuration
    preset_name_lower = preset_name.lower()
    is_soleil = "soleil" in preset_name_lower
    
    # For SOLEIL II, default to Venturini as Alves has compatibility issues
    default_method_idx = 0 if is_soleil else 0
    
    method = st.selectbox(
        "Solution Method",
        options=["Venturini", "Bosch", "Alves"],
        index=default_method_idx,
        help="Method for solving the Haissinski equation" if not is_soleil else 
             "Method for solving the Haissinski equation. **Note:** SOLEIL II works best with Venturini or Bosch."
    )
    
    driver_hc_value = preset.get("passive_hc", True)
    is_alves = (method == "Alves")
    if is_alves:
        driver_hc_value = True

    passive_hc = st.checkbox(
        "Passive Harmonic Cavity",
        value=driver_hc_value,
        help="Whether the harmonic cavity is passive",
        disabled=is_alves
    )
    if is_alves:
        if is_soleil:
            st.warning("⚠️ Alves method may have compatibility issues with SOLEIL II. Consider using Venturini or Bosch instead.")
        else:
            st.caption("ℹ️ Alves method requires Passive Harmonic Cavity")


# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Parameters", "💾 Save Config", "▶️ Run Scan", "📈 Results"])

with tab1:
    st.header("⚙️ Configuration Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ring Parameters")
        ring_params = preset["ring"]
        
        circumference = st.number_input(
            "Circumference (m)",
            value=float(ring_params["circumference"]),
            min_value=1.0,
            max_value=10000.0,
            format="%.2f",
            key="ring_circumference"
        )
        
        energy = st.number_input(
            "Energy (GeV)",
            value=float(ring_params["energy"]),
            min_value=0.1,
            max_value=10.0,
            format="%.3f",
            key="ring_energy"
        )
        
        momentum_compaction = st.number_input(
            "Momentum Compaction",
            value=float(ring_params["momentum_compaction"]),
            min_value=0.0,
            max_value=0.1,
            format="%.6f",
            key="ring_momentum"
        )
        
        energy_loss = st.number_input(
            "Energy Loss per Turn (keV)",
            value=float(ring_params["energy_loss_per_turn"]) * 1e6,  # Convert GeV to keV (1 GeV = 1e6 keV)
            min_value=0.0,
            max_value=10000.0,
            format="%.2f",
            key="ring_eloss_kev"
        )
        
        harmonic_number = st.number_input(
            "Harmonic Number",
            value=int(ring_params["harmonic_number"]),
            min_value=1,
            max_value=10000,
            key="ring_harmonic"
        )
        
        damping_time = st.number_input(
            "Damping Time (s)",
            value=float(ring_params["damping_time"]),
            min_value=0.0001,
            max_value=1.0,
            format="%.6f",
            key="ring_damping"
        )
    
    with col2:
        st.subheader("Main Cavity")
        mc_params = preset["main_cavity"]
        
        mc_voltage = st.number_input(
            "Voltage (MV)",
            value=float(mc_params["voltage"]),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="mc_voltage"
        )
        
        mc_frequency = st.number_input(
            "Frequency (MHz)",
            value=float(mc_params["frequency"]),
            min_value=1.0,
            max_value=5000.0,
            format="%.3f",
            key="mc_freq"
        )
        
        mc_harmonic = st.number_input(
            "Harmonic",
            value=int(mc_params["harmonic"]),
            min_value=1,
            max_value=10000,
            key="mc_harm"
        )
        
        mc_q = st.number_input(
            "Quality Factor Q",
            value=float(mc_params["Q"]),
            min_value=100.0,
            max_value=1000000.0,
            format="%.0f",
            key="mc_q"
        )
        
        mc_roq = st.number_input(
            "R/Q (Ω)",
            value=float(mc_params["R_over_Q"]),
            min_value=1.0,
            max_value=1000.0,
            format="%.1f",
            key="mc_roq"
        )
        
        st.subheader("Harmonic Cavity")
        hc_params = preset["harmonic_cavity"]
        
        hc_voltage = st.number_input(
            "Voltage (MV)",
            value=float(hc_params["voltage"]),
            min_value=0.0,
            max_value=10.0,
            format="%.3f",
            key="hc_voltage"
        )
        
        # Calculate initial ratio from preset or session state
        # Default to 3 or 4 if calculation fails
        try:
             # If we have session state for mc inputs, use them, otherwise use preset
            mc_f = st.session_state.mc_freq if 'mc_freq' in st.session_state else float(mc_params["frequency"])
            hc_f = float(hc_params["frequency"])
            if mc_f > 0:
                initial_ratio = hc_f / mc_f
            else:
                initial_ratio = 4.0
        except:
             initial_ratio = 4.0

        hc_ratio = st.number_input(
            "Harmonic Ratio (n)",
            value=float(round(initial_ratio)),
            min_value=1.0,
            max_value=100.0,
            step=1.0,
            format="%.1f",
            key="hc_ratio"
        )
        
        # Derived values calculation
        hc_frequency = mc_frequency * hc_ratio
        hc_harmonic = int(mc_harmonic * hc_ratio)
        
        st.info(f"Frequency: **{hc_frequency:.3f} MHz**")
        st.info(f"Harmonic: **{hc_harmonic}** (Absolute)")
        
        hc_q = st.number_input(
            "Quality Factor Q",
            value=float(hc_params["Q"]),
            min_value=100.0,
            max_value=1000000.0,
            format="%.0f",
            key="hc_q"
        )
        
        hc_roq = st.number_input(
            "R/Q (Ω)",
            value=float(hc_params["R_over_Q"]),
            min_value=1.0,
            max_value=1000.0,
            format="%.1f",
            key="hc_roq"
        )

with tab2:
    st.header("💾 Save Configuration")
    st.markdown("Save your current parameter configuration for easy reuse later.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Save As New Configuration")
        save_name = st.text_input(
            "Configuration Name",
            value="my_config",
            help="Name for this configuration"
        )
        
        accelerator_name = st.selectbox(
            "Accelerator",
            options=["Aladdin", "SOLEIL II", "Custom"],
            help="Which accelerator does this config belong to?"
        )
        
        source_from_preset = st.checkbox(
            "Based on preset",
            value=True,
            help="Track which preset this was modified from"
        )
        
        if source_from_preset:
            source_preset = st.selectbox(
                "Source Preset",
                options=get_preset_names(),
                help="The original preset this was modified from"
            )
        else:
            source_preset = None
        
        if st.button("💾 Save Configuration", type="primary", width='stretch'):
            # Build configuration from current UI state
            config_data = {
                "ring": {
                    "circumference": circumference,
                    "energy": energy,
                    "momentum_compaction": momentum_compaction,
                    "energy_loss_per_turn": energy_loss / 1e6,  # Convert keV to GeV (1 GeV = 1e6 keV)
                    "harmonic_number": harmonic_number,
                    "damping_time": damping_time,
                },
                "main_cavity": {
                    "voltage": mc_voltage,
                    "frequency": mc_frequency,
                    "harmonic": mc_harmonic,
                    "Q": mc_q,
                    "R_over_Q": mc_roq,
                },
                "harmonic_cavity": {
                    "voltage": hc_voltage,
                    "frequency": hc_frequency,
                    "harmonic": hc_harmonic,
                    "Q": hc_q,
                    "R_over_Q": hc_roq,
                },
                "current": 0.2,
                "passive_hc": passive_hc,
            }
            
            manager = ConfigManager()
            source_config = source_preset if source_from_preset else None
            manager.save_config(save_name, accelerator_name, config_data, source_config)
            manager.save_session_config(save_name, accelerator_name)
            
            st.success(f"✅ Configuration saved as '{accelerator_name}_{save_name}'!")
            st.info(f"This configuration will be loaded next time you open the app.")
    
    with col2:
        st.subheader("Quick Update Current Config")
        
        if st.session_state.get("current_config"):
            current_name = st.session_state.get("current_config", "Unnamed")
            current_accel = st.session_state.get("current_accelerator", "Custom")
            
            st.info(f"""
            **Current Configuration:**
            - Name: {current_name}
            - Accelerator: {current_accel}
            """)
            
            if st.button("💾 Update Current Config", width='stretch'):
                # Build configuration from current UI state
                config_data = {
                    "ring": {
                        "circumference": circumference,
                        "energy": energy,
                        "momentum_compaction": momentum_compaction,
                        "energy_loss_per_turn": energy_loss / 1e6,  # Convert keV to GeV (1 GeV = 1e6 keV)
                        "harmonic_number": harmonic_number,
                        "damping_time": damping_time,
                    },
                    "main_cavity": {
                        "voltage": mc_voltage,
                        "frequency": mc_frequency,
                        "harmonic": mc_harmonic,
                        "Q": mc_q,
                        "R_over_Q": mc_roq,
                    },
                    "harmonic_cavity": {
                        "voltage": hc_voltage,
                        "frequency": hc_frequency,
                        "harmonic": hc_harmonic,
                        "Q": hc_q,
                        "R_over_Q": hc_roq,
                    },
                    "current": 0.2,
                    "passive_hc": passive_hc,
                }
                
                manager = ConfigManager()
                manager.save_config(current_name, current_accel, config_data)
                manager.save_session_config(current_name, current_accel)
                
                st.success(f"✅ '{current_name}' configuration updated!")
        else:
            st.info("No current configuration selected. Save a new one first.")

with tab3:
    st.header("▶️ Run Parameter Scan")
    
    col1, col2 = st.columns(2)
    
    # Get scan parameters from preset
    scan_params = preset.get("scan_params", {})
    default_psi_min = scan_params.get("psi_min", 60.0)
    default_psi_max = scan_params.get("psi_max", 90.0)
    default_psi_points = scan_params.get("psi_points", 30)
    
    with col1:
        st.subheader("Psi Range")
        psi_min = st.number_input(
            "Min (degrees)", 
            value=default_psi_min, 
            min_value=0.1, 
            max_value=180.0, 
            step=0.1,
            key="scan_psi_min"
        )
        psi_max = st.number_input(
            "Max (degrees)", 
            value=default_psi_max, 
            min_value=0.1, 
            max_value=180.0, 
            step=0.1,
            key="scan_psi_max"
        )
        psi_points = st.slider(
            "Number of Points", 
            min_value=10, 
            max_value=100, 
            value=default_psi_points,
            key="scan_psi_pts"
        )
    
    with col2:
        if scan_type == "Psi vs Current":
            st.subheader("Current Range")
            current_min = st.number_input("Min (A)", value=0.2, min_value=0.0, max_value=10.0, format="%.3f")
            current_max = st.number_input("Max (A)", value=0.5, min_value=0.0, max_value=10.0, format="%.3f")
            current_points = st.slider("Number of Points", min_value=10, max_value=100, value=30, key="current_pts")
        else:
            st.subheader("R/Q Range")
            roq_min = st.number_input("Min (Ω)", value=10.0, min_value=1.0, max_value=1000.0)
            roq_max = st.number_input("Max (Ω)", value=200.0, min_value=1.0, max_value=1000.0)
            roq_points = st.slider("Number of Points", min_value=10, max_value=100, value=30, key="roq_pts")
            
            current_fixed = st.number_input(
                "Fixed Current (A)",
                value=float(preset.get("current", 0.2)),
                min_value=0.0,
                max_value=10.0,
                format="%.3f"
            )
    
    st.markdown("---")

    
    if st.button("🚀 Run Scan", type="primary", width='stretch'):
        with st.spinner("Running parameter scan... This may take a few minutes."):
            try:
                # Create ring and cavities (convert keV to GeV for backend)
                ring = create_ring_from_params(
                    circumference, energy, momentum_compaction,
                    energy_loss / 1e6, harmonic_number, damping_time  # Convert keV to GeV (1 GeV = 1e6 keV)
                )
                
                main_cavity = create_cavity_from_params(
                    mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq
                )
                
                harmonic_cavity = create_cavity_from_params(
                    hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq
                )
                
                # Run scan
                if scan_type == "Psi vs Current":
                    result = run_psi_current_scan(
                        ring, main_cavity, harmonic_cavity,
                        psi_range=(psi_min, psi_max, psi_points),
                        current_range=(current_min, current_max, current_points),
                        method=method,
                        passive_hc=passive_hc
                    )
                else:
                    result = run_psi_roq_scan(
                        ring, main_cavity, harmonic_cavity,
                        current=current_fixed,
                        psi_range=(psi_min, psi_max, psi_points),
                        roq_range=(roq_min, roq_max, roq_points),
                        method=method,
                        passive_hc=passive_hc
                    )
                
                # Store results in session state
                st.session_state['scan_results'] = result
                st.session_state['scan_type'] = scan_type
                
                if result['success']:
                    st.success("✅ Scan completed successfully!")
                    # Check if fallback method was used
                    if result.get('fallback_used', False):
                        st.warning("⚠️ **Note:** The Venturini method had low convergence for this configuration. "
                                   "Results were obtained using the Bosch method fallback.")
                    st.info("Switch to the 'Results' tab to view the results.")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    status = result.get('status', 'UNKNOWN')
                    hint = result.get('hint', '')
                    traceback_str = result.get('traceback', '')
                    
                    st.error(f"❌ Scan failed")
                    st.error(f"**Error:** {error_msg}")
                    
                    if traceback_str:
                        with st.expander("📋 Full Error Traceback"):
                            st.code(traceback_str, language="python")
                    
                    if status == "TYPE_MISMATCH":
                        st.warning("""
                        **⚠️ This is a known limitation:**
                        
                        The ALBuMS library requires full object instances (Synchrotron, CavityResonator classes) 
                        from the mbtrack2 package. However, these are not available or not fully installed in your environment.
                        
                        **Solutions:**
                        1. **Install mbtrack2-stable**: The mbtrack2-stable folder in your project should be installed
                        2. **Check MPI installation**: ALBuMS uses MPI (Message Passing Interface)
                        3. **Use degraded mode**: The UI will show parameter configurations but cannot run actual scans
                        """)
                    elif status == "NOT_AVAILABLE":
                        st.info("""
                        **ℹ️ ALBuMS Not Fully Installed**
                        
                        The application is running in UI mode but cannot execute scans. 
                        For full functionality, you need:
                        - mbtrack2-stable fully installed and importable
                        - MPI libraries configured
                        - ALBuMS Python modules available
                        """)
                    
                    if hint:
                        st.info(f"**Hint:** {hint}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

with tab4:
    st.header("📈 Scan Results")
    
    if 'scan_results' in st.session_state:
        result = st.session_state['scan_results']
        scan_type_result = st.session_state.get('scan_type', 'Psi vs Current')
        
        # Display scan success/failure status
        st.write(f"**Scan Success:** {result.get('success', False)}")
        st.write(f"**Result Keys:** {list(result.keys())}")
        
        if result['success']:
            st.success("✅ Results from last scan")
            
            # Display scan parameters
            st.markdown("### Scan Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                psi_count = len(result.get('psi_vals', []))
                st.metric("Psi Points", psi_count)
            with col2:
                current_count = len(result.get('current_vals', []))
                if scan_type_result == "Psi vs Current":
                    st.metric("Current Points", current_count)
                else:
                    st.metric("R/Q Points", current_count)
            with col3:
                total_points = psi_count * current_count if psi_count > 0 and current_count > 0 else 0
                st.metric("Total Points", total_points)
            
            # Display actual results if available
            scan_results = result.get('results', {})
            
            st.write(f"**Results dict keys:** {list(scan_results.keys())}")
            st.write(f"**Results dict type:** {type(scan_results)}")
            
            if scan_results and len(scan_results) > 0:
                # Check for NaN/Inf values
                has_valid_data = False
                data_summary = {}
                for key, data in scan_results.items():
                    if hasattr(data, 'size'):
                        if hasattr(data, '__iter__'):
                            valid_count = np.sum(np.isfinite(data.flatten()))
                            total_count = data.size
                            has_valid_data = has_valid_data or valid_count > 0
                            data_summary[key] = f"{valid_count}/{total_count} valid"
                
                st.markdown("### Data Quality Report")
                with st.expander("📊 Data Summary (Click to expand)", expanded=False):
                    for key, summary in data_summary.items():
                        st.write(f"• **{key}**: {summary}")
                
                if has_valid_data:
                    st.markdown("### Stability Maps")
                    
                    # Create tabs for different visualizations
                    results_tab0, results_tab1, results_tab2, results_tab3 = st.tabs(
                        ["📊 Stability Regions", "🌡️ Heatmaps", "📈 Bunch Length", "⏱️ Touschek Lifetime"]
                    )
                    
                    with results_tab0:
                        st.subheader("Stability Analysis Map")
                        st.markdown("Detailed stability map identifying different instability regimes.")
                        
                        try:
                            # Determine Y values based on scan type
                            if 'current_vals' in result:
                                y_vals = result['current_vals']
                                y_label = "Beam current I0 [mA]" # The plot function converts A to mA automatically if value < 100
                            elif 'roq_vals' in result:
                                y_vals = result['roq_vals']
                                y_label = "R/Q [Ω]"
                            else:
                                y_vals = []
                                y_label = "Y Axis"
                                
                            # Check if we have necessary keys in results
                            if 'xi' in scan_results and 'robinson_coup' in scan_results:
                                # Calculate and display convergence statistics
                                converged = scan_results.get('converged_coup')
                                if converged is not None:
                                    total_points = converged.size
                                    converged_points = np.sum(converged)
                                    convergence_pct = 100 * converged_points / total_points if total_points > 0 else 0
                                    
                                    if convergence_pct < 50:
                                        st.warning(f"⚠️ **Low convergence rate: {convergence_pct:.1f}%** ({converged_points}/{total_points} points)")
                                        st.info("Results may be scattered/unreliable. Try: 1) Using Bosch method, 2) Reducing parameter range, 3) Checking cavity parameters")
                                    elif convergence_pct < 80:
                                        st.info(f"ℹ️ Convergence rate: {convergence_pct:.1f}% ({converged_points}/{total_points} points)")
                                    else:
                                        st.success(f"✅ Good convergence: {convergence_pct:.1f}% ({converged_points}/{total_points} points)")
                                
                                fig_stab = plot_stability_regions(
                                    psi_vals=result['psi_vals'],
                                    y_vals=y_vals,
                                    results=scan_results,
                                    x_label="Harmonic cavity tuning angle ψ2 [deg]",
                                    y_label=y_label,
                                    title="Stability Map",
                                    mode_coupling=True 
                                )
                                st.plotly_chart(fig_stab, width='stretch')
                                
                                # Add legend explanation
                                with st.expander("ℹ️ Legend Explanation"):
                                    st.markdown("""
                                    - **Xi Isoline**: Contour lines of the tuning angle parameter ξ
                                    - **CBI driven by HOMs (▲)**: Coupled Bunch Instability driven by Higher Order Modes
                                    - **Dipole Robinson (●)**: Instability in the dipole mode
                                    - **Quadrupole Robinson (▼)**: Instability in the quadrupole mode
                                    - **Fast mode-coupling (★)**: Fast instability due to mode coupling
                                    - **Zero-frequency (♦)**: Zero-frequency instability
                                    - **PTBL (X)**: Periodic Transient Beam Loading instability
                                    - **Not converged (Y)**: Simulation did not converge
                                    - **Stable beam (○)**: Region with no detected instabilities
                                    """)
                            else:
                                st.warning("⚠️ Results missing required data for stability map (xi, robinson_coup). check if the scan completed successfully.")
                                if 'robinson_coup' not in scan_results:
                                    st.info(f"Available keys: {list(scan_results.keys())}")
                        except Exception as e:
                            st.error(f"❌ Error plotting stability map: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    with results_tab1:
                        try:
                            st.write("**Attempting to find valid data for Growth Rates:**")
                            
                            # Try multiple data sources with proper handling
                            plot_data = None
                            data_source = None
                            
                            # 1. Try robinson_coup (shape: 30,30,4 - 4 modes)
                            if 'robinson_coup' in scan_results:
                                rc = scan_results['robinson_coup']
                                if hasattr(rc, 'shape') and len(rc.shape) == 3:
                                    # Extract first mode, preserving valid values
                                    raw_data = rc[:,:,0]
                                    plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                    data_source = f"Robinson coupling (mode 0) - shape {rc.shape} → {plot_data.shape}"
                                    st.success(f"✓ Using {data_source}")
                                else:
                                    raw_data = rc
                                    plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                    if np.sum(np.isfinite(rc)) > 0:
                                        data_source = f"Robinson coupling - shape {rc.shape}"
                                        st.success(f"✓ Using {data_source}")
                                    else:
                                        plot_data = None
                            
                            # 2. Fallback to modes_coup (shape: 30,30,2)
                            if plot_data is None and 'modes_coup' in scan_results:
                                mc = scan_results['modes_coup']
                                if hasattr(mc, 'shape') and len(mc.shape) == 3:
                                    raw_data = mc[:,:,0]
                                    plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                    data_source = f"Modes coupling (mode 0) - shape {mc.shape} → {plot_data.shape}"
                                    st.info(f"ℹ Using {data_source}")
                                else:
                                    raw_data = mc
                                    plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                    if np.sum(np.isfinite(mc)) > 0:
                                        data_source = f"Modes coupling - shape {mc.shape}"
                                        st.info(f"ℹ Using {data_source}")
                                    else:
                                        plot_data = None
                            
                            # 3. Fallback to zero_freq_coup
                            if plot_data is None and 'zero_freq_coup' in scan_results:
                                zfc = scan_results['zero_freq_coup']
                                raw_data = zfc
                                plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                data_source = f"Zero frequency coupling - shape {zfc.shape}"
                                st.warning(f"⚠ Using {data_source}")
                            
                            has_valid_growth_data = (plot_data is not None and 
                                                    np.sum(np.isfinite(plot_data)) > 0)
                            
                            if not has_valid_growth_data:
                                st.error("❌ No valid data available")
                            else:
                                st.write(f"**Plot info:** {data_source}")
                                st.write(f"**Data range:** [{plot_data.min():.4f}, {plot_data.max():.4f}]")
                                
                                fig = plot_2d_heatmap(
                                    x_vals=result['psi_vals'],
                                    y_vals=result['current_vals'],
                                    z_vals=plot_data,
                                    x_label="Psi (degrees)",
                                    y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                    z_label="Growth Rate / Coupling",
                                    title="Growth Rate Stability Map",
                                    colorscale="RdBu_r"
                                )
                                st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.error(f"❌ Error in growth rate plot: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    with results_tab2:
                        try:
                            st.write("**Attempting to find valid data for Bunch Length:**")
                            
                            plot_data = None
                            data_source = None
                            
                            # 1. Try modes_coup first (shape: 30,30,2)
                            if 'modes_coup' in scan_results:
                                mc = scan_results['modes_coup']
                                valid_count = np.sum(np.isfinite(mc.flatten()))
                                st.write(f"- modes_coup: shape={mc.shape}, valid={valid_count}/1800")
                                
                                if valid_count > 0:
                                    if len(mc.shape) == 3:
                                        # Extract first mode, preserving valid values
                                        raw_data = mc[:,:,0]
                                        # Only replace NaN and Inf with 0, keep real values
                                        plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                        data_source = f"Modes (mode 0) - shape {mc.shape} → {plot_data.shape}"
                                    else:
                                        raw_data = mc
                                        plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                        data_source = f"Modes - shape {mc.shape}"
                                    st.success(f"✓ Using {data_source}")
                            
                            # 2. Try converged_coup (shape: 30,30,2)
                            if plot_data is None and 'converged_coup' in scan_results:
                                cc = scan_results['converged_coup']
                                valid_count = np.sum(np.isfinite(cc.flatten()))
                                st.write(f"- converged_coup: shape={cc.shape}, valid={valid_count}/1800")
                                
                                if valid_count > 0:
                                    if len(cc.shape) == 3:
                                        raw_data = cc[:,:,0]
                                        plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                        data_source = f"Convergence (1st elem) - shape {cc.shape} → {plot_data.shape}"
                                    else:
                                        raw_data = cc
                                        plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                        data_source = f"Convergence - shape {cc.shape}"
                                    st.info(f"ℹ Using {data_source}")
                            
                            # 3. Try robinson_coup (shape: 30,30,4) - use mean
                            if plot_data is None and 'robinson_coup' in scan_results:
                                rc = scan_results['robinson_coup']
                                valid_count = np.sum(np.isfinite(rc.flatten()))
                                st.write(f"- robinson_coup: shape={rc.shape}, valid={valid_count}/3600")
                                
                                if valid_count > 0:
                                    if len(rc.shape) == 3:
                                        # Use nanmean which ignores NaN values
                                        plot_data = np.nanmean(rc, axis=2)
                                        # Replace remaining NaN with 0
                                        plot_data = np.where(np.isfinite(plot_data), plot_data, 0.0)
                                        data_source = f"Robinson (mean) - shape {rc.shape} → {plot_data.shape}"
                                    else:
                                        raw_data = rc
                                        plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                        data_source = f"Robinson - shape {rc.shape}"
                                    st.warning(f"⚠ Using {data_source}")
                            
                            # 4. Try zero_freq_coup
                            if plot_data is None and 'zero_freq_coup' in scan_results:
                                zfc = scan_results['zero_freq_coup']
                                raw_data = zfc
                                plot_data = np.where(np.isfinite(raw_data), raw_data, 0.0)
                                data_source = f"Zero frequency coupling - shape {zfc.shape}"
                                st.warning(f"⚠ Using {data_source}")
                            
                            # Check if we have meaningful data (has at least some non-zero values)
                            has_valid_data = (plot_data is not None and 
                                            np.sum(np.isfinite(plot_data)) > 0 and 
                                            np.max(np.abs(plot_data)) > 0)
                            
                            if not has_valid_data:
                                st.error("❌ No valid data available for bunch length")
                            else:
                                st.write(f"**Plot info:** {data_source}")
                                st.write(f"**Data range:** [{plot_data.min():.4f}, {plot_data.max():.4f}]")
                                
                                fig = plot_2d_heatmap(
                                    x_vals=result['psi_vals'],
                                    y_vals=result['current_vals'],
                                    z_vals=plot_data,
                                    x_label="Psi (degrees)",
                                    y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                    z_label="Bunch Length Proxy",
                                    title="Bunch Length Stability Map",
                                    colorscale="Viridis"
                                )
                                st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.error(f"❌ Error in bunch length plot: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    with results_tab3:
                        try:
                            st.write("**Attempting to find valid data for Touschek Lifetime:**")
                            
                            plot_data = None
                            data_source = None
                            
                            # 1. Try R factor first (actual Touschek lifetime ratio)
                            if 'R' in scan_results:
                                r_data = scan_results['R']
                                if hasattr(r_data, 'flatten'):
                                    valid_count = np.sum(np.isfinite(r_data.flatten()))
                                    total_count = r_data.size
                                    st.write(f"- R (Touschek lifetime ratio): shape={r_data.shape}, valid={valid_count}/{total_count}")
                                    
                                    if valid_count > 0:
                                        # Preserve valid values, replace inf with 0
                                        plot_data = np.where(np.isfinite(r_data), r_data, 0.0)
                                        data_source = f"Touschek Lifetime (R factor) - shape {r_data.shape}"
                                        st.success(f"✓ Using {data_source}")
                            
                            # 2. Try HOM_coup (shape: 30,30)
                            if plot_data is None and 'HOM_coup' in scan_results:
                                hc = scan_results['HOM_coup']
                                valid_count = np.sum(np.isfinite(hc.flatten()))
                                st.write(f"- HOM_coup: shape={hc.shape}, valid={valid_count}/900")
                                
                                if valid_count > 0:
                                    plot_data = np.where(np.isfinite(hc), hc, 0.0)
                                    data_source = f"HOM coupling - shape {hc.shape}"
                                    st.info(f"ℹ Fallback: Using {data_source}")
                            
                            # 3. Try PTBL_coup (shape: 30,30)
                            if plot_data is None and 'PTBL_coup' in scan_results:
                                pc = scan_results['PTBL_coup']
                                valid_count = np.sum(np.isfinite(pc.flatten()))
                                st.write(f"- PTBL_coup: shape={pc.shape}, valid={valid_count}/900")
                                
                                if valid_count > 0:
                                    plot_data = np.where(np.isfinite(pc), pc, 0.0)
                                    data_source = f"PTBL coupling - shape {pc.shape}"
                                    st.warning(f"⚠ Fallback: Using {data_source}")
                            
                            # 4. Try robinson_coup (shape: 30,30,4) - use mean
                            if plot_data is None and 'robinson_coup' in scan_results:
                                rc = scan_results['robinson_coup']
                                valid_count = np.sum(np.isfinite(rc.flatten()))
                                st.write(f"- robinson_coup: shape={rc.shape}, valid={valid_count}/3600")
                                
                                if valid_count > 0:
                                    if len(rc.shape) == 3:
                                        # Use nanmean to average across modes
                                        raw_mean = np.nanmean(rc, axis=2)
                                        plot_data = np.where(np.isfinite(raw_mean), raw_mean, 0.0)
                                        data_source = f"Robinson (mean) - shape {rc.shape} → {plot_data.shape}"
                                    else:
                                        plot_data = np.where(np.isfinite(rc), rc, 0.0)
                                        data_source = f"Robinson - shape {rc.shape}"
                                    st.warning(f"⚠ Fallback: Using {data_source}")
                            
                            # 5. Try zero_freq_coup
                            if plot_data is None and 'zero_freq_coup' in scan_results:
                                zfc = scan_results['zero_freq_coup']
                                plot_data = np.where(np.isfinite(zfc), zfc, 0.0)
                                data_source = f"Zero frequency coupling - shape {zfc.shape}"
                                st.warning(f"⚠ Fallback: Using {data_source}")
                            
                            has_valid_touschek_data = (plot_data is not None and 
                                                       np.sum(np.isfinite(plot_data)) > 0 and 
                                                       np.max(np.abs(plot_data)) > 0)
                            
                            if not has_valid_touschek_data:
                                st.error("❌ No valid data available for Touschek lifetime")
                            else:
                                st.write(f"**Plot info:** {data_source}")
                                st.write(f"**Data range:** [{plot_data.min():.4f}, {plot_data.max():.4f}]")
                                
                                fig = plot_2d_heatmap(
                                    x_vals=result['psi_vals'],
                                    y_vals=result['current_vals'],
                                    z_vals=plot_data,
                                    x_label="Psi (degrees)",
                                    y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                    z_label="Touschek Lifetime Proxy",
                                    title="Touschek Lifetime Stability Map",
                                    colorscale="Plasma"
                                )
                                st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.error(f"❌ Error in Touschek lifetime plot: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            
                            fig = plot_2d_heatmap(
                                x_vals=result['psi_vals'],
                                y_vals=result['current_vals'],
                                z_vals=r_clean,
                                x_label="Psi (degrees)",
                                y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                z_label="Touschek Lifetime / Coupling",
                                title="Touschek Lifetime / Stability Map",
                                colorscale="Plasma"
                            )
                            st.plotly_chart(fig, width='stretch')
                        except Exception as e:
                            st.warning(f"Could not display Touschek lifetime map: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    # Display raw data summary
                    with st.expander("📋 Full Results Summary"):
                        st.write("Available data in results:")
                        for key in scan_results.keys():
                            data = scan_results[key]
                            if hasattr(data, 'shape'):
                                st.write(f"• **{key}**: shape {data.shape}, dtype {data.dtype}")
                            else:
                                st.write(f"• **{key}**: {type(data).__name__}")
                else:
                    st.warning("⚠️ Scan completed but results contain mostly NaN/Inf values")
            else:
                st.warning("💡 Scan completed but detailed results are not available. This may indicate:")
                st.write("- Scan was run with limited/degraded mode (dict parameters)")
                st.write("- ALBuMS functions not fully available")
                st.write("- Results structure issue")
                st.write(f"\nRaw result structure:")
                st.json({k: str(type(v).__name__) for k, v in result.items()})
            
        else:
            st.error(f"❌ Last scan failed: {result.get('error', 'Unknown error')}")
            if 'hint' in result:
                st.info(f"**Hint:** {result['hint']}")
    else:
        st.info("👈 Configure parameters and run a scan to see results here.")
