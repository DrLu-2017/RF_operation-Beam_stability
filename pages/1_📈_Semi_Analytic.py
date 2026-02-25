"""
Semi-Analytic Tools — Unified Page
====================================
Consolidated interface for ALBuMS semi-analytical algorithms:
  1. Parameter Scans   — 2D stability maps (ψ vs I, ψ vs R/Q, etc.)
  2. R-Factor Optimization — find optimal HC phase for Touschek lifetime
  3. Robinson Mode Analysis — track modes and identify instabilities

Reference:
  A. Gamelin, V. Gubaidulin, M. Alves, T. Olsson,
  "Semi-analytical algorithms to study longitudinal beam instabilities
   in double rf systems".
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
    run_psi_roq_scan,
    run_optimization,
    analyze_robinson_modes,
)
from utils.visualization import (
    plot_2d_heatmap,
    plot_stability_map,
    plot_stability_regions,
    plot_optimization_result,
    plot_mode_frequencies,
    plot_growth_rates,
    plot_r_factor_vs_psi,
)
from utils.ui_utils import fmt, render_display_settings
from utils.config_manager import ConfigManager
from utils.config_utils import (
    save_current_config,
    load_current_config,
    build_config_from_ui,
    initialize_session_config,
    get_saved_configs_for_accelerator,
)
import plotly.graph_objects as go

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(page_title="Semi-Analytic", page_icon="📈", layout="wide")

if not st.session_state.get("authentication_status"):
    st.info("Please login from the Home page.")
    st.stop()

st.title("📈 Semi-Analytic Tools")
st.markdown(
    "Unified interface for **ALBuMS** semi-analytical algorithms — "
    "parameter scans, optimisation, and Robinson mode analysis."
)

# Initialize session configuration
initialize_session_config()

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align: center;'><h1 style='color: #4facfe;'>DRFB</h1></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Quick Navigation")
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/0_🔧_Double_RF_System.py", label="Double RF System", icon="🔧")
    st.page_link("pages/1_📈_Semi_Analytic.py", label="Semi-Analytic Tools", icon="📈")
    st.page_link("pages/2_🚀_MBTrack2_Remote.py", label="mbTrack2 Remote Job", icon="🚀")

    st.markdown("---")
    st.header("Configuration Management")

    # ── Preset selection ──
    preset_name_default = get_preset_names()[0]
    preset, source_config = load_config_with_source(preset_name_default)

    config_mode = st.radio(
        "Config Selection", options=["Load Preset", "Manage Saved"], horizontal=True
    )

    if config_mode == "Load Preset":

        def update_inputs_from_preset():
            name = st.session_state.preset_select
            new_preset, _ = load_config_with_source(name)
            if new_preset:
                # Loop through all prefixes to sync tabs
                for prefix in ["", "opt_", "mode_"]:
                    # Ring
                    st.session_state[f"{prefix}ring_circumference"] = float(ring.get("circumference", 0))
                    st.session_state[f"{prefix}ring_energy"] = float(ring.get("energy", 0))
                    st.session_state[f"{prefix}ring_momentum"] = float(ring.get("momentum_compaction", 0))
                    st.session_state[f"{prefix}ring_eloss_kev"] = float(ring.get("energy_loss_per_turn", 0)) * 1e6
                    st.session_state[f"{prefix}ring_harmonic"] = int(ring.get("harmonic_number", 0))
                    st.session_state[f"{prefix}ring_damping"] = float(ring.get("damping_time", 0))
                    st.session_state[f"{prefix}ring_espread"] = float(ring.get("energy_spread", 0.001))

                    # Main Cavity
                    st.session_state[f"{prefix}mc_voltage"] = float(mc.get("voltage", 0))
                    st.session_state[f"{prefix}mc_freq"] = float(mc.get("frequency", 0))
                    st.session_state[f"{prefix}mc_harm"] = int(mc.get("harmonic", 0))
                    st.session_state[f"{prefix}mc_q"] = float(mc.get("Q", 0))
                    st.session_state[f"{prefix}mc_roq"] = float(mc.get("R_over_Q", 0))

                    # Harmonic Cavity
                    st.session_state[f"{prefix}hc_voltage"] = float(hc.get("voltage", 0))
                    st.session_state[f"{prefix}hc_freq"] = float(hc.get("frequency", 0))
                    st.session_state[f"{prefix}hc_harm"] = int(hc.get("harmonic", 0))
                    st.session_state[f"{prefix}hc_q"] = float(hc.get("Q", 0))
                    st.session_state[f"{prefix}hc_roq"] = float(hc.get("R_over_Q", 0))

                    # Ratio
                    mc_f = float(mc.get("frequency", 1))
                    hc_f = float(hc.get("frequency", 0))
                    ratio = float(round(hc_f / mc_f)) if mc_f > 0 else 4.0
                    st.session_state[f"{prefix}hc_ratio"] = ratio

                # Scan Params (Update for default and mode_ prefixes)
                scan_params = new_preset.get("scan_params", {})
                if scan_params:
                    # Parameter Scans
                    st.session_state["psi_min"] = float(scan_params.get("psi_min", 1.0))
                    st.session_state["psi_max"] = float(scan_params.get("psi_max", 180.0))
                    st.session_state["psi_points"] = int(scan_params.get("psi_points", 30))
                    st.session_state["current_min"] = float(scan_params.get("current_min", 0.001))
                    st.session_state["current_max"] = float(scan_params.get("current_max", 0.5))
                    st.session_state["current_points"] = int(scan_params.get("current_points", 30))
                    
                    # Mode Analysis
                    st.session_state["mode_psi_min"] = float(scan_params.get("psi_min", 1.0))
                    st.session_state["mode_psi_max"] = float(scan_params.get("psi_max", 180.0))
                    st.session_state["mode_psi_points"] = int(scan_params.get("psi_points", 50))

                # Update shared items
                st.session_state["opt_current"] = float(new_preset.get("current", 0.2))
                st.session_state["mode_current"] = float(new_preset.get("current", 0.2))
                st.session_state["passive_hc"] = new_preset.get("passive_hc", True)


        preset_names = get_preset_names()
        default_index = preset_names.index("SOLEIL II") if "SOLEIL II" in preset_names else 0
        preset_name = st.selectbox(
            "Select Configuration",
            options=preset_names,
            index=default_index,
            key="preset_select",
            on_change=update_inputs_from_preset,
        )
        preset, source_config = load_config_with_source(preset_name)
        if source_config:
            st.info(f"📝 Based on: **{source_config}**")
    else:
        st.subheader("Saved Configurations")
        accelerators = ["Aladdin", "SOLEIL II", "Custom"]
        accelerator = st.selectbox("Select Accelerator", options=accelerators, index=0)
        saved_configs = get_saved_configs_for_accelerator(accelerator)
        if saved_configs:
            saved_config_name = st.selectbox("Saved Configurations", options=saved_configs)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("📂 Load", use_container_width=True):
                    preset = load_current_config(saved_config_name)
                    if preset:
                        st.session_state.current_config = saved_config_name
                        st.session_state.current_accelerator = accelerator
                        st.success("✅ Config loaded!")
                        st.rerun()
            with col_b:
                if st.button("🗑️ Delete", use_container_width=True):
                    manager = ConfigManager()
                    if manager.delete_config(f"{accelerator}_{saved_config_name}"):
                        st.success("✅ Config deleted!")
                        st.rerun()
            with col_c:
                if st.button("📤 Export", use_container_width=True):
                    manager = ConfigManager()
                    export_path = f"/tmp/{accelerator}_{saved_config_name}.json"
                    if manager.export_config(f"{accelerator}_{saved_config_name}", export_path):
                        st.success(f"✅ Exported to {export_path}")
        else:
            st.info(f"No saved configurations for {accelerator} yet.")

    st.markdown("---")
    st.markdown("### Algorithm Settings")

    # ── Method selector (common to all tools) ──
    METHOD_INFO = {
        "Gamelin": (
            "**New algorithm** — Solves the Haissinski equation for self-consistent "
            "equilibrium, then applies Robinson mode coupling + PTBL instability analysis. "
            "*(Gamelin et al.)*"
        ),
        "Venturini": (
            "Uses Venturini's Haissinski solver (passive & active HC). "
            "*(Venturini, PRAB 2018)*"
        ),
        "Bosch": (
            "Original form-factor iteration (Bosch 1993/2001). "
            "Computes amplitude form factor and bunch length only."
        ),
        "Alves": (
            "Arbitrary filling pattern + broadband impedance via pycolleff. "
            "Passive HC only. *(Alves & de Sá, PRAB 2023)*"
        ),
        "Hofmann": (
            "Beam dynamics in a double RF system. "
            "Flat potential solver and analytical approximation. *(Hofmann & Myers, 1980)*"
        ),
    }

    preset_name_lower = preset_name.lower() if 'preset_name' in dir() else ""
    is_soleil = "soleil" in preset_name_lower

    method = st.selectbox(
        "Solution Method",
        options=list(METHOD_INFO.keys()),
        index=0,
        help="Algorithm for the self-consistent problem (Haissinski eq.)",
    )
    with st.expander("ℹ️ About this method"):
        st.markdown(METHOD_INFO[method])

    # Map "Gamelin" to the backend method name
    def _backend_method(ui_method: str) -> str:
        """Map UI method name → albums/robinson.py method name."""
        return "Venturini" if ui_method == "Gamelin" else ui_method

    is_alves = method == "Alves"
    driver_hc_value = preset.get("passive_hc", True)
    if is_alves:
        driver_hc_value = True

    passive_hc = st.checkbox(
        "Passive Harmonic Cavity",
        value=driver_hc_value,
        help="Whether the harmonic cavity is passive",
        disabled=is_alves,
        key="passive_hc",
    )
    if is_alves:
        if is_soleil:
            st.warning("⚠️ Alves method may have compatibility issues with SOLEIL II.")
        else:
            st.caption("ℹ️ Alves method requires Passive Harmonic Cavity")

    render_display_settings()


# ═══════════════════════════════════════════════════════════════════════
# SHARED PARAMETER INPUTS
# ═══════════════════════════════════════════════════════════════════════

def _render_ring_params(prefix: str = ""):
    """Render ring parameter inputs and return values."""
    ring_params = preset["ring"]
    circumference = st.number_input(
        "Circumference (m)",
        value=float(st.session_state.get(f"{prefix}ring_circumference", ring_params["circumference"])),
        min_value=1.0, max_value=10000.0, format="%.2f",
        key=f"{prefix}ring_circumference",
    )
    energy = st.number_input(
        "Energy (GeV)",
        value=float(st.session_state.get(f"{prefix}ring_energy", ring_params["energy"])),
        min_value=0.1, max_value=10.0, format="%.3f",
        key=f"{prefix}ring_energy",
    )
    momentum_compaction = st.number_input(
        "Momentum Compaction",
        value=float(st.session_state.get(f"{prefix}ring_momentum", ring_params["momentum_compaction"])),
        min_value=0.0, max_value=0.1, format="%.6f",
        key=f"{prefix}ring_momentum",
    )
    energy_loss = st.number_input(
        "Energy Loss per Turn (keV)",
        value=float(st.session_state.get(f"{prefix}ring_eloss_kev", ring_params["energy_loss_per_turn"] * 1e6)),
        min_value=0.0, max_value=10000.0, format="%.2f",
        key=f"{prefix}ring_eloss_kev",
    )
    harmonic_number = st.number_input(
        "Harmonic Number",
        value=int(st.session_state.get(f"{prefix}ring_harmonic", ring_params["harmonic_number"])),
        min_value=1, max_value=10000,
        key=f"{prefix}ring_harmonic",
    )
    damping_time = st.number_input(
        "Damping Time (s)",
        value=float(st.session_state.get(f"{prefix}ring_damping", ring_params["damping_time"])),
        min_value=0.0001, max_value=1.0, format="%.6f",
        key=f"{prefix}ring_damping",
    )
    energy_spread = st.number_input(
        "Energy Spread",
        value=float(st.session_state.get(f"{prefix}ring_espread", ring_params.get("energy_spread", 0.001))),
        min_value=0.00001, max_value=0.01, format="%.6f",
        key=f"{prefix}ring_espread",
        help="Relative energy spread (σ_δ).",
    )
    return circumference, energy, momentum_compaction, energy_loss, harmonic_number, damping_time, energy_spread


def _render_cavity_params(prefix: str = ""):
    """Render MC + HC parameter inputs and return values."""
    mc_params = preset["main_cavity"]
    hc_params = preset["harmonic_cavity"]

    st.subheader("Main Cavity")
    mc_voltage = st.number_input(
        "Voltage (MV)", value=float(st.session_state.get(f"{prefix}mc_voltage", mc_params["voltage"])),
        min_value=0.0, max_value=10.0, format="%.3f", key=f"{prefix}mc_voltage",
    )
    mc_frequency = st.number_input(
        "Frequency (MHz)", value=float(st.session_state.get(f"{prefix}mc_freq", mc_params["frequency"])),
        min_value=1.0, max_value=5000.0, format="%.3f", key=f"{prefix}mc_freq",
    )
    mc_harmonic = st.number_input(
        "Harmonic", value=int(st.session_state.get(f"{prefix}mc_harm", mc_params["harmonic"])),
        min_value=1, max_value=10000, key=f"{prefix}mc_harm",
    )
    mc_q = st.number_input(
        "Quality Factor Q", value=float(st.session_state.get(f"{prefix}mc_q", mc_params["Q"])),
        min_value=100.0, max_value=1000000.0, format="%.0f", key=f"{prefix}mc_q",
    )
    mc_roq = st.number_input(
        "R/Q (Ω)", value=float(st.session_state.get(f"{prefix}mc_roq", mc_params["R_over_Q"])),
        min_value=1.0, max_value=1000.0, format="%.1f", key=f"{prefix}mc_roq",
    )

    st.subheader("Harmonic Cavity")
    hc_voltage = st.number_input(
        "Voltage (MV)", value=float(st.session_state.get(f"{prefix}hc_voltage", hc_params["voltage"])),
        min_value=0.0, max_value=10.0, format="%.3f", key=f"{prefix}hc_voltage",
    )

    # Harmonic ratio
    try:
        mc_f = st.session_state.get(f"{prefix}mc_freq", float(mc_params["frequency"]))
        hc_f = float(hc_params["frequency"])
        initial_ratio = hc_f / mc_f if mc_f > 0 else 4.0
    except Exception:
        initial_ratio = 4.0

    hc_ratio = st.number_input(
        "Harmonic Ratio (n)", value=float(round(initial_ratio)),
        min_value=1.0, max_value=100.0, step=1.0, format="%.1f", key=f"{prefix}hc_ratio",
    )
    hc_frequency = mc_frequency * hc_ratio
    hc_harmonic = int(mc_harmonic * hc_ratio)
    st.info(f"Frequency: **{fmt(hc_frequency, 3)} MHz** · Harmonic: **{hc_harmonic}**")

    hc_q = st.number_input(
        "Quality Factor Q", value=float(st.session_state.get(f"{prefix}hc_q", hc_params["Q"])),
        min_value=100.0, max_value=1000000.0, format="%.0f", key=f"{prefix}hc_q",
    )
    hc_roq = st.number_input(
        "R/Q (Ω)", value=float(st.session_state.get(f"{prefix}hc_roq", hc_params["R_over_Q"])),
        min_value=1.0, max_value=1000.0, format="%.1f", key=f"{prefix}hc_roq",
    )
    return (mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq,
            hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq)


def _make_objects(circumference, energy, momentum_compaction, energy_loss,
                  harmonic_number, damping_time,
                  mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq,
                  hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq):
    """Create ring/cavity wrapper objects from UI values."""
    ring = create_ring_from_params(
        circumference, energy, momentum_compaction,
        energy_loss / 1e6, harmonic_number, damping_time,
    )
    main_cavity = create_cavity_from_params(mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq)
    harmonic_cavity = create_cavity_from_params(hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq)
    return ring, main_cavity, harmonic_cavity


# ═══════════════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ═══════════════════════════════════════════════════════════════════════

main_tab1, main_tab2, main_tab3 = st.tabs([
    "📊 Parameter Scans",
    "🎯 R-Factor Optimization",
    "🔬 Mode Analysis",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — PARAMETER SCANS
# ═══════════════════════════════════════════════════════════════════════

with main_tab1:
    st.header("📊 Parameter Scans")
    st.markdown("Explore stability regions across 2D parameter spaces.")

    scan_tabs = st.tabs(["⚙️ Parameters", "💾 Save Config", "▶️ Run Scan", "📈 Results"])

    # ── Parameters ──
    with scan_tabs[0]:
        st.subheader("⚙️ Configuration Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ring Parameters")
            (circumference, energy, momentum_compaction, energy_loss,
             harmonic_number, damping_time, energy_spread) = _render_ring_params()
        with col2:
            (mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq,
             hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq) = _render_cavity_params()

    # ── Save Config ──
    with scan_tabs[1]:
        st.subheader("💾 Save Configuration")
        st.markdown("Save your current parameter configuration for easy reuse later.")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.subheader("Save As New Configuration")
            save_name = st.text_input("Configuration Name", value="my_config")
            accelerator_name = st.selectbox(
                "Accelerator", options=["Aladdin", "SOLEIL II", "Custom"],
                key="save_accel"
            )
            source_from_preset = st.checkbox("Based on preset", value=True)
            if source_from_preset:
                source_preset = st.selectbox("Source Preset", options=get_preset_names(), key="save_source")
            else:
                source_preset = None

            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                config_data = {
                    "ring": {
                        "circumference": circumference,
                        "energy": energy,
                        "momentum_compaction": momentum_compaction,
                        "energy_loss_per_turn": energy_loss / 1e6,
                        "harmonic_number": harmonic_number,
                        "damping_time": damping_time,
                    },
                    "main_cavity": {
                        "voltage": mc_voltage, "frequency": mc_frequency,
                        "harmonic": mc_harmonic, "Q": mc_q, "R_over_Q": mc_roq,
                    },
                    "harmonic_cavity": {
                        "voltage": hc_voltage, "frequency": hc_frequency,
                        "harmonic": hc_harmonic, "Q": hc_q, "R_over_Q": hc_roq,
                    },
                    "current": 0.2, "passive_hc": passive_hc,
                }
                manager = ConfigManager()
                manager.save_config(save_name, accelerator_name, config_data, source_preset)
                manager.save_session_config(save_name, accelerator_name)
                st.success(f"✅ Configuration saved as '{accelerator_name}_{save_name}'!")

        with s_col2:
            st.subheader("Quick Update Current Config")
            if st.session_state.get("current_config"):
                st.info(
                    f"**Current:** {st.session_state.get('current_config', 'Unnamed')} "
                    f"({st.session_state.get('current_accelerator', 'Custom')})"
                )

    # ── Run Scan ──
    with scan_tabs[2]:
        st.subheader("▶️ Run Scan")

        scan_type = st.selectbox(
            "Scan Type", options=["Psi vs Current", "Psi vs R/Q"],
            help="Choose which parameters to scan", key="scan_type_select"
        )

        scan_params = preset.get("scan_params", {})

        if scan_type == "Psi vs Current":
            c1, c2, c3 = st.columns(3)
            with c1:
                psi_min = st.number_input("Psi Min (°)", value=float(scan_params.get("psi_min", 1.0)),
                                          min_value=0.1, max_value=180.0, format="%.1f", step=0.1)
            with c2:
                psi_max = st.number_input("Psi Max (°)", value=float(scan_params.get("psi_max", 180.0)),
                                          min_value=-180.0, max_value=180.0, format="%.1f")
            with c3:
                psi_points = st.slider("Psi Points", 5, 100, int(scan_params.get("psi_points", 30)))

            c4, c5, c6 = st.columns(3)
            with c4:
                current_min = st.number_input("Current Min (A)", value=float(scan_params.get("current_min", 0.001)),
                                              min_value=0.0001, format="%.4f")
            with c5:
                current_max = st.number_input("Current Max (A)", value=float(scan_params.get("current_max", 0.5)),
                                              min_value=0.001, format="%.4f")
            with c6:
                current_points = st.slider("Current Points", 5, 100, int(scan_params.get("current_points", 30)))
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                psi_min = st.number_input("Psi Min (°)", value=float(scan_params.get("psi_min", 1.0)),
                                          min_value=0.1, max_value=180.0, format="%.1f", step=0.1, key="psi_min_roq")
            with c2:
                psi_max = st.number_input("Psi Max (°)", value=float(scan_params.get("psi_max", 180.0)),
                                          min_value=-180.0, max_value=180.0, format="%.1f", key="psi_max_roq")
            with c3:
                psi_points = st.slider("Psi Points", 5, 100, 30, key="psi_pts_roq")

            c4, c5, c6 = st.columns(3)
            with c4:
                roq_min = st.number_input("R/Q Min (Ω)", value=1.0, min_value=0.1, format="%.1f")
            with c5:
                roq_max = st.number_input("R/Q Max (Ω)", value=200.0, min_value=1.0, format="%.1f")
            with c6:
                roq_points = st.slider("R/Q Points", 5, 100, 30, key="roq_pts")

            current_fixed = st.number_input("Fixed Current (A)", value=0.2, min_value=0.001, format="%.3f")

        st.markdown("---")

        # Show selected method
        backend = _backend_method(method)
        st.info(
            f"**Method:** {method}"
            + (f" → backend: *{backend}*" if method == "Gamelin" else "")
            + f"  ·  **Passive HC:** {passive_hc}"
        )

        if st.button("🚀 Run Scan", type="primary", use_container_width=True):
            if psi_min >= psi_max:
                st.error("❌ Psi Min must be less than Psi Max!")
            else:
                with st.spinner("Running scan … this may take a few minutes."):
                    try:
                        ring, main_cavity, harmonic_cavity = _make_objects(
                            circumference, energy, momentum_compaction, energy_loss,
                            harmonic_number, damping_time,
                            mc_voltage, mc_frequency, mc_harmonic, mc_q, mc_roq,
                            hc_voltage, hc_frequency, hc_harmonic, hc_q, hc_roq,
                        )

                        if backend == "Hofmann":
                            st.warning("Hofmann model does not support full parameter scans in the same way. It is available under the Double RF System dashboard.")
                            result = {"success": False, "error": "Hofmann model parameter scans not implemented. Use Double RF System dashboard instead."}
                        elif scan_type == "Psi vs Current":
                            result = run_psi_current_scan(
                                ring, main_cavity, harmonic_cavity,
                                psi_range=(psi_min, psi_max, psi_points),
                                current_range=(current_min, current_max, current_points),
                                method=backend,
                                passive_hc=passive_hc,
                            )
                        else:
                            result = run_psi_roq_scan(
                                ring, main_cavity, harmonic_cavity,
                                current=current_fixed,
                                psi_range=(psi_min, psi_max, psi_points),
                                roq_range=(roq_min, roq_max, roq_points),
                                method=backend,
                                passive_hc=passive_hc,
                            )

                        st.session_state["scan_results"] = result
                        st.session_state["scan_type"] = scan_type

                        if result["success"]:
                            st.success("✅ Scan completed successfully!")
                            if result.get("fallback_used", False):
                                st.warning(
                                    "⚠️ Low convergence with selected method. "
                                    "Results obtained via Bosch fallback."
                                )
                            st.info("Switch to the **Results** tab to view the plots.")
                        else:
                            st.error(f"❌ Scan failed: {result.get('error', 'Unknown')}")
                            tb = result.get("traceback", "")
                            if tb:
                                with st.expander("📋 Full Error Traceback"):
                                    st.code(tb, language="python")
                            hint = result.get("hint", "")
                            if hint:
                                st.info(f"**Hint:** {hint}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

    # ── Results ──
    with scan_tabs[3]:
        st.subheader("📈 Scan Results")

        if "scan_results" in st.session_state:
            result = st.session_state["scan_results"]
            scan_type_result = st.session_state.get("scan_type", "Psi vs Current")

            if result["success"]:
                st.success("✅ Results from last scan")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Psi Points", len(result.get("psi_vals", [])))
                with col2:
                    if scan_type_result == "Psi vs Current":
                        st.metric("Current Points", len(result.get("current_vals", [])))
                    else:
                        st.metric("R/Q Points", len(result.get("roq_vals", [])))
                with col3:
                    p = len(result.get("psi_vals", [])) * len(result.get("current_vals", result.get("roq_vals", [])))
                    st.metric("Total Points", p)

                scan_results = result.get("results", {})

                if scan_results and len(scan_results) > 0:
                    has_valid_data = False
                    for key, data in scan_results.items():
                        if hasattr(data, "size") and hasattr(data, "__iter__"):
                            if np.sum(np.isfinite(data.flatten())) > 0:
                                has_valid_data = True
                                break

                    if has_valid_data:
                        rtab0, rtab1, rtab2, rtab3 = st.tabs(
                            ["📊 Stability Regions", "🌡️ Heatmaps", "📈 Bunch Length", "⏱️ Touschek Lifetime"]
                        )

                        with rtab0:
                            st.subheader("Stability Analysis Map")
                            try:
                                y_vals = result.get("current_vals", result.get("roq_vals", []))
                                y_label = "Beam current I0 [mA]" if scan_type_result == "Psi vs Current" else "R/Q [Ω]"

                                if "xi" in scan_results and "robinson_coup" in scan_results:
                                    converged = scan_results.get("converged_coup")
                                    if converged is not None:
                                        total_pts = converged.size
                                        conv_pts = np.sum(converged)
                                        pct = 100 * conv_pts / total_pts if total_pts > 0 else 0
                                        if pct < 50:
                                            st.warning(f"⚠️ Low convergence: {fmt(pct,1)}%")
                                        elif pct < 80:
                                            st.info(f"ℹ️ Convergence: {fmt(pct,1)}%")
                                        else:
                                            st.success(f"✅ Good convergence: {fmt(pct,1)}%")

                                    fig_stab = plot_stability_regions(
                                        psi_vals=result["psi_vals"], y_vals=y_vals,
                                        results=scan_results,
                                        x_label="Harmonic cavity tuning angle ψ₂ [deg]",
                                        y_label=y_label, title="Stability Map",
                                        mode_coupling=True,
                                    )
                                    st.plotly_chart(fig_stab, use_container_width=True)

                                    with st.expander("ℹ️ Legend Explanation"):
                                        st.markdown(
                                            "- **Xi Isoline** — ξ contours\n"
                                            "- **CBI driven by HOMs (▲)** — Coupled Bunch Instability\n"
                                            "- **Dipole Robinson (●)**\n"
                                            "- **Quadrupole Robinson (▼)**\n"
                                            "- **Fast mode-coupling (★)**\n"
                                            "- **Zero-frequency (♦)**\n"
                                            "- **PTBL (X)** — Periodic Transient Beam Loading\n"
                                            "- **Stable beam (○)**"
                                        )
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                        with rtab1:
                            try:
                                plot_data = None
                                if "robinson_coup" in scan_results:
                                    rc = scan_results["robinson_coup"]
                                    if hasattr(rc, "shape") and len(rc.shape) == 3:
                                        plot_data = np.where(np.isfinite(rc[:, :, 0]), rc[:, :, 0], 0.0)
                                if plot_data is not None and np.sum(np.isfinite(plot_data)) > 0:
                                    fig = plot_2d_heatmap(
                                        x_vals=result["psi_vals"],
                                        y_vals=result.get("current_vals", result.get("roq_vals", [])),
                                        z_vals=plot_data,
                                        x_label="Psi (degrees)",
                                        y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                        z_label="Growth Rate", title="Growth Rate Map", colorscale="RdBu_r",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No valid growth rate data.")
                            except Exception as e:
                                st.error(f"❌ {e}")

                        with rtab2:
                            try:
                                bl = scan_results.get("bl")
                                if bl is not None and np.sum(np.isfinite(bl)) > 0:
                                    bl_clean = np.where(np.isfinite(bl), bl, 0.0)
                                    fig = plot_2d_heatmap(
                                        x_vals=result["psi_vals"],
                                        y_vals=result.get("current_vals", result.get("roq_vals", [])),
                                        z_vals=bl_clean,
                                        x_label="Psi (degrees)",
                                        y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                        z_label="Bunch Length (ps)", title="Bunch Length Map", colorscale="Viridis",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No valid bunch length data.")
                            except Exception as e:
                                st.error(f"❌ {e}")

                        with rtab3:
                            try:
                                r_data = scan_results.get("R")
                                if r_data is not None and np.sum(np.isfinite(r_data)) > 0:
                                    r_clean = np.where(np.isfinite(r_data), r_data, 0.0)
                                    fig = plot_2d_heatmap(
                                        x_vals=result["psi_vals"],
                                        y_vals=result.get("current_vals", result.get("roq_vals", [])),
                                        z_vals=r_clean,
                                        x_label="Psi (degrees)",
                                        y_label="Current (A)" if scan_type_result == "Psi vs Current" else "R/Q (Ω)",
                                        z_label="R-Factor", title="Touschek Lifetime Map", colorscale="Plasma",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("No valid Touschek lifetime data.")
                            except Exception as e:
                                st.error(f"❌ {e}")

                        with st.expander("📋 Full Results Summary"):
                            for key in scan_results.keys():
                                data = scan_results[key]
                                if hasattr(data, "shape"):
                                    st.write(f"• **{key}**: shape {data.shape}, dtype {data.dtype}")
                    else:
                        st.warning("Scan completed but results contain mostly NaN/Inf.")
            else:
                st.error(f"❌ Last scan failed: {result.get('error', 'Unknown')}")
                if "hint" in result:
                    st.info(f"**Hint:** {result['hint']}")
        else:
            st.info("👈 Configure parameters and run a scan to see results here.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — R-FACTOR OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

with main_tab2:
    st.header("🎯 R-Factor Optimization")
    st.markdown("Find optimal harmonic cavity phase to maximize Touschek lifetime.")

    opt_tabs = st.tabs(["⚙️ Parameters", "🎯 Optimize", "📊 Results"])

    with opt_tabs[0]:
        st.subheader("Configuration Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ring Parameters")
            (opt_circ, opt_energy, opt_ac, opt_eloss,
             opt_h, opt_tau, _) = _render_ring_params(prefix="opt_")
            opt_current = st.number_input(
                "Beam Current (A)",
                value=float(preset.get("current", 0.2)),
                min_value=0.0, max_value=10.0, format="%.3f", key="opt_current",
            )
        with col2:
            (opt_mc_v, opt_mc_f, opt_mc_m, opt_mc_q, opt_mc_roq,
             opt_hc_v, opt_hc_f, opt_hc_m, opt_hc_q, opt_hc_roq) = _render_cavity_params(prefix="opt_")

    with opt_tabs[1]:
        st.subheader("Run Optimization")
        st.info(
            "The optimization searches for the HC phase (ψ) that maximizes the R-factor "
            "(Touschek lifetime enhancement)."
        )

        oc1, oc2 = st.columns(2)
        with oc1:
            st.subheader("Initial Guess")
            psi0 = st.number_input("Initial Psi (°)", value=90.0, min_value=-180.0, max_value=180.0, format="%.1f")
        with oc2:
            st.subheader("Search Bounds")
            opt_psi_min = st.number_input("Min Psi (°)", value=60.0, min_value=0.1, max_value=180.0, format="%.1f", step=0.1)
            opt_psi_max = st.number_input("Max Psi (°)", value=90.0, min_value=-180.0, max_value=180.0, format="%.1f")

        equilibrium_only = st.checkbox("Equilibrium Only (faster)", value=False)

        backend = _backend_method(method)
        st.info(f"**Method:** {method}" + (f" → *{backend}*" if method == "Gamelin" else ""))

        if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
            if opt_psi_min >= opt_psi_max:
                st.error("❌ Min Psi must be less than Max Psi!")
            else:
                with st.spinner("Running optimization…"):
                    try:
                        ring, mc, hc = _make_objects(
                            opt_circ, opt_energy, opt_ac, opt_eloss, opt_h, opt_tau,
                            opt_mc_v, opt_mc_f, opt_mc_m, opt_mc_q, opt_mc_roq,
                            opt_hc_v, opt_hc_f, opt_hc_m, opt_hc_q, opt_hc_roq,
                        )
                        if backend == "Hofmann":
                            st.warning("Hofmann model is an analytical tool and does not support automated scanning or optimization here.")
                            result = {"success": False, "error": "Hofmann model not supported for optimization. Use the Double RF System Dashboard."}
                        else:
                            result = run_optimization(
                                ring, mc, hc, current=opt_current, psi0=psi0,
                                bounds=(opt_psi_min, opt_psi_max),
                                method=backend, equilibrium_only=equilibrium_only,
                            )
                        st.session_state["opt_results"] = result
                        if result["success"]:
                            st.success("✅ Optimization completed!")
                            st.info("Switch to **Results** tab.")
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown')}")
                    except Exception as e:
                        st.error(f"❌ {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

    with opt_tabs[2]:
        st.subheader("Optimization Results")
        if "opt_results" in st.session_state:
            result = st.session_state["opt_results"]
            if result["success"]:
                st.success("✅ Optimization Results")
                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    st.metric("Initial Psi", fmt(result["psi0"]) + "°")
                with oc2:
                    st.metric("Optimal Psi", fmt(result["optimal_psi"]) + "°",
                              delta=fmt(result["optimal_psi"] - result["psi0"]) + "°")
                with oc3:
                    if "r_factor" in result:
                        st.metric("R-Factor", fmt(result["r_factor"], 4))

                fig = plot_optimization_result(
                    psi0=result["psi0"], optimal_psi=result["optimal_psi"],
                    bounds=(opt_psi_min, opt_psi_max) if "opt_psi_min" in dir() else (0, 180),
                    r_factor=result.get("r_factor"),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 💡 Recommendations")
                st.info(
                    f"**Optimal:** Set HC phase to **{fmt(result['optimal_psi'])}°** "
                    f"(R = {fmt(result.get('r_factor', 0), 4)})."
                )
            else:
                st.error(f"Failed: {result.get('error')}")
        else:
            st.info("👈 Run optimization first.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — ROBINSON MODE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

with main_tab3:
    st.header("🔬 Robinson Mode Analysis")
    st.markdown("Track Robinson modes and analyze instabilities across parameter ranges.")

    mode_tabs = st.tabs(["⚙️ Parameters", "▶️ Run Analysis", "📈 Results"])

    with mode_tabs[0]:
        st.subheader("Configuration Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ring Parameters")
            (m_circ, m_energy, m_ac, m_eloss,
             m_h, m_tau, _) = _render_ring_params(prefix="mode_")
            m_current = st.number_input(
                "Beam Current (A)",
                value=float(preset.get("current", 0.2)),
                min_value=0.0, max_value=10.0, format="%.3f", key="mode_current",
            )
        with col2:
            (m_mc_v, m_mc_f, m_mc_m, m_mc_q, m_mc_roq,
             m_hc_v, m_hc_f, m_hc_m, m_hc_q, m_hc_roq) = _render_cavity_params(prefix="mode_")

    with mode_tabs[1]:
        st.subheader("Run Mode Analysis")
        st.info(
            "This analysis tracks Robinson modes across a range of HC phases, "
            "showing mode frequencies, growth rates, and potential instabilities."
        )

        scan_params = preset.get("scan_params", {})
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            m_psi_min = st.number_input("Min Psi (°)", value=float(scan_params.get("psi_min", 1.0)),
                                        min_value=0.1, max_value=180.0, format="%.1f", step=0.1, key="mode_psi_min")
        with mc2:
            m_psi_max = st.number_input("Max Psi (°)", value=float(scan_params.get("psi_max", 180.0)),
                                        min_value=-180.0, max_value=180.0, format="%.1f", key="mode_psi_max")
        with mc3:
            m_psi_pts = st.slider("Points", 10, 100, int(scan_params.get("psi_points", 50)), key="mode_psi_points")

        backend = _backend_method(method)
        st.info(f"**Method:** {method}" + (f" → *{backend}*" if method == "Gamelin" else ""))

        if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
            if m_psi_min >= m_psi_max:
                st.error("❌ Min Psi must be less than Max Psi!")
            else:
                with st.spinner("Analyzing Robinson modes…"):
                    try:
                        ring, mc, hc = _make_objects(
                            m_circ, m_energy, m_ac, m_eloss, m_h, m_tau,
                            m_mc_v, m_mc_f, m_mc_m, m_mc_q, m_mc_roq,
                            m_hc_v, m_hc_f, m_hc_m, m_hc_q, m_hc_roq,
                        )
                        if backend == "Hofmann":
                            st.warning("Hofmann model does not support automated mode analysis scanning here.")
                            result = {"success": False, "error": "Hofmann model not supported for mode analysis. Use the Double RF System Dashboard."}
                        else:
                            result = analyze_robinson_modes(
                                ring, mc, hc, current=m_current,
                                psi_range=(m_psi_min, m_psi_max, m_psi_pts),
                                method=backend, passive_hc=passive_hc,
                            )
                        st.session_state["mode_results"] = result
                        if result["success"]:
                            st.success("✅ Analysis completed!")
                            st.info("Switch to the **Results** tab.")
                        else:
                            st.error(f"❌ {result.get('error')}")
                    except Exception as e:
                        st.error(f"❌ {e}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

    with mode_tabs[2]:
        st.subheader("Mode Analysis Results")
        if "mode_results" in st.session_state:
            result = st.session_state["mode_results"]
            if result["success"]:
                st.success("✅ Mode Analysis Results")

                psi_vals = result["psi_vals"]
                scan_results = result["results"]

                try:
                    (zero_freq_coup, robinson_coup, modes_coup, HOM_coup, xi,
                     converged_coup, PTBL_coup, bl, R) = scan_results

                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.metric("Psi Range", f"{fmt(psi_vals[0],1)}° → {fmt(psi_vals[-1],1)}°")
                    with mc2:
                        st.metric("Points", len(psi_vals))
                    with mc3:
                        conv = np.sum(np.any(converged_coup, axis=1))
                        st.metric("Converged", f"{conv}/{len(psi_vals)}")

                    # ── Mode Frequencies ──
                    st.markdown("### Mode Frequencies")
                    n_modes = modes_coup.shape[1]
                    mode_labels = [f"Mode {i+1}" for i in range(n_modes)]
                    mode_freqs = [modes_coup[:, i] for i in range(n_modes)]
                    fig_freq = plot_mode_frequencies(
                        psi_vals, mode_freqs, mode_labels=mode_labels,
                        title="Robinson Mode Frequencies vs ψ",
                    )
                    st.plotly_chart(fig_freq, use_container_width=True)

                    # ── Growth Rates ──
                    st.markdown("### Growth Rates")
                    growth_rates = [np.imag(modes_coup[:, i]) for i in range(n_modes)]
                    fig_growth = plot_growth_rates(
                        psi_vals, growth_rates, mode_labels=mode_labels,
                        title="Mode Growth Rates vs ψ",
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)

                    # ── Instability Summary ──
                    st.markdown("### Instability Summary")
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    with ic1:
                        d = np.sum(robinson_coup[:, 0])
                        st.metric("Dipole Robinson", f"{d} pts",
                                  delta="Unstable" if d > 0 else "Stable", delta_color="inverse")
                    with ic2:
                        q = np.sum(robinson_coup[:, 1])
                        st.metric("Quadrupole Robinson", f"{q} pts",
                                  delta="Unstable" if q > 0 else "Stable", delta_color="inverse")
                    with ic3:
                        h = np.sum(HOM_coup)
                        st.metric("HOM", f"{h} pts",
                                  delta="Unstable" if h > 0 else "Stable", delta_color="inverse")
                    with ic4:
                        p = np.sum(PTBL_coup)
                        st.metric("PTBL", f"{p} pts",
                                  delta="Unstable" if p > 0 else "Stable", delta_color="inverse")

                    # ── Additional Metrics ──
                    st.markdown("### Additional Metrics")
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        fig_r = plot_r_factor_vs_psi(psi_vals, R, title="R-Factor vs ψ")
                        st.plotly_chart(fig_r, use_container_width=True)
                    with ac2:
                        fig_bl = go.Figure()
                        fig_bl.add_trace(go.Scatter(
                            x=psi_vals, y=bl, mode="lines+markers", name="Bunch Length",
                            line=dict(color="cyan", width=2), marker=dict(size=4),
                            hovertemplate="ψ: %{x}°<br>BL: %{y:.2f} ps<extra></extra>",
                        ))
                        fig_bl.update_layout(
                            title="Bunch Length vs ψ",
                            xaxis_title="ψ (degrees)", yaxis_title="Bunch Length (ps)",
                            template="plotly_dark", height=400,
                        )
                        st.plotly_chart(fig_bl, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing results: {e}")
                    import traceback
                    with st.expander("Details"):
                        st.code(traceback.format_exc())
                        if isinstance(scan_results, tuple):
                            for i, item in enumerate(scan_results):
                                st.write(f"[{i}] {type(item).__name__} shape={getattr(item,'shape','N/A')}")
            else:
                st.error(f"Analysis failed: {result.get('error')}")
        else:
            st.info("👈 Run analysis first.")
