import streamlit as st
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add project root to path for presets
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.presets import get_preset, get_preset_names

try:
    import paramiko
    import stat
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

st.set_page_config(
    page_title="mbTrack2 Remote Job",
    page_icon="🚀",
    layout="wide"
)

if not st.session_state.get("authentication_status"):
    st.info("Please login from the Home page.")
    st.stop()


if "ssh_connected" not in st.session_state:
    st.session_state.ssh_connected = False
if "ssh_pwd_temp" not in st.session_state:
    st.session_state.ssh_pwd_temp = ""

SIM_KEYS = [
    "IDs", "wake", "P_FB", "PI_FB", "Tuner_FB", "Direct_FB", "spec", "mpi_flag", "sweep_bool", "remove_higher_mcf",
    "fill_type", "n_gaps", "gap_len", "Itot", "Itot_final", "mp", "tot_turns",
    "Vc", "detune_deg", "MC_det", "HC_det_end", "xi_start", "xi_end",
    "ContinuousTuning", "Det_calc_Xi", "Injection", "Set_optimal_detune", "Injection_optimal_detune", 
    "fb_gain_real", "fb_gain_imag", "directFB_gain", "directFB_phaseShift",
    "mode0_damper", "mode0_gain", "mode0_delay", "mode0_phase", 
    "bbb_lfb", "lfb_gain", "lfb_tap", "lfb_delay", "lfb_phase"
]

st.title("🚀 mbTrack2 Remote Job Submission")
st.markdown("Configure mbTrack2 parameters, submit it directly to a remote server, monitor progress via `.o` log files, and fetch your HDF5 simulation results.")

if not PARAMIKO_AVAILABLE:
    st.warning("⚠️ `paramiko` library is required for SSH connections. Please install it using `pip install paramiko`.")

with st.sidebar:
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/0_🔧_Double_RF_System.py", label="Double RF System", icon="🔧")
    st.page_link("pages/1_📈_Semi_Analytic.py", label="Semi-Analytic Tools", icon="📈")
    st.page_link("pages/2_🚀_MBTrack2_Remote.py", label="MBTrack2 Remote Job", icon="🚀")

    st.markdown("---")
    st.markdown("### 1. SSH Server Settings")
    
    # Load SSH Profiles
    import json
    username = st.session_state.get("username", "default")
    profile_path = project_root / f"ssh_profiles_{username}.json"
    def load_profiles():
        if profile_path.exists():
            with open(profile_path, "r") as f: return json.load(f)
        return {}
    
    profiles = load_profiles()

    def apply_profile():
        sel = st.session_state.profile_selector
        if sel and sel != "+ New Profile":
            pd = profiles.get(sel, {})
            # Server settings
            st.session_state.ssh_host = pd.get("host", "localhost")
            st.session_state.ssh_port = pd.get("port", 22)
            st.session_state.ssh_user = pd.get("user", "")
            # Job Rules
            if "remote_base_dir" in pd: st.session_state.remote_base_dir = pd["remote_base_dir"]
            if "job_system" in pd: st.session_state.job_system = pd["job_system"]
            if "slurm_partition" in pd: st.session_state.slurm_partition = pd["slurm_partition"]
            if "slurm_nodes" in pd: st.session_state.slurm_nodes = pd["slurm_nodes"]
            if "slurm_tasks" in pd: st.session_state.slurm_tasks = pd["slurm_tasks"]
            if "slurm_time" in pd: st.session_state.slurm_time = pd["slurm_time"]
            # Simulation settings
            params = pd.get("params", {})
            for k in SIM_KEYS:
                if f"mb2_{k}" in params:
                    st.session_state[f"mb2_{k}"] = params[f"mb2_{k}"]
            st.session_state.ssh_connected = False

    selected_profile_name = st.selectbox("Saved Profiles", list(profiles.keys()) + ["+ New Profile"], key="profile_selector", on_change=apply_profile)

    if "ssh_host" not in st.session_state:
        st.session_state.ssh_host = st.session_state.get('ssh_host_last', 'localhost')
    if "ssh_port" not in st.session_state:
        st.session_state.ssh_port = 22
    if "ssh_user" not in st.session_state:
        st.session_state.ssh_user = st.session_state.get('ssh_user_last', '')

    ssh_host = st.text_input("Hostname / IP", key="ssh_host")
    ssh_port = st.number_input("Port", step=1, key="ssh_port")
    ssh_user = st.text_input("Username", key="ssh_user")
    
    new_profile_name = st.text_input("Profile Name (to save)")
    if st.button("Save Profile"):
        if new_profile_name:
            pd = {"host": ssh_host, "port": ssh_port, "user": ssh_user}
            pd["remote_base_dir"] = st.session_state.get("remote_base_dir", "~/mbtrack2_jobs")
            pd["job_system"] = st.session_state.get("job_system", "Direct (nohup)")
            pd["slurm_partition"] = st.session_state.get("slurm_partition", "compute")
            pd["slurm_nodes"] = st.session_state.get("slurm_nodes", 1)
            pd["slurm_tasks"] = st.session_state.get("slurm_tasks", 40)
            pd["slurm_time"] = st.session_state.get("slurm_time", "24:00:00")
            
            p_params = {}
            for k in SIM_KEYS:
                if f"mb2_{k}" in st.session_state:
                    p_params[f"mb2_{k}"] = st.session_state[f"mb2_{k}"]
            pd["params"] = p_params
            
            profiles[new_profile_name] = pd
            with open(profile_path, "w") as f: json.dump(profiles, f)
            st.success(f"Saved {new_profile_name}!")
            st.rerun()

    st.markdown("### 2. Job Execution Rules")
    if "remote_base_dir" not in st.session_state: st.session_state.remote_base_dir = "~/mbtrack2_jobs"
    remote_base_dir = st.text_input("Remote Base Directory", key="remote_base_dir")
    st.caption("Note: Jobs will be placed in a unique folder inside this base path.")
    if "job_system" not in st.session_state: st.session_state.job_system = "Direct (nohup)"
    job_system = st.selectbox("Execution Method", ["Direct (nohup)", "SLURM (sbatch)"], key="job_system")
    if job_system == "SLURM (sbatch)":
        if "slurm_partition" not in st.session_state: st.session_state.slurm_partition = "compute"
        slurm_partition = st.text_input("Partition", key="slurm_partition")
        if "slurm_nodes" not in st.session_state: st.session_state.slurm_nodes = 1
        slurm_nodes = st.number_input("Nodes", step=1, key="slurm_nodes")
        if "slurm_tasks" not in st.session_state: st.session_state.slurm_tasks = 40
        slurm_tasks = st.number_input("Tasks per node", step=1, key="slurm_tasks")
        if "slurm_time" not in st.session_state: st.session_state.slurm_time = "24:00:00"
        slurm_time = st.text_input("Walltime", key="slurm_time")
        
    st.markdown("---")
    st.markdown("### 📚 Reference Values (from Presets)")
    preset_names = get_preset_names()
    default_index = preset_names.index("SOLEIL II") if "SOLEIL II" in preset_names else 0
    ref_preset_name = st.selectbox("Compare your simulation with:", preset_names, index=default_index)
    ref_preset = get_preset(ref_preset_name)
    mc = ref_preset.get("main_cavity", {})
    hc = ref_preset.get("harmonic_cavity", {})
    st.info(f"""
    **Beam Current**: {ref_preset.get('current', '-')} A
    
    **Main Cavity:**
    - Vc: {mc.get('voltage', '-')} MV
    - Freq: {mc.get('frequency', '-')} MHz
    - Q0: {mc.get('Q0', mc.get('Q', '-'))} 
    
    **Harmonic Cavity:**
    - Vc: {hc.get('voltage', '-')} MV
    - n: {hc.get('harmonic_number', hc.get('harmonic', '-'))}
    """)

# --- Main Parameters ---
st.header("⚙️ Simulation Settings")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Flags")
    if "mb2_IDs" not in st.session_state: st.session_state.mb2_IDs = "open"
    IDs = st.selectbox("IDs", ["open", "close", "close_phase2"], key="mb2_IDs")
    if "mb2_wake" not in st.session_state: st.session_state.mb2_wake = True
    wake = st.checkbox("Wake", key="mb2_wake")
    if "mb2_P_FB" not in st.session_state: st.session_state.mb2_P_FB = False
    P_FB = st.checkbox("P_FB", key="mb2_P_FB")
    if "mb2_PI_FB" not in st.session_state: st.session_state.mb2_PI_FB = False
    PI_FB = st.checkbox("PI_FB", key="mb2_PI_FB")
    if "mb2_Tuner_FB" not in st.session_state: st.session_state.mb2_Tuner_FB = False
    Tuner_FB = st.checkbox("Tuner_FB", key="mb2_Tuner_FB")
    if "mb2_Direct_FB" not in st.session_state: st.session_state.mb2_Direct_FB = True
    Direct_FB = st.checkbox("Direct_FB", key="mb2_Direct_FB")
    if "mb2_spec" not in st.session_state: st.session_state.mb2_spec = True
    spec = st.checkbox("spec", key="mb2_spec")
    if "mb2_mpi_flag" not in st.session_state: st.session_state.mb2_mpi_flag = True
    mpi_flag = st.checkbox("mpi", key="mb2_mpi_flag")
    if "mb2_sweep_bool" not in st.session_state: st.session_state.mb2_sweep_bool = False
    sweep_bool = st.checkbox("sweep_bool", key="mb2_sweep_bool")
    if "mb2_remove_higher_mcf" not in st.session_state: st.session_state.mb2_remove_higher_mcf = False
    remove_higher_mcf = st.checkbox("remove_higher_mcf", key="mb2_remove_higher_mcf")

with col2:
    st.subheader("Filling & Beam")
    if "mb2_fill_type" not in st.session_state: st.session_state.mb2_fill_type = "uniform"
    fill_type = st.selectbox("Fill Type", ["uniform", "32b", "single", "gaps"], key="mb2_fill_type")
    if "mb2_n_gaps" not in st.session_state: st.session_state.mb2_n_gaps = 0
    n_gaps = st.number_input("Number of gaps", key="mb2_n_gaps")
    if "mb2_gap_len" not in st.session_state: st.session_state.mb2_gap_len = 0
    gap_len = st.number_input("Gap length", key="mb2_gap_len")
    if "mb2_Itot" not in st.session_state: st.session_state.mb2_Itot = 0.325
    Itot = st.number_input("Itot (A)", format="%.3f", key="mb2_Itot")
    if "mb2_Itot_final" not in st.session_state: st.session_state.mb2_Itot_final = 0.5
    Itot_final = st.number_input("Itot_final (A)", format="%.3f", key="mb2_Itot_final")
    if "mb2_mp" not in st.session_state: st.session_state.mb2_mp = int(1e5)
    mp = st.number_input("Macro-particles (mp)", step=1000, key="mb2_mp")
    if "mb2_tot_turns" not in st.session_state: st.session_state.mb2_tot_turns = int(100e3)
    tot_turns = st.number_input("Total turns", step=1000, key="mb2_tot_turns")

with col3:
    st.subheader("Cavity & Tuning")
    st.caption("Adjust using reference parameters on your left.")
    if "mb2_Vc" not in st.session_state: st.session_state.mb2_Vc = 1.7e6
    Vc = st.number_input("Vc (V)", format="%.1e", key="mb2_Vc")
    if "mb2_detune_deg" not in st.session_state: st.session_state.mb2_detune_deg = 68.3
    detune_deg = st.number_input("Detune deg", format="%.2f", key="mb2_detune_deg")
    if "mb2_MC_det" not in st.session_state: st.session_state.mb2_MC_det = -35e3
    MC_det = st.number_input("MC detune (Hz)", format="%.1e", key="mb2_MC_det")
    if "mb2_HC_det_end" not in st.session_state: st.session_state.mb2_HC_det_end = 3e3
    HC_det_end = st.number_input("HC detune end (Hz)", format="%.1e", key="mb2_HC_det_end")
    if "mb2_xi_start" not in st.session_state: st.session_state.mb2_xi_start = 1.0
    xi_start = st.number_input("xi_start", key="mb2_xi_start")
    if "mb2_xi_end" not in st.session_state: st.session_state.mb2_xi_end = 1.1
    xi_end = st.number_input("xi_end", key="mb2_xi_end")

col4, col5 = st.columns(2)
with col4:
    if "mb2_ContinuousTuning" not in st.session_state: st.session_state.mb2_ContinuousTuning = False
    ContinuousTuning = st.checkbox("ContinuousTuning", key="mb2_ContinuousTuning")
    if "mb2_Det_calc_Xi" not in st.session_state: st.session_state.mb2_Det_calc_Xi = False
    Det_calc_Xi = st.checkbox("Det_calc_Xi", key="mb2_Det_calc_Xi")
    if "mb2_Injection" not in st.session_state: st.session_state.mb2_Injection = False
    Injection = st.checkbox("Injection", key="mb2_Injection")
    if "mb2_Set_optimal_detune" not in st.session_state: st.session_state.mb2_Set_optimal_detune = True
    Set_optimal_detune = st.checkbox("Set_optimal_detune", key="mb2_Set_optimal_detune")
    if "mb2_Injection_optimal_detune" not in st.session_state: st.session_state.mb2_Injection_optimal_detune = False
    Injection_optimal_detune = st.checkbox("Injection_optimal_detune", key="mb2_Injection_optimal_detune")
with col5:
    if "mb2_fb_gain_real" not in st.session_state: st.session_state.mb2_fb_gain_real = 0.01
    fb_gain_real = st.number_input("FB Gain IQ Real", key="mb2_fb_gain_real")
    if "mb2_fb_gain_imag" not in st.session_state: st.session_state.mb2_fb_gain_imag = 1000.0
    fb_gain_imag = st.number_input("FB Gain IQ Imag", key="mb2_fb_gain_imag")
    if "mb2_directFB_gain" not in st.session_state: st.session_state.mb2_directFB_gain = 0.35
    directFB_gain = st.number_input("Direct FB Gain", key="mb2_directFB_gain")
    if "mb2_directFB_phaseShift" not in st.session_state: st.session_state.mb2_directFB_phaseShift = 0.0
    directFB_phaseShift = st.number_input("Direct FB Phase Shift (deg)", key="mb2_directFB_phaseShift")

fb_col1, fb_col2, fb_col3 = st.columns(3)
with fb_col1:
    st.subheader("Feedback Options")
    if "mb2_mode0_damper" not in st.session_state: st.session_state.mb2_mode0_damper = True
    mode0_damper = st.checkbox("Mode 0 Damper", key="mb2_mode0_damper")
    if "mb2_mode0_gain" not in st.session_state: st.session_state.mb2_mode0_gain = -10.0
    mode0_gain = st.number_input("Mode 0 Gain", key="mb2_mode0_gain")
    if "mb2_mode0_delay" not in st.session_state: st.session_state.mb2_mode0_delay = 0
    mode0_delay = int(st.number_input("Mode 0 Delay", step=1, key="mb2_mode0_delay"))
    if "mb2_mode0_phase" not in st.session_state: st.session_state.mb2_mode0_phase = 0.0
    mode0_phase = st.number_input("Mode 0 Phase Off.", key="mb2_mode0_phase")
with fb_col2:
    st.write(" ")
    st.write(" ")
    if "mb2_bbb_lfb" not in st.session_state: st.session_state.mb2_bbb_lfb = False
    bbb_lfb = st.checkbox("Bunch by Bunch LFB (FIR)", key="mb2_bbb_lfb")
    if "mb2_lfb_gain" not in st.session_state: st.session_state.mb2_lfb_gain = 0.01
    lfb_gain = st.number_input("LFB Gain", key="mb2_lfb_gain")
    if "mb2_lfb_tap" not in st.session_state: st.session_state.mb2_lfb_tap = 5
    lfb_tap = int(st.number_input("LFB Tap Number", step=1, key="mb2_lfb_tap"))
with fb_col3:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    if "mb2_lfb_delay" not in st.session_state: st.session_state.mb2_lfb_delay = 1
    lfb_delay = int(st.number_input("LFB Turn Delay", step=1, key="mb2_lfb_delay"))
    if "mb2_lfb_phase" not in st.session_state: st.session_state.mb2_lfb_phase = 0.0
    lfb_phase = st.number_input("LFB Phase", key="mb2_lfb_phase")

st.markdown("---")

def build_script_content(job_name_str):
    return f"""import numpy as np
from machine_data import v2366_v3, load_TDR2_wf
from mbtrack2 import Beam, CavityResonator, LongitudinalMap, SynchrotronRadiation
from mbtrack2 import BeamMonitor, CavityMonitor, BunchSpectrumMonitor, BeamSpectrumMonitor
from mbtrack2 import WakePotential, WakePotentialMonitor, TunerLoop, ProportionalIntegralLoop, DirectFeedback, ProportionalLoop
from mbtrack2.tracking.rf import SimpleRFMode0Damper
from mbtrack2.tracking.feedback import FIRDamper
from pathlib import Path

#% Simulation Settings
IDs = "{IDs}"
wake = {wake}
P_FB = {P_FB}
PI_FB = {PI_FB}
Tuner_FB = {Tuner_FB}
Direct_FB = {Direct_FB}
MODE0_DAMPER = {mode0_damper}
BBB_LFB = {bbb_lfb}
spec = {spec}
mpi = {mpi_flag}
sweep_bool = {sweep_bool}
remove_higher_mcf = {remove_higher_mcf}

fill_type = "{fill_type}"
n_gaps = {n_gaps}
gap_len = {gap_len}

ContinuousTuning = {ContinuousTuning}
Det_calc_Xi = {Det_calc_Xi}
Injection = {Injection}
HC_offset = True
HC_offset_Phasor = False
HC_offset_Phasor_power = False
impedance_offset = False
estimated_bunch_length = 40e-12
MCphase_update_during_tracking = False

Ncav_MC = 4
Set_optimal_detune = {Set_optimal_detune}
Injection_optimal_detune = {Injection_optimal_detune}

m_HC = 4
RoQ_HC = 30
Q0_HC = 31e3
QL_HC = 31e3
Ncav_HC = 2
HOM_factor = 1.0

Use_fixed_detuned_HC = False
fixed_detuned_HC = 425e3

Vc = {Vc}
Itot = {Itot}
detune_deg = {detune_deg}
HC_det = np.tan(detune_deg/180*np.pi)*1.4e9/2/QL_HC
HC_det_end = {HC_det_end}
MC_det = {MC_det}
xi_start = {xi_start}
xi_end = {xi_end}
Itot_final = {Itot_final}

if Det_calc_Xi:
    if isinstance(xi_start, (int,float)):
        iter_values = [xi_start]
    else:
        iter_values = xi_start
else:
    if isinstance(HC_det, (int,float)):
        iter_values = [HC_det]
    else:
        iter_values = HC_det
N_iter = len(iter_values)

fb_gain = [{fb_gain_real}, {fb_gain_imag}]
fb_sample_num = 208
fb_every = 208
fb_delay = 704
fb_IIR_cutoff = 0
directFB_gain = {directFB_gain}
directFB_phaseShift = {directFB_phaseShift}/180*np.pi
tuner_gain = 0.01
PFB_gainA = 0.01
PFB_gainP = 0.01
PFB_delay = 1

f0 = 400
f1 = 2000
t1 = 5000
level = 1e3
plane = "tau"
bunch_to_sweep = None

mp = int({mp})
tot_turns = int({tot_turns})
delay_before = int(0e3)
delay_after = int(0e3)
spec_step = int(50e3)
delay_before_sweep = int(50e3)
n_bin_wake = 80
tot_turns_iter = int(tot_turns*N_iter)

path = Path(__file__)

#%% Simulation setup
ring = v2366_v3(IDs=IDs, V_RF=Vc, HC_power=0)

if remove_higher_mcf:
    ring.mcf_order = ring.mcf_order[-1]

long = LongitudinalMap(ring)
rad = SynchrotronRadiation(ring)

#%% Setup monitors
spec_num = int(tot_turns_iter/spec_step)
name = "{job_name_str}"

MCmon = CavityMonitor("MC", ring, file_name=name, save_every=100, buffer_size=100, total_size=tot_turns_iter/100, mpi_mode=mpi)
HCmon = CavityMonitor("HHC", ring, file_name=None, save_every=100, buffer_size=100, total_size=tot_turns_iter/100, mpi_mode=mpi)
bbmon = BeamMonitor(h=ring.h, file_name=None, save_every=10, buffer_size=100, total_size=tot_turns_iter/10, mpi_mode=mpi)

if spec:
    beamspec = BeamSpectrumMonitor(ring, save_every=spec_step, buffer_size=1, total_size=spec_num, dim="tau", n_fft=None, file_name=None, mpi_mode=mpi)
    bunchspec = BunchSpectrumMonitor(ring, bunch_number=0, mp_number=mp, sample_size=1e3, save_every=spec_step, buffer_size=1, total_size=spec_num, dim="tau", n_fft=None, file_name=None, mpi_mode=mpi, higher_orders=True)

if wake:
    wakemon = WakePotentialMonitor(0, "Wlong", n_bin_wake, save_every=100, buffer_size=100, total_size=int(tot_turns_iter/100), mpi_mode=mpi)

if Use_fixed_detuned_HC:
    HC_fixed_mon = CavityMonitor("HC_fixed", ring, file_name=None, save_every=100, buffer_size=100, total_size=tot_turns_iter/100, mpi_mode=mpi)

for jj, val in enumerate(iter_values):
    if Det_calc_Xi:
        xi_start = val
    else:
        HC_det = val

    #%% Cavity setup
    MC = CavityResonator(ring, 1, 5e6, 35.7e3, 6e3, MC_det, Ncav=Ncav_MC)
    HHC = CavityResonator(ring, m_HC, RoQ_HC*Q0_HC, Q0_HC, QL_HC, MC_det, Ncav=Ncav_HC) # initial detune dummy, set below
    HHC.Vg = 0
    HHC.theta_g = 0
    HHC.detune = HC_det
        
    if MODE0_DAMPER:
        damper = SimpleRFMode0Damper(ring=ring, cavity=MC, gain={mode0_gain}, delay={mode0_delay}, phase_offset={mode0_phase}) 
    if BBB_LFB:
        lfb = FIRDamper(ring=ring, plane="s", turn_delay={lfb_delay}, tune=ring.synchrotron_tune({Vc}), tap_number={lfb_tap}, gain={lfb_gain}, phase={lfb_phase})

    HOM = np.load("/ccc/work/cont003/soleil/gamelina/Data_mbtrack2/SOLEIL_II_HOM_MC_fit.npz")
    HOM_list = []
    nHOM = len(HOM["saved_Rs"])
    for i in range(nHOM):
        fr = (HOM["saved_M"][i]*ring.h + HOM["saved_mu"][i] + ring.synchrotron_tune(Vc)) * ring.f0
        mt = fr/ring.f1
        HOMt = CavityResonator(ring, mt, HOM["saved_Rs"][i]*HOM_factor, HOM["saved_Q"][i], HOM["saved_Q"][i], 100, Ncav=1)
        HOMt.Vg = 0; HOMt.theta_g = 0; HOMt.fr = fr
        HOM_list.append(HOMt)

    if Use_fixed_detuned_HC:
        HC_fixed = CavityResonator(ring, m_HC, RoQ_HC*Q0_HC, Q0_HC, QL_HC, fixed_detuned_HC, Ncav=1)
        HC_fixed.Vg = 0
        HC_fixed.theta_g = 0
    
    #%% Wakes
    if wake:
        ID_wake = "open" if IDs == "open" else "close"
        wf = load_TDR2_wf(f"TDR2.1_ID{{ID_wake}}")
        wf.drop(["Zxdip","Wxdip","Zydip","Wydip"])
        wp = WakePotential(ring, wf, n_bin=n_bin_wake)
    
    #%% Beam setup
    bb = Beam(ring)
    if fill_type == "uniform":
        filling = np.ones(ring.h)*Itot/ring.h
        nb = ring.h
    elif fill_type == "32b":
        filling = np.zeros(ring.h)
        filling[0::13] = Itot/32
        nb = 32
    elif fill_type == "single":
        filling = np.zeros(ring.h)
        filling[0] = Itot
        nb = 1
    elif fill_type == "gaps":
        filling = np.ones(ring.h)
        gap_var = int(ring.h/n_gaps)
        for i in range(n_gaps):
            filling[gap_var*(i+1) - gap_len:gap_var*(i+1)] = 0
        nb = ring.h - n_gaps*gap_len
        filling *= Itot/nb
        
    bb.init_beam(filling, mp_per_bunch=mp, track_alive=False, mpi=mpi)
    
    current_step = (Itot_final-Itot)/(tot_turns-delay_after)/nb
    
    #%% RF setup
    MC.Vc = Vc
    if Det_calc_Xi:
        HC_det = Itot*HHC.Rs/HHC.Q*ring.f1/Vc*HHC.m**2/xi_start
        HHC.detune = HC_det
        HC_det_end = Itot*HHC.Rs/HHC.Q*ring.f1/Vc*HHC.m**2/xi_end
    
    delta = 0
    if HC_offset or HC_offset_Phasor or HC_offset_Phasor_power:
        delta += HHC.Vb(Itot)*np.cos(HHC.psi)
    if impedance_offset:
        delta += bb[0].charge * wf.Wlong.loss_factor(estimated_bunch_length)
    MC.theta = np.arccos((ring.U0 + delta)/MC.Vc)
    
    if Set_optimal_detune:
        MC.set_optimal_detune(Itot)
    MC.set_generator(Itot)
    
    if P_FB: MC.feedback.append(ProportionalLoop(ring, MC, PFB_gainA, PFB_gainP, PFB_delay))
    if PI_FB and not Direct_FB: MC.feedback.append(ProportionalIntegralLoop(ring, MC, fb_gain, fb_sample_num, fb_every, fb_delay, IIR_cutoff=fb_IIR_cutoff))
    if Tuner_FB: MC.feedback.append(TunerLoop(ring, MC, tuner_gain))
    if Direct_FB: MC.feedback.append(DirectFeedback(ring=ring, cav_res=MC, gain=fb_gain, sample_num=fb_sample_num, every=fb_every, delay=fb_delay, IIR_cutoff=fb_IIR_cutoff, DFB_gain=directFB_gain, DFB_phase_shift=directFB_phaseShift))
    
    MC.init_phasor(bb)
    HHC.init_phasor(bb)
    if Use_fixed_detuned_HC: HC_fixed.init_phasor(bb)
    
    det_step = np.abs(HC_det - HC_det_end)/(tot_turns-delay_before-delay_after)
    
    #%% Sweep
    if sweep_bool:
        from mbtrack2 import Sweep
        sweep = Sweep(ring, f0, f1, ring.T0*t1, level, plane, bunch_to_sweep)
    
    #%% Tracking start
    for i in range(int(tot_turns+1)):
        if (i % 1000 == 0):
            if mpi:
                if (bb.mpi.rank == 0): print(i)
            else:
                print(i)
    
        long.track(bb)
        rad.track(bb)
        if wake: wp.track(bb)
        if mpi: bb.mpi.share_distributions(bb)
    
        MC.track(bb)
        if i > delay_before: HHC.track(bb)
        if i > 50e3:
            for HOM in HOM_list: HOM.track(bb)
        if sweep_bool and (i > delay_before_sweep): sweep.track(bb)
        if Use_fixed_detuned_HC: HC_fixed.track(bb)
    
        if ContinuousTuning:
            if i < (tot_turns-delay_after): HHC.detune -= det_step
    
        if Injection and (i < (tot_turns-delay_after)):
            for bunch in bb: bunch.current += current_step
            if Injection_optimal_detune: MC.set_optimal_detune(bb.current)
            MC.set_generator(bb.current)
    
        if MCphase_update_during_tracking:
            delta = 0
            if HC_offset: delta += HHC.Vb(Itot)*np.cos(HHC.psi)
            elif HC_offset_Phasor:
                Vc_c = np.abs(HHC.cavity_phasor)
                theta_c = np.angle(HHC.cavity_phasor)
                delta += Vc_c*np.cos(theta_c)
            elif HC_offset_Phasor_power:
                Vc_c = np.abs(HHC.cavity_phasor)
                delta += Vc_c**2 / (2*HHC.Rs*bb.current)
            MC.theta = np.arccos((ring.U0 + delta)/MC.Vc)

        if MODE0_DAMPER: damper.track(bb, i)
        if BBB_LFB: lfb.track(bb)
        bbmon.track(bb)
        MCmon.track(bb, MC)  
    
        HCmon.track(bb, HHC)
        if spec:
            bunchspec.track(bb)
            beamspec.track(bb)
        if wake: wakemon.track(bb, wp)
        if Use_fixed_detuned_HC: HC_fixed_mon.track(bb, HC_fixed)

bbmon.close()
"""

@st.dialog("🔑 Enter SSH Password")
def ssh_password_dialog():
    st.write(f"Connecting to **{ssh_user}@{ssh_host}:{ssh_port}**")
    pwd = st.text_input("Password (or empty for key)", type="password")
    if st.button("🔌 Connect"):
        with st.spinner("Testing SSH connection..."):
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                if pwd: ssh.connect(ssh_host, port=ssh_port, username=ssh_user, password=pwd, timeout=5)
                else: ssh.connect(ssh_host, port=ssh_port, username=ssh_user, timeout=5)
                st.session_state.ssh_pwd_temp = pwd
                st.session_state.ssh_connected = True
                ssh.close()
                st.success("✅ SSH Connection successful!")
                st.rerun()
            except Exception as e:
                st.error(f"SSH Error: {e}")

st.subheader("🚀 Actions")
col_action1, col_action2, col_action3 = st.columns(3)
with col_action1:
    if st.session_state.ssh_connected:
        st.success("✅ SSH Connected")
        if st.button("❌ Disconnect SSH", use_container_width=True):
            st.session_state.ssh_connected = False
            st.session_state.ssh_pwd_temp = ""
            st.rerun()
    else:
        btn_ssh = st.button("🔌 Test SSH Connection", use_container_width=True)
        if btn_ssh:
            if not PARAMIKO_AVAILABLE:
                st.error("Cannot proceed: `paramiko` is not installed.")
            else:
                ssh_password_dialog()

with col_action2:
    btn_generate = st.button("📝 Generate Code", use_container_width=True)
with col_action3:
    btn_submit = st.button("🚀 Submit Job", type="primary", use_container_width=True, disabled=not st.session_state.ssh_connected)


if btn_generate:
    job_name_preview = f"mb2_job_preview_{int(time.time())}"
    st.session_state['generated_script_preview'] = build_script_content(job_name_preview)
    st.success("Code generated successfully. You can view it below.")
    
if 'generated_script_preview' in st.session_state:
    with st.expander("👁️ View Generated Code", expanded=False):
        st.code(st.session_state['generated_script_preview'], language='python')

if btn_submit:
    if not PARAMIKO_AVAILABLE:
        st.error("Cannot proceed: `paramiko` is not installed.")
        st.stop()
        
    job_name = f"mb2_job_{int(time.time())}"
    job_dir = f"{remote_base_dir}/{job_name}".replace('//', '/')
    script_content = build_script_content(job_name)
    
    # Execute SSH logic
    with st.spinner(f"Preparing remote directory: {job_dir}"):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            if st.session_state.ssh_pwd_temp:
                ssh.connect(ssh_host, port=ssh_port, username=ssh_user, password=st.session_state.ssh_pwd_temp, timeout=10)
            else:
                ssh.connect(ssh_host, port=ssh_port, username=ssh_user, timeout=10)
            
            # Create remote directory specifically for this job
            job_input_dir = f"{job_dir}/input".replace('//', '/')
            ssh.exec_command(f"mkdir -p {job_input_dir}")
            
            # Upload python script
            sftp = ssh.open_sftp()
            remote_py = f"{job_input_dir}/{job_name}.py"
            with sftp.file(remote_py, 'w') as f:
                f.write(script_content)
            
            # Execution script & Submitting
            if job_system == "SLURM (sbatch)":
                slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_dir}/%j.out
#SBATCH --error={job_dir}/%j.err
#SBATCH --nodes={slurm_nodes}
#SBATCH --ntasks-per-node={slurm_tasks}
#SBATCH --time={slurm_time}
#SBATCH --partition={slurm_partition}

cd {job_dir}
mpirun -np {slurm_nodes * slurm_tasks} python input/{job_name}.py
"""
                remote_sh = f"{job_dir}/{job_name}.sh"
                with sftp.file(remote_sh, 'w') as f:
                    f.write(slurm_script)
                
                stdin, stdout, stderr = ssh.exec_command(f"sbatch {remote_sh}")
                output = stdout.read().decode().strip()
                st.success(f"Job submitted to SLURM! Remote output: {output}")
                
                job_id = output.split()[-1] if 'Submitted batch job' in output else output
                st.session_state['remote_job_id'] = job_id
                st.session_state['remote_job_system'] = 'SLURM'
                # For slurm we must guess output file via id
                st.session_state['remote_job_log'] = f"{job_dir}/{job_id}.out" 
                
            else:
                # Direct execute
                run_cmd = f"cd {job_dir} && "
                if mpi_flag:
                    run_cmd += f"nohup mpirun python input/{job_name}.py > {job_name}.log 2>&1 & echo $!"
                else:
                    run_cmd += f"nohup python input/{job_name}.py > {job_name}.log 2>&1 & echo $!"
                    
                stdin, stdout, stderr = ssh.exec_command(run_cmd)
                pid = stdout.read().decode().strip()
                st.success(f"Job started! Remote PID: {pid}")
                
                st.session_state['remote_job_id'] = pid
                st.session_state['remote_job_system'] = 'Direct'
                st.session_state['remote_job_log'] = f"{job_dir}/{job_name}.log"
                
            st.session_state['ssh_host_last'] = ssh_host
            st.session_state['ssh_user_last'] = ssh_user
            
            st.session_state['remote_job_dir'] = job_dir
            
            sftp.close()
            ssh.close()
            
        except Exception as e:
            st.error(f"SSH Error: {str(e)}")


st.markdown("---")

@st.fragment(run_every=2)
def live_log_reader(log_file, lines):
    if not log_file:
         return
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        host = st.session_state.get('ssh_host_last', 'localhost')
        port = st.session_state.get('ssh_port', 22)
        user = st.session_state.get('ssh_user_last', '')
        pwd = st.session_state.get('ssh_pwd_temp', '')
        if pwd: ssh.connect(host, port=port, username=user, password=pwd, timeout=5)
        else: ssh.connect(host, port=port, username=user, timeout=5)
        
        stdin, stdout, stderr = ssh.exec_command(f"tail -n {lines} {log_file}")
        content = stdout.read().decode()
        err = stderr.read().decode()
        if content:
             st.code(content, language='text')
        if err:
             st.error(err)
        ssh.close()
    except Exception as e:
        st.error(f"Waiting for log file or connection error: {e}")

# Tabs for Monitoring and Result Fetching
mon_tab, fetch_tab = st.tabs(["📡 Job Monitoring & Outputs", "📥 Fetch HDF5 Results"])

with mon_tab:
    st.markdown("### Read Specific Log File")
    st.caption("Read content from a specific log file (e.g., .out, .log, .err)")
    log_file_input = st.text_input("Log File Path (Full Remote Path)", st.session_state.get('remote_job_log', ''))
    lines_to_read = st.number_input("Number of lines to read (tail)", min_value=10, max_value=2000, value=50, step=10)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        if st.button("🔄 Read Log / Check Status"):
            with st.spinner("Checking..."):
                try:
                    ssh = paramiko.SSHClient()
                    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    ssh.connect(st.session_state.get('ssh_host_last'), port=st.session_state.get('ssh_port', 22), username=st.session_state.get('ssh_user_last'), password=st.session_state.get('ssh_pwd_temp', ''), timeout=10)
                        
                    if st.session_state.get('remote_job_system') == 'SLURM' and st.session_state.get('remote_job_id'):
                        stdin, stdout, stderr = ssh.exec_command(f"squeue -j {st.session_state['remote_job_id']}")
                        st.text("SLURM Queue Status:")
                        st.code(stdout.read().decode())
                    elif st.session_state.get('remote_job_system') == 'Direct' and st.session_state.get('remote_job_id'):
                        stdin, stdout, stderr = ssh.exec_command(f"ps -p {st.session_state['remote_job_id']}")
                        ps_out = stdout.read().decode()
                        if str(st.session_state['remote_job_id']) in ps_out:
                            st.success("PID Process is currently RUNNING")
                        else:
                            st.warning("PID Process has FINISHED or STOPPED")
                            
                    # Read log
                    if log_file_input:
                        stdin, stdout, stderr = ssh.exec_command(f"tail -n {lines_to_read} {log_file_input}")
                        out_text = stdout.read().decode()
                        err_text = stderr.read().decode()
                        if out_text:
                            st.text(f"Latest {lines_to_read} lines from {log_file_input}:")
                            st.code(out_text)
                        if err_text:
                            st.error(f"Error accessing log file: {err_text}")
    
                    ssh.close()
                except Exception as e:
                    st.error(f"Error checking status: {str(e)}")
                    
    with col_stat2:
        st.write("When live monitoring is enabled, the latest log output will be fetched automatically every few seconds.")
        live_mon_toggle = st.toggle("🚀 Enable Live Monitor", value=False)
        
    if live_mon_toggle:
        live_log_reader(log_file_input, lines_to_read)

    st.markdown("---")
    st.markdown("### 🔍 Auto-Find & Read Debug Logs (.o / .out)")
    st.caption("Scan a directory (e.g., `.../script name-job id/`) recursively for `*-<job_id>.o` or `.out` debug files.")
    target_debug_dir = st.text_input("Remote Job Directory (for log files)", st.session_state.get('remote_job_dir', '~/mbtrack2_jobs'))
    if st.button("🔎 Scan for Log Files"):
        st.session_state['scanned_debug_dir'] = target_debug_dir

    if 'scanned_debug_dir' in st.session_state:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Connect with or without password
            if st.session_state.get('ssh_pwd_temp'):
                ssh.connect(st.session_state.get('ssh_host_last'), port=st.session_state.get('ssh_port', 22), username=st.session_state.get('ssh_user_last'), password=st.session_state.get('ssh_pwd_temp', ''), timeout=10)
            else:
                ssh.connect(st.session_state.get('ssh_host_last'), port=st.session_state.get('ssh_port', 22), username=st.session_state.get('ssh_user_last'), timeout=10)

            # Find .o, .out, .err files recursively (maxdepth 3) in case they are inside job_id folders
            stdin, stdout, stderr = ssh.exec_command(f"find {st.session_state['scanned_debug_dir']} -maxdepth 3 -type f \\( -name '*.o*' -o -name '*.out' -o -name '*.err' -o -name '*.log' \\) | sort")
            o_files = stdout.read().decode().strip().split('\n')
            o_files = [f for f in o_files if f and "No such file" not in f]
            
            if o_files:
                st.success(f"Found {len(o_files)} debug log files.")
                selected_o_file = st.selectbox("Select a log file to view", o_files)
                tail_lines_o = st.number_input("Number of lines to read", min_value=10, max_value=5000, value=100, step=50, key="o_tail")
                
                live_mon_o_toggle = st.toggle("🚀 Enable Live Monitor for this file", value=False, key="o_live_toggle")

                if st.button("📖 Read Selected Log File") and not live_mon_o_toggle:
                    with st.spinner("Fetching contents..."):
                        stdin, stdout, stderr = ssh.exec_command(f"tail -n {tail_lines_o} {selected_o_file}")
                        o_content = stdout.read().decode()
                        o_err = stderr.read().decode()
                        if o_content:
                            st.code(o_content)
                        if o_err:
                            st.error(f"Error: {o_err}")
                            
                if live_mon_o_toggle:
                    live_log_reader(selected_o_file, tail_lines_o)
            else:
                st.warning("No log files found in the specified directory.")
            ssh.close()
        except Exception as e:
            st.error(f"Error scanning for log files: {str(e)}")




with fetch_tab:
    st.markdown("### List & Fetch HDF5 Simulation Outputs")
    fetch_target_dir = st.text_input("Remote Results Path", st.session_state.get('remote_job_dir', '~/mbtrack2_jobs'))
    
    colA, colB = st.columns([1, 4])
    with colA:
        if st.button("🔎 Scan Directory"):
            st.session_state['scanned_dir'] = fetch_target_dir
            
    if 'scanned_dir' in st.session_state:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(st.session_state.get('ssh_host_last'), username=st.session_state.get('ssh_user_last'), password=st.session_state.get('ssh_pwd_temp', ''), timeout=10)
            
            # Find hdf5 / h5 files recursively (maxdepth 3) in case they are inside job_id folders
            stdin, stdout, stderr = ssh.exec_command(f"find {st.session_state['scanned_dir']} -maxdepth 3 -type f \\( -name '*.h5' -o -name '*.hdf5' \\)")
            files_list = stdout.read().decode().strip().split('\n')
            files_list = [f for f in files_list if f and "No such file" not in f]
            
            if files_list:
                st.success(f"Found {len(files_list)} dataset files")
                for fn in files_list:
                    # just to render nicely, user might just want the names
                    st.code(fn)
                
                st.info("💡 Next step: Provide the exact file path from the list above to fetch the dataset locally.")
                target_file_to_dl = st.text_input("Remote path to download:")
                
                if st.button("💾 Fetch File to Local Machine"):
                    with st.spinner(f"Downloading {target_file_to_dl} ..."):
                        sftp = ssh.open_sftp()
                        local_downloads_path = os.path.join(Path.home(), "Downloads")
                        os.makedirs(local_downloads_path, exist_ok=True)
                        file_basename = os.path.basename(target_file_to_dl)
                        local_path = os.path.join(local_downloads_path, file_basename)
                        
                        sftp.get(target_file_to_dl, local_path)
                        sftp.close()
                        st.success(f"✅ Successfully downloaded to {local_path} !")
            else:
                st.warning("No .h5 or .hdf5 files found in this directory. Perhaps the job hasn't output them yet?")
                
            ssh.close()
        except Exception as e:
            st.error(f"Error accessing directory: {str(e)}")
