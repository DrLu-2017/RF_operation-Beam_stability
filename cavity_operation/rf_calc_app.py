import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Application Page Configuration
st.set_page_config(page_title="2.75 GeV RF Calculation Platform", layout="wide")

def calculate_rf_parameters(i_beam_ma, v_main_kv, nh):
    """
    Calculates storage ring RF parameters based on beam loading and harmonic conditions.
    
    Args:
        i_beam_ma (float): Beam current in mA.
        v_main_kv (float): Total main cavity voltage in kV.
        nh (int): Harmonic multiplier (e.g., 4).
    """
    # Parameters from the storage ring specification 
    u0_base_kv = 803.0  # Base radiation loss per turn
    i_beam_a = i_beam_ma / 1000.0
    
    # 1. Energy Loss with induced harmonic effect 
    # Induced voltage/loss varies with beam current in storage ring models
    u_total_kv = u0_base_kv + (0.15 * i_beam_ma if i_beam_ma > 0 else 0) 
    
    # 2. Synchronous Phase calculation 
    # Fundamental condition: V_total * sin(phi_s) = U_total
    try:
        phi_s_rad = np.arcsin(u_total_kv / v_main_kv)
    except ValueError:
        phi_s_rad = 0.0
        
    phi_s_deg = np.degrees(phi_s_rad)
    
    # 3. Flat Potential condition for harmonic voltage 
    # Optimal harmonic voltage vh_opt to maximize bunch lengthening
    term1 = (v_main_kv / nh)**2
    term2 = (u0_base_kv**2) / (nh**2 - 1)
    vh_opt_kv = np.sqrt(max(0, term1 - term2))
    
    # 4. Beam Power calculation 
    p_beam_kw = i_beam_a * u_total_kv
    
    return {
        "phi_s_deg": phi_s_deg,
        "vh_opt_kv": vh_opt_kv,
        "p_beam_kw": p_beam_kw,
        "u0_kv": u0_base_kv,
        "ut_kv": u_total_kv
    }

# --- User Interface Layout ---
st.title("2.75 GeV Storage Ring RF System Analysis Tool")
st.markdown("""
This platform simulates the RF system parameters of a 2.75 GeV storage ring. 
It implements the physical modeling of beam loading and harmonic cavity optimization.
""")

# Sidebar inputs for real-time interaction [5]
st.sidebar.header("System Settings")
i_beam = st.sidebar.slider("Beam Current $I_{beam}$ (mA)", 0.0, 500.0, 250.0, step=1.0)
v_main = st.sidebar.number_input("Main Cavity Voltage $VF_{total}$ (kV)", value=1700.0, step=10.0)
nh_val = st.sidebar.selectbox("Harmonic Multiplier $n_h$", options=[1, 2, 3], index=1)

# Perform Physics Calculations
results = calculate_rf_parameters(i_beam, v_main, nh_val)

# Top-level Dashboard Metrics
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Synchronous Phase $\phi_s$", f"{results['phi_s_deg']:.2f} °")
m_col2.metric("Optimal Harmonic Voltage $V_{h,opt}$", f"{results['vh_opt_kv']:.2f} kV")
m_col3.metric("Beam Power $P_{beam}$", f"{results['p_beam_kw']:.2f} kW")

# Interactive Waveform Visualization
st.subheader("RF Potential and Voltage Distribution")
st.write("Visualizing the total RF potential well under the 'Flat Potential' condition.[3]")

phi = np.linspace(-np.pi, np.pi, 1000)
# Sum of fundamental and harmonic voltages
v_fundamental = v_main * np.sin(phi + np.radians(results['phi_s_deg']))
v_harmonic = results['vh_opt_kv'] * np.sin(nh_val * phi)
v_total = v_fundamental + v_harmonic

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.degrees(phi), y=v_total, name="Total Voltage", line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=np.degrees(phi), y=v_fundamental, name="Main Voltage", line=dict(dash='dash', color='gray')))
fig.update_layout(
    xaxis_title="Phase (degrees)",
    yaxis_title="Voltage (kV)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white",
    height=500
)
st.plotly_chart(fig, width='stretch')

# Static Parameter Reference Table
st.subheader("Machine Parameter Reference")
param_table = {
    "Parameter": ["Nominal Energy", "Fundamental Frequency", "Radiation Loss per Turn", "Ring Harmonic Number", "Maximum Target Current"],
    "Value": ["2.75 GeV", "352.2 MHz", f"{results['u0_kv']} kV", "416", "500 mA"],
    "Description": ["Nominal operating energy", "Fundamental frequency ", "Radiation loss per turn ", "Ring harmonic number", "Maximum target current"]
}
st.table(param_table)

st.info("Note: The model updates instantly as you adjust sliders, allowing real-time evaluation of beam loading effects.[6]")