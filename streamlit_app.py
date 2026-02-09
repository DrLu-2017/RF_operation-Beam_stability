"""
ALBuMS - Streamlit Web Interface
Main application entry point.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add local mbtrack2 if available (workaround for installation issues)
mbtrack2_local = project_root / "mbtrack2-stable"
if mbtrack2_local.exists():
    sys.path.insert(0, str(mbtrack2_local))

# Add local pycolleff if available
pycolleff_local = project_root / "collective_effects-master" / "pycolleff"
if pycolleff_local.exists():
    sys.path.insert(0, str(pycolleff_local))

# Page configuration
st.set_page_config(
    page_title="DRFB - Double RF & Beam Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: #a0aec0;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 1px;
    }

    .theme-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
    }

    .theme-card:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.1);
        border-color: #4facfe;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }

    .theme-card h2 {
        font-family: 'Orbitron', sans-serif;
        color: #4facfe;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
        line-height: 1.2;
    }

    .theme-card p {
        color: #cbd5e0;
        line-height: 1.6;
    }

    .feature-tag {
        display: inline-block;
        background: rgba(79, 172, 254, 0.2);
        color: #4facfe;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.6);
        transform: scale(1.02);
    }

    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown('<h1 class="main-header">DRFB</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Double RF Beam Analysis & Stability Dashboard</p>', unsafe_allow_html=True)

# Main Themes
col1, col2 = st.columns(2)

with col1:
    st.image("static/double_rf_system.png", width='stretch')
    st.markdown("""
    <div class="theme-card">
        <h2>Double RF Systems</h2>
        <p>Advanced analysis of main and harmonic cavity interactions. Optimize RF parameters, 
        power distribution, and cavity detunings to ensure ideal beam characteristics.</p>
        <div style="margin-top: 1rem;">
            <span class="feature-tag">Cavity Optimization</span>
            <span class="feature-tag">Phasing Control</span>
            <span class="feature-tag">Power Balance</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image("static/stability_research.png", width='stretch')
    st.markdown("""
    <div class="theme-card">
        <h2>Stability Research</h2>
        <p>Leveraging ALBuMS algorithms to investigate longitudinal multibunch instabilities. 
        Track Robinson modes, calculate growth rates, and find stable operating regions.</p>
        <div style="margin-top: 1rem;">
            <span class="feature-tag">Mode Tracking</span>
            <span class="feature-tag">Growth Rates</span>
            <span class="feature-tag">Stability Maps</span>
            <span class="feature-tag">Landau Damping</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Detailed Features or Quick Links
st.markdown("---")
st.markdown("### 🔍 Explore Analytical Tools")

feat1, feat2, feat3 = st.columns(3)

with feat1:
    st.info("**Parameter Scans**")
    st.write("Generate high-resolution stability maps (Psi vs Current) to identify safe storage ring operation limits.")
    if st.button("Launch Scans"):
        st.switch_page("pages/1_📊_Parameter_Scans.py")

with feat2:
    st.info("**R-Factor Optimization**")
    st.write("Maximize Touschek lifetime by finding optimal harmonic cavity detuning and voltage settings.")
    if st.button("Start Optimization"):
        st.switch_page("pages/2_🎯_Optimization.py")

with feat3:
    st.info("**Mode Analysis**")
    st.write("Deep dive into Robinson modes and multibunch coupling with interactive spectral visualization.")
    if st.button("Analyze Modes"):
        st.switch_page("pages/3_🔬_Mode_Analysis.py")

# Footer/About
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
with st.expander("Scientific Foundation"):
    st.markdown("""
    ### ALBuMS (Algorithms for Longitudinal Multibunch Beam Stability)
    
    This platform integrates state-of-the-art semi-analytical algorithms for studying longitudinal beam instabilities 
    in double RF systems. Developed at **Synchrotron SOLEIL**, it provides researchers with robust tools for 
    4th generation light source design and operation.
    
    **Core Backend:** 
    - `ALBuMS` Physics Engine
    - `mbtrack2` Many-particle tracking library
    - `pycolleff` Collective effects calculations
    """)

# Sidebar
with st.sidebar:
    st.markdown("<div style='text-align: center;'><h1 style='color: #4facfe;'>DRFB</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Navigation")
    st.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.page_link("pages/0_🔧_Double_RF_System.py", label="Double RF System", icon="🔧")
    st.page_link("pages/1_📊_Parameter_Scans.py", label="Parameter Scans", icon="📊")
    st.page_link("pages/2_🎯_Optimization.py", label="R-Factor Optimization", icon="🎯")
    st.page_link("pages/3_🔬_Mode_Analysis.py", label="Robinson Mode Analysis", icon="🔬")
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("Backend: Connected")
    st.info("Version: 1.0.0 (DRFB Edition)")

