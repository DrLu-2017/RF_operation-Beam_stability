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
    page_title="ALBuMS - Beam Stability Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">⚛️ ALBuMS</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Algorithms for Longitudinal Multibunch Beam Stability</p>', unsafe_allow_html=True)

# Introduction
st.markdown("""
## Welcome to ALBuMS Web Interface

**ALBuMS** is an open-source Python package for analyzing longitudinal beam instabilities in double RF systems 
used in synchrotron light sources and storage rings.

### Key Features

- 🔬 **Parameter Scanning**: Explore stability regions across parameter spaces
- 🎯 **Optimization**: Maximize Touschek lifetime through R-factor optimization  
- 📊 **Mode Analysis**: Track Robinson modes and identify instabilities
- 📈 **Interactive Visualization**: Explore results with dynamic plots

### Quick Start

1. **Navigate** to a page using the sidebar
2. **Select** a preset configuration or enter custom parameters
3. **Run** analysis and explore interactive results
4. **Export** data and plots for your research

---
""")

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>📊 Parameter Scans</h3>
        <p>Scan stability regions across psi vs current, psi vs R/Q, or psi vs QL parameter spaces.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Optimization</h3>
        <p>Find optimal harmonic cavity settings to maximize the Touschek lifetime R-factor.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-box">
        <h3>🔬 Mode Analysis</h3>
        <p>Analyze Robinson modes, growth rates, and mode coupling phenomena.</p>
    </div>
    """, unsafe_allow_html=True)

# Getting started section
st.markdown("---")
st.markdown("### 🚀 Getting Started")

st.info("""
**New to ALBuMS?** Start with the **Parameter Scans** page and select the "SOLEIL II" preset 
to see a quick example of stability analysis.
""")

# Documentation and references
with st.expander("📚 Documentation & References"):
    st.markdown("""
    #### Documentation
    - [ALBuMS Documentation](https://albums.readthedocs.io/)
    - [GitHub Repository](https://github.com/synchrotron-soleil/albums)
    
    #### Citation
    If you use ALBuMS in your research, please cite:
    
    > Gamelin, A., Gubaidulin, V., Alves, M. B., & Olsson, T. (2024). 
    > Semi-analytical algorithms to study longitudinal beam instabilities in double rf systems. 
    > arXiv preprint arXiv:2412.06539.
    
    #### About
    ALBuMS is developed at Synchrotron SOLEIL for the analysis of longitudinal multibunch beam stability 
    in 4th-generation light sources.
    """)

# Sidebar information
with st.sidebar:
    st.markdown("## Navigation")
    st.info("""
    Use the pages above to access different analysis tools:
    
    - **Parameter Scans**: 2D stability maps
    - **Optimization**: R-factor maximization
    - **Mode Analysis**: Robinson mode tracking
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Version**: 0.1.0  
    **Framework**: Streamlit  
    **Backend**: ALBuMS + mbtrack2
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ for accelerator physics")
