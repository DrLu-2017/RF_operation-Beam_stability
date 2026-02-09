"""
UI utilities for the ALBuMS Streamlit application.
Handles localized number formatting and display settings.
"""

import streamlit as st
import numpy as np

def fmt(value, precision=2):
    """
    Format number as string, replacing dot with comma if user preference is set.
    """
    # Simply use a session state toggle to store preference
    sep = st.session_state.get("decimal_sep", ".")
    if isinstance(value, (int, float, np.float64, np.float32)):
        # Handle cases where value might be NaN or Inf
        if not np.isfinite(value):
            return str(value)
            
        s = f"{value:.{precision}f}"
        if sep == ",":
            return s.replace(".", ",")
        return s
    return str(value)

def render_display_settings():
    """
    Render decimal separator toggle in the sidebar and update session state.
    """
    st.sidebar.divider()
    st.sidebar.write("🌍 **Display Settings**")
    
    # Initialize session state if not present
    if "use_comma_decimal" not in st.session_state:
        # Default to False
        st.session_state.use_comma_decimal = False
    
    use_comma = st.sidebar.checkbox(
        "Use comma as decimal (e.g. 5,50)", 
        value=st.session_state.use_comma_decimal, 
        key="use_comma_decimal_toggle"
    )
    
    # Update preferences
    st.session_state.use_comma_decimal = use_comma
    st.session_state.decimal_sep = "," if use_comma else "."
