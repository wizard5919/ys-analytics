# app.py (Simplified: Only navigation sidebar, no other content)
import streamlit as st
from utils import load_global_css, render_sidebar_navigation

# Analytics init (unchanged)
if "analytics_initialized" not in st.session_state:
    st.session_state.analytics_initialized = True
    st.markdown("""<!-- Google Tag Manager and Analytics code -->""", unsafe_allow_html=True)

st.set_page_config(
    page_title="YS Analytics - Financial Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load global CSS
load_global_css()

# Render sidebar navigation only
render_sidebar_navigation()

# No other content on root page - navigation leads to pages
