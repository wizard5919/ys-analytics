# app.py (Corrected to match exact desired output: No images, plain tags, no cards for projects, no icons in footer)
import streamlit as st
from utils import load_global_css, render_sidebar_navigation, render_project_search

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

# Render sidebar navigation
render_sidebar_navigation()

# Title and subtitle
st.title("YS Analytics")
st.markdown("Data-Driven Market Intelligence")

# Mission section with card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("Precision Analytics for Financial Markets")
st.markdown("We transform complex market data into actionable intelligence through:")
st.markdown("* Quantitative Research • Algorithmic market analysis")
st.markdown("* Predictive Modeling • Machine learning-driven forecasts")
st.markdown("* Strategic Visualization • Interactive financial dashboards")
st.markdown("* Risk Analytics • Options pricing and volatility insights")
st.markdown("</div>", unsafe_allow_html=True)

# Featured Projects
st.header("Featured Analytics Projects")
st.markdown("Select case studies demonstrating our financial analytics capabilities")

# Search bar (only once)
projects = [
    {"title": "📈 Options Analytics Suite", "desc": "Real-time Greeks calculation and volatility surface visualization", "tags": "Python Streamlit QuantLib", "link": "pages/2_Projects.py"},
    {"title": "🤖 Market Sector Classifier", "desc": "ML-driven sector analysis using price movement patterns", "tags": "Scikit-learn TA-Lib Plotly", "link": "pages/2_Projects.py"},
    {"title": "🌍 Macroeconomic Dashboard", "desc": "Global economic indicators with forecasting capabilities", "tags": "FRED API Prophet Altair", "link": "pages/2_Projects.py"}
]

filtered_projects = render_project_search(projects)

cols = st.columns(3)
for i, proj in enumerate(filtered_projects):
    with cols[i]:
        st.subheader(proj["title"])
        st.markdown(proj["desc"])
        st.markdown(proj["tags"])
        if st.button("View Project", key=f"p{i}", use_container_width=True):
            st.switch_page(proj["link"])

# CTA
st.markdown("---")
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# Footer
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("© 2025 YS Analytics")
with footer_cols[1]:
    st.markdown("GitHub LinkedIn")
with footer_cols[2]:
    st.markdown("Data Sources: FRED • Yahoo Finance • OANDA")
