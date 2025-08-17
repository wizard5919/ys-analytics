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

# Render sidebar
render_sidebar_navigation()

# Header with consistent logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"  # Ensure high-res
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# Mission statement card
st.markdown("""
<div class="card">
    <h2 style="color: #00C2FF; border-bottom: 2px solid #00C2FF; padding-bottom: 10px;">Precision Analytics for Financial Markets</h2>
    <p>We transform complex market data into actionable intelligence through:</p>
    <ul>
        <li><strong>Quantitative Research</strong> • Algorithmic market analysis</li>
        <li><strong>Predictive Modeling</strong> • Machine learning-driven forecasts</li>
        <li><strong>Strategic Visualization</strong> • Interactive financial dashboards</li>
        <li><strong>Risk Analytics</strong> • Options pricing and volatility insights</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Featured Projects with search and previews
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

projects = [
    {"title": "📈 Options Analytics Suite", "desc": "Real-time Greeks calculation and volatility surface visualization", "tags": ["Python", "Streamlit", "QuantLib"], "preview_url": "https://via.placeholder.com/300x200?text=Options+Chart", "link": "pages/2_Projects.py"},
    {"title": "🤖 Market Sector Classifier", "desc": "ML-driven sector analysis using price movement patterns", "tags": ["Scikit-learn", "TA-Lib", "Plotly"], "preview_url": "https://via.placeholder.com/300x200?text=ML+Model", "link": "pages/2_Projects.py"},
    {"title": "🌍 Macroeconomic Dashboard", "desc": "Global economic indicators with forecasting capabilities", "tags": ["FRED API", "Prophet", "Altair"], "preview_url": "https://via.placeholder.com/300x200?text=Macro+Dashboard", "link": "pages/2_Projects.py"}
]

filtered_projects = render_project_search(projects)

cols = st.columns(3)
for i, proj in enumerate(filtered_projects):
    with cols[i % 3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(proj["title"])
        st.image(proj["preview_url"], use_column_width=True)  # Added preview
        st.markdown(proj["desc"])
        st.markdown(" ".join([f'<span class="tech-tag">{tag}</span>' for tag in proj["tags"]]), unsafe_allow_html=True)
        if st.button("🔍 View Project", key=f"p{i}", use_container_width=True):
            st.switch_page(proj["link"])
        st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("**© 2025 YS Analytics**")
with footer_cols[1]:
    st.markdown("[<i class='fab fa-github'></i> GitHub](https://github.com/wizard5919) • [<i class='fab fa-linkedin'></i> LinkedIn](https://linkedin.com)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
