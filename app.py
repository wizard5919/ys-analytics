import streamlit as st
from utils import load_global_css, render_sidebar_navigation, render_project_search

# Analytics init
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

# Add additional styling for better spacing and alignment
st.markdown("""
<style>
    /* Improved card styling */
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background: #0E1117;
        border: 1px solid #00C2FF;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    /* Better tag styling */
    .tech-tag {
        background-color: #00C2FF;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8em;
        display: inline-block;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(to right, #00C2FF, #0077FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: bold !important;
        margin-top: auto;
    }
    
    /* Better spacing */
    .section-spacer {
        margin-top: 40px;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e3a5f;
        font-size: 0.9em;
    }
    
    /* Project title styling */
    .project-title {
        min-height: 60px;
        margin-bottom: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Render sidebar
render_sidebar_navigation()

# Header with consistent logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"
col1, col2 = st.columns([1, 4])  # Adjusted ratio for better spacing
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# Mission statement card
st.markdown("""
<div class="card">
    <h2 style="color: #00C2FF; border-bottom: 2px solid #00C2FF; padding-bottom: 10px; margin-top: 0;">Precision Analytics for Financial Markets</h2>
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
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

# Add search component
st.markdown("**AI-Powered Search: Find Projects**")
st.caption("e.g., 'options' or 'ML'")

projects = [
    {"title": "📈 Options Analytics Suite", "desc": "Real-time Greeks calculation and volatility surface visualization", "tags": ["Python", "Streamlit", "QuantLib"], "preview_url": "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/options-preview.png", "link": "pages/2_Projects.py"},
    {"title": "🤖 Market Sector Classifier", "desc": "ML-driven sector analysis using price movement patterns", "tags": ["Scikit-learn", "TA-Lib", "Plotly"], "preview_url": "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/sector-preview.png", "link": "pages/2_Projects.py"},
    {"title": "🌍 Macroeconomic Dashboard", "desc": "Global economic indicators with forecasting capabilities", "tags": ["FRED API", "Prophet", "Altair"], "preview_url": "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/macro-preview.png", "link": "pages/2_Projects.py"}
]

filtered_projects = render_project_search(projects)

cols = st.columns(3)
for i, proj in enumerate(filtered_projects):
    with cols[i % 3]:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Project title with consistent height
            st.markdown(f'<div class="project-title"><h3>{proj["title"]}</h3></div>', unsafe_allow_html=True)
            
            # Project image
            st.image(proj["preview_url"], use_container_width=True)
            
            # Project description
            st.markdown(proj["desc"])
            
            # Technology tags
            st.markdown('<div style="margin: 10px 0;">', unsafe_allow_html=True)
            st.markdown(" ".join([f'<span class="tech-tag">{tag}</span>' for tag in proj["tags"]]), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # View project button
            if st.button("🔍 View Project", key=f"p{i}", use_container_width=True):
                st.switch_page(proj["link"])
            st.markdown("</div>", unsafe_allow_html=True)

# CTA section
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.markdown("---")
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# Footer
st.markdown('<div class="footer"></div>', unsafe_allow_html=True)
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**© 2025 YS Analytics**")
with footer_cols[1]:
    st.markdown("[<i class='fab fa-github'></i> GitHub](https://github.com/wizard5919) • [<i class='fab fa-linkedin'></i> LinkedIn](https://linkedin.com)", unsafe_allow_html=True)
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
