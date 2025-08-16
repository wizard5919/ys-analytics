import streamlit as st
import requests
from PIL import Image
import io
import os

# ==============================
# SECURITY & PERFORMANCE IMPROVEMENTS
# ==============================
# Use environment variables for sensitive data if needed
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# Cache logo to reduce network calls
@st.cache_data(ttl=86400, show_spinner=False)
def load_logo(url):
    try:
        response = requests.get(url, timeout=5)
        return Image.open(io.BytesIO(response.content))
    except:
        # Fallback to a solid color image
        return Image.new('RGB', (150, 150), color='#0A1F44')

logo_img = load_logo(LOGO_URL)

# ==============================
# PAGE CONFIG & SEO OPTIMIZATION
# ==============================
st.set_page_config(
    page_title="YS Analytics - Financial Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://ysanalytics.me/support',
        'Report a bug': 'https://ysanalytics.me/bug',
        'About': "### Data-Driven Market Intelligence Platform"
    }
)

# Add SEO meta tags in a hidden div
st.markdown(f'''
<div style="display: none;">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Quantitative research, predictive modeling, and interactive dashboards for financial markets">
    <meta name="keywords" content="financial analytics, market intelligence, investment research, quantitative finance">
    <meta name="author" content="YS Analytics">
    <link rel="canonical" href="https://app.ysanalytics.me">
    <!-- Open Graph Tags for Social Sharing -->
    <meta property="og:title" content="YS Analytics - Financial Market Intelligence">
    <meta property="og:description" content="Quantitative research, predictive modeling, and interactive dashboards for financial markets">
    <meta property="og:image" content="{LOGO_URL}">
    <meta property="og:url" content="https://app.ysanalytics.me">
    <meta property="og:type" content="website">
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:creator" content="@ys_analytics">
</div>
''', unsafe_allow_html=True)

# ==================
# CUSTOM CSS STYLING FOR PROFESSIONAL LOOK
# ==================
st.markdown("""
<style>
:root {
    --primary: #00C2FF;
    --dark-bg: #0A1F44;
    --darker-bg: #152852;
    --text-light: #FFFFFF;
    --accent: #0077B6;
}
/* Overall app styling */
body {
    background-color: var(--dark-bg);
    color: var(--text-light);
}
.stApp {
    background-color: var(--dark-bg);
}
/* Header styling */
.stTitle {
    color: var(--primary);
}
/* Unified card styling */
.card {
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    background: linear-gradient(135deg, var(--dark-bg) 0%, var(--darker-bg) 100%);
    border: 1px solid var(--primary);
    margin-bottom: 30px;
    color: var(--text-light);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,194,255,0.25);
}
.card h2, .card h3 {
    color: var(--primary);
    margin-top: 0;
    border-bottom: 1px solid var(--accent);
    padding-bottom: 10px;
}
.card p, .card ul {
    font-size: 1.05em;
    line-height: 1.6;
}
/* Tech tags */
.tech-tag {
    display: inline-block;
    background-color: rgba(0, 194, 255, 0.15);
    color: var(--primary);
    border-radius: 6px;
    padding: 4px 12px;
    margin: 4px 4px 4px 0;
    font-size: 0.85em;
    font-weight: 500;
    border: 1px solid rgba(0, 194, 255, 0.3);
    transition: all 0.2s ease;
}
.tech-tag:hover {
    background-color: rgba(0, 194, 255, 0.3);
    transform: scale(1.05);
}
/* Button enhancements */
.stButton > button {
    background: linear-gradient(to right, var(--primary), var(--accent));
    color: var(--text-light) !important;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 16px rgba(0,194,255,0.3);
}
/* Footer styling */
.footer {
    margin-top: 40px;
    padding: 20px 0;
    border-top: 1px solid var(--accent);
    font-size: 0.95em;
}
.footer a {
    color: var(--primary);
    text-decoration: none;
    transition: color 0.2s;
}
.footer a:hover {
    color: #FFFFFF;
}
/* Responsive adjustments */
@media (max-width: 768px) {
    .card {
        padding: 15px;
    }
    .stButton > button {
        padding: 8px 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# ================
# HEADER SECTION
# ================
col1, col2 = st.columns([1, 3])
with col1:
    st.image(
        logo_img,
        width=150,
        use_container_width=True
    )
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# ===================
# MISSION STATEMENT
# ===================
st.markdown("""
<div class="card">
    <h2>Precision Analytics for Financial Markets</h2>
    <p>We transform complex market data into actionable intelligence through:</p>
    <ul>
        <li><strong>Quantitative Research</strong> • Algorithmic market analysis</li>
        <li><strong>Predictive Modeling</strong> • Machine learning-driven forecasts</li>
        <li><strong>Strategic Visualization</strong> • Interactive financial dashboards</li>
        <li><strong>Risk Analytics</strong> • Options pricing and volatility insights</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# =================
# FEATURED PROJECTS
# =================
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

col1, col2, col3 = st.columns(3)

# Project 1 - Options Analytics Suite
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Options Analytics Suite")
    st.markdown("Real-time Greeks calculation and volatility surface visualization")
    st.markdown("""
    <div style="margin: 15px 0;">
        <span class="tech-tag">Python</span>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">QuantLib</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Project", key="p1", use_container_width=True):
        st.session_state.navigate_to = "options"
        st.switch_page("pages/2_Projects.py")
    st.markdown("</div>", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Market Sector Classifier")
    st.markdown("ML-driven sector analysis using price movement patterns")
    st.markdown("""
    <div style="margin: 15px 0;">
        <span class="tech-tag">Scikit-learn</span>
        <span class="tech-tag">TA-Lib</span>
        <span class="tech-tag">Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Project", key="p2", use_container_width=True):
        st.session_state.navigate_to = "sector"
        st.switch_page("pages/2_Projects.py")
    st.markdown("</div>", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Macroeconomic Dashboard")
    st.markdown("Global economic indicators with forecasting capabilities")
    st.markdown("""
    <div style="margin: 15px 0;">
        <span class="tech-tag">FRED API</span>
        <span class="tech-tag">Prophet</span>
        <span class="tech-tag">Altair</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Project", key="p3", use_container_width=True):
        st.session_state.navigate_to = "macro"
        st.switch_page("pages/2_Projects.py")
    st.markdown("</div>", unsafe_allow_html=True)

# ================
# CALL TO ACTION
# ================
st.markdown("---")
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# ================
# FOOTER SECTION
# ================
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**© 2025 YS Analytics**")
with footer_cols[1]:
    st.markdown("[GitHub](https://github.com/wizard5919) • [LinkedIn](https://www.linkedin.com/in/youssef-sbai-ba-59b0172b0)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
st.markdown('</div>', unsafe_allow_html=True)

# ================
# PERFORMANCE OPTIMIZATION
# ================
st.markdown("""
<link rel="preconnect" href="https://raw.githubusercontent.com">
<link rel="preconnect" href="https://fonts.googleapis.com">
""", unsafe_allow_html=True)
