import streamlit as st
import requests
from PIL import Image
import io
import os

# ==============================
# SECURITY & PERFORMANCE IMPROVEMENTS
# ==============================
# Use environment variables for sensitive data
GTM_ID = os.getenv('GTM_ID', 'GTM-XXXXXX')
GA_ID = os.getenv('GA_ID', 'G-XXXXXX')
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
# GOOGLE ANALYTICS INITIALIZATION
# ==============================
if "analytics_initialized" not in st.session_state:
    st.session_state.analytics_initialized = True
    st.markdown(f"""
        <!-- Google Tag Manager -->
        <script>(function(w,d,s,l,i){{
            w[l]=w[l]||[];w[l].push({{'gtm.start':new Date().getTime(),event:'gtm.js'}});
            var f=d.getElementsByTagName(s)[0],j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';
            j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;
            f.parentNode.insertBefore(j,f);
        }})(window,document,'script','dataLayer','{GTM_ID}');</script>
        <!-- End Google Tag Manager -->
        
        <!-- Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          gtag('config', '{GA_ID}');
        </script>
    """, unsafe_allow_html=True)

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

# Add SEO meta tags
st.markdown(f"""
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="google-site-verification" content="f58608a8571b51fd" />
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
</head>
""", unsafe_allow_html=True)

# ==================
# STYLE IMPROVEMENTS (FIXED)
# ==================
# Using f-string with double braces for CSS escaping
css_style = f"""
<style>
:root {{
    --primary: #00C2FF;
    --dark-bg: #0A1F44;
    --darker-bg: #152852;
    --text-light: #FFFFFF;
}}

/* Google Tag Manager iframe */
#iframe-gtm {{
    display: none;
    visibility: hidden;
}}

/* Unified card styling */
.card {{
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    background: linear-gradient(135deg, var(--dark-bg) 0%, var(--darker-bg) 100%);
    border: 1px solid var(--primary);
    margin-bottom: 25px;
    color: var(--text-light);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}}

.card-content {{
    flex-grow: 1;
}}

.card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,194,255,0.2);
}}

/* Improved tech tags */
.tech-tag {{
    display: inline-block;
    background-color: rgba(0, 194, 255, 0.15);
    color: var(--primary);
    border-radius: 4px;
    padding: 4px 12px;
    margin: 4px 2px;
    font-size: 0.8em;
    font-weight: 500;
    border: 1px solid rgba(0, 194, 255, 0.3);
    transition: all 0.2s ease;
}}

.tech-tag:hover {{
    background-color: rgba(0, 194, 255, 0.3);
    transform: scale(1.05);
}}

/* Button enhancements */
.stButton>button {{
    background: linear-gradient(to right, var(--primary), #0077B6);
    color: var(--text-light) !important;
    border: none;
    transition: all 0.3s ease;
    font-weight: 600;
    width: 100%;
}}

.stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,194,255,0.25);
}}

/* Card grid for responsiveness */
.card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}}

/* Responsive fixes for mobile */
@media (max-width: 768px) {{
    .card-grid {{
        grid-template-columns: 1fr;
    }}
    
    .stImage {{
        text-align: center;
        margin-bottom: 20px;
    }}
    
    .card {{
        padding: 15px;
    }}
    
    .footer-cols {{
        flex-direction: column;
        gap: 10px;
    }}
}}

/* Footer styling */
footer {{
    padding: 20px 0;
    font-size: 0.9em;
}}

.footer-cols {{
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
}}

/* Accessibility Improvements */
a:focus, button:focus {{
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}}

/* Performance optimizations */
img {{
    max-width: 100%;
    height: auto;
}}
</style>
"""

# Combine CSS with noscript tag
st.markdown(css_style + f"""
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id={GTM_ID}"
height="0" width="0" style="display:none;visibility:hidden" id="iframe-gtm"></iframe></noscript>
""", unsafe_allow_html=True)

# ================
# HEADER SECTION
# ================
col1, col2 = st.columns([1, 3])
with col1:
    st.image(logo_img, width=150, output_format='PNG', use_column_width='auto', 
            caption="YS Analytics Logo", clamp=False, channels='RGB')
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# ===================
# MISSION STATEMENT
# ===================
st.markdown("""
<div class="card">
    <div class="card-content">
        <h2 style="color: #00C2FF; border-bottom: 2px solid #00C2FF; padding-bottom: 10px; margin-top: 0;">
            Precision Analytics for Financial Markets
        </h2>
        <p style="font-size: 1.1em;">We transform complex market data into actionable intelligence through:</p>
        <ul>
            <li style="margin-bottom: 10px;"><strong>Quantitative Research</strong> • Algorithmic market analysis</li>
            <li style="margin-bottom: 10px;"><strong>Predictive Modeling</strong> • Machine learning-driven forecasts</li>
            <li style="margin-bottom: 10px;"><strong>Strategic Visualization</strong> • Interactive financial dashboards</li>
            <li><strong>Risk Analytics</strong> • Options pricing and volatility insights</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# =================
# FEATURED PROJECTS
# =================
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

# Using CSS grid for responsive layout
st.markdown('<div class="card-grid">', unsafe_allow_html=True)

# Project 1 - Options Analytics Suite
st.markdown("""
<div class="card">
    <div class="card-content">
        <h3>Options Analytics Suite</h3>
        <p>Real-time Greeks calculation and volatility surface visualization</p>
        <div style="margin: 15px 0;">
            <span class="tech-tag">Python</span>
            <span class="tech-tag">Streamlit</span>
            <span class="tech-tag">QuantLib</span>
        </div>
        <button class="stButton" onclick="window.location.href='pages/2_Projects.py'">View Project</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
st.markdown("""
<div class="card">
    <div class="card-content">
        <h3>Market Sector Classifier</h3>
        <p>ML-driven sector analysis using price movement patterns</p>
        <div style="margin: 15px 0;">
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">TA-Lib</span>
            <span class="tech-tag">Plotly</span>
        </div>
        <button class="stButton" onclick="window.location.href='pages/2_Projects.py'">View Project</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
st.markdown("""
<div class="card">
    <div class="card-content">
        <h3>Macroeconomic Dashboard</h3>
        <p>Global economic indicators with forecasting capabilities</p>
        <div style="margin: 15px 0;">
            <span class="tech-tag">FRED API</span>
            <span class="tech-tag">Prophet</span>
            <span class="tech-tag">Altair</span>
        </div>
        <button class="stButton" onclick="window.location.href='pages/2_Projects.py'">View Project</button>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close card-grid

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
st.markdown("""
<div class="footer-cols">
    <div><strong>© 2025 YS Analytics</strong></div>
    <div>
        <a href="https://github.com/wizard5919" target="_blank" aria-label="GitHub Repository">
            <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub">
        </a>
        <a href="https://linkedin.com" target="_blank" aria-label="LinkedIn">
            <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin" alt="LinkedIn">
        </a>
    </div>
    <div><strong>Data Sources:</strong> FRED • Yahoo Finance • OANDA</div>
</div>
""", unsafe_allow_html=True)

# ================
# PERFORMANCE OPTIMIZATION
# ================
# Preconnect to external domains
st.markdown("""
<link rel="preconnect" href="https://raw.githubusercontent.com">
<link rel="preconnect" href="https://www.googletagmanager.com">
<link rel="preconnect" href="https://fonts.googleapis.com">
""", unsafe_allow_html=True)
