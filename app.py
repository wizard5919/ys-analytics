import streamlit as st

# ==============================
# SEO & VERIFICATION IMPROVEMENTS
# ==============================
st.set_page_config(
    page_title="YS Analytics - Financial Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add SEO meta tags and verification
st.markdown(f"""
<head>
    <meta name="google-site-verification" content="f58608a8571b51fd" />
    <meta name="description" content="Quantitative research, predictive modeling, and interactive dashboards for financial markets">
    <meta name="keywords" content="financial analytics, market intelligence, investment research, quantitative finance">
    <link rel="canonical" href="https://app.ysanalytics.me">
    
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXX"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', 'G-XXXXXX');
    </script>
</head>
""", unsafe_allow_html=True)

# ==================
# STYLE IMPROVEMENTS
# ==================
st.markdown("""
<style>
/* Unified card styling */
.card {
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    background: linear-gradient(135deg, #0A1F44 0%, #152852 100%);
    border: 1px solid #00C2FF;
    margin-bottom: 25px;
    color: white;
}

/* Improved tech tags */
.tech-tag {
    display: inline-block;
    background-color: rgba(0, 194, 255, 0.15);
    color: #00C2FF;
    border-radius: 4px;
    padding: 4px 12px;
    margin: 4px 2px;
    font-size: 0.8em;
    font-weight: 500;
    border: 1px solid rgba(0, 194, 255, 0.3);
}

/* Button enhancements */
.stButton>button {
    background: linear-gradient(to right, #00C2FF, #0077B6);
    color: white !important;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,194,255,0.25);
}

/* Responsive fixes for mobile */
@media (max-width: 768px) {
    .col1, .col2, .col3 {
        min-width: 100% !important;
    }
    .stImage {
        text-align: center;
        margin-bottom: 20px;
    }
}
</style>
""", unsafe_allow_html=True)

# ================
# HEADER SECTION
# ================
# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# ===================
# MISSION STATEMENT
# ===================
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
    <div style="margin: 10px 0;">
        <span class="tech-tag">Python</span>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">QuantLib</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create link to project section
    if st.button("View Project", key="p1", use_container_width=True):
        st.switch_page("pages/2_📚_Projects.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Market Sector Classifier")
    st.markdown("ML-driven sector analysis using price movement patterns")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span class="tech-tag">Scikit-learn</span>
        <span class="tech-tag">TA-Lib</span>
        <span class="tech-tag">Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p2", use_container_width=True):
        st.switch_page("pages/2_📚_Projects.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Macroeconomic Dashboard")
    st.markdown("Global economic indicators with forecasting capabilities")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span class="tech-tag">FRED API</span>
        <span class="tech-tag">Prophet</span>
        <span class="tech-tag">Altair</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p3", use_container_width=True):
        st.switch_page("pages/2_📚_Projects.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================
# CALL TO ACTION
# ================
st.markdown("---")
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_📚_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link("pages/3_📈_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link("pages/5_✉️_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# ================
# FOOTER SECTION
# ================
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**© 2025 YS Analytics**")
with footer_cols[1]:
    st.markdown("[GitHub](https://github.com/wizard5919) • [LinkedIn](https://linkedin.com)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
