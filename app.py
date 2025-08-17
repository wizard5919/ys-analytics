import streamlit as st

# ==============================
# GOOGLE ANALYTICS INITIALIZATION
# ==============================
if "analytics_initialized" not in st.session_state:
    st.session_state.analytics_initialized = True
    st.markdown("""
        <!-- Google Tag Manager -->
        <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
        new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
        j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
        'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
        })(window,document,'script','dataLayer','GTM-XXXXXX');</script>
        <!-- End Google Tag Manager -->
        
        <!-- Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXX"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
          gtag('config', 'G-XXXXXX');
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
)

# Add SEO meta tags
st.markdown("""
<head>
    <meta name="google-site-verification" content="f58608a8571b51fd" />
    <meta name="description" content="Quantitative research, predictive modeling, and interactive dashboards for financial markets">
    <meta name="keywords" content="financial analytics, market intelligence, investment research, quantitative finance">
    <link rel="canonical" href="https://app.ysanalytics.me">
</head>
""", unsafe_allow_html=True)

# ==================
# STYLE IMPROVEMENTS (UPDATED: Warmer text, more spacing, responsive fonts, higher tag contrast, smoother transitions)
# ==================
st.markdown("""
<style>
/* Google Tag Manager iframe */
#iframe-gtm {
    display: none;
    visibility: hidden;
}

/* Unified card styling (Increased padding to 30px, added margin-top:40px for sections) */
.card {
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    background: linear-gradient(135deg, #0A1F44 0%, #152852 100%);
    border: 1px solid #00C2FF;
    margin-bottom: 25px;
    margin-top: 40px;  /* Added for better section spacing */
    color: #E0E0E0;  /* Warmer gray text for better readability */
    transition: transform 0.3s ease, box-shadow 0.3s ease, opacity 0.3s ease;  /* Smoother with opacity */
    opacity: 1;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,194,255,0.2);
}

/* Improved tech tags (Higher opacity for contrast) */
.tech-tag {
    display: inline-block;
    background-color: rgba(0, 194, 255, 0.25);  /* Increased opacity */
    color: #00C2FF;
    border-radius: 4px;
    padding: 4px 12px;
    margin: 4px 2px;
    font-size: 0.8em;
    font-weight: 500;
    border: 1px solid rgba(0, 194, 255, 0.3);
    transition: all 0.2s ease;
    text-shadow: 0 0 2px rgba(255,255,255,0.1);  /* Subtle shadow for legibility */
}

.tech-tag:hover {
    background-color: rgba(0, 194, 255, 0.3);
    transform: scale(1.05);
}

/* Button enhancements (More vibrant hover) */
.stButton>button {
    background: linear-gradient(to right, #00C2FF, #0077B6);
    color: white !important;
    border: none;
    transition: all 0.3s ease;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(to right, #00E5FF, #0099D9);  /* Brighter vibrant color */
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,194,255,0.25);
    border: 1px solid #00E5FF;  /* Subtle border for emphasis */
}

/* Responsive fixes for mobile (Increased padding, responsive fonts) */
@media (max-width: 768px) {
    .col1, .col2, .col3 {
        min-width: 100% !important;
    }
    .stImage {
        text-align: center;
        margin-bottom: 20px;
    }
    .card {
        padding: 25px;  /* Increased from 15px */
    }
    h2 { font-size: clamp(1.5rem, 6vw, 2rem); }  /* Responsive headings */
    p { font-size: clamp(0.9rem, 3vw, 1.1rem); line-height: 1.6; }  /* Better readability */
}

/* Footer styling (Added padding-top) */
footer {
    padding: 40px 0 20px 0;  /* Increased top padding */
    font-size: 0.9em;
}
</style>

<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-XXXXXX"
height="0" width="0" style="display:none;visibility:hidden" id="iframe-gtm"></iframe></noscript>
""", unsafe_allow_html=True)

# ================
# HEADER SECTION (Added centering on mobile via CSS if needed)
# ================
# GitHub raw URL for your logo (Consider CDN for faster load)
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# ===================
# MISSION STATEMENT (Added fade-in via CSS transition)
# ===================
st.markdown("""
<div class="card">
    <h2 style="color: #00C2FF; border-bottom: 2px solid #00C2FF; padding-bottom: 10px; margin-top: 0;">Precision Analytics for Financial Markets</h2>
    <p style="font-size: 1.1em;">We transform complex market data into actionable intelligence through:</p>
    <ul>
        <li style="margin-bottom: 10px;"><strong>Quantitative Research</strong> • Algorithmic market analysis</li>
        <li style="margin-bottom: 10px;"><strong>Predictive Modeling</strong> • Machine learning-driven forecasts</li>
        <li style="margin-bottom: 10px;"><strong>Strategic Visualization</strong> • Interactive financial dashboards</li>
        <li><strong>Risk Analytics</strong> • Options pricing and volatility insights</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# =================
# FEATURED PROJECTS (Added icons to cards for visual polish)
# =================
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

col1, col2, col3 = st.columns(3)

# Project 1 - Options Analytics Suite (Added icon emoji as placeholder)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Options Analytics Suite")  # Added icon
    st.markdown("Real-time Greeks calculation and volatility surface visualization")
    st.markdown("""
    <div style="margin: 15px 0;">
        <span class="tech-tag">Python</span>
        <span class="tech-tag">Streamlit</span>
        <span class="tech-tag">QuantLib</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create link to project section (Added icon to button)
    if st.button("🔍 View Project", key="p1", use_container_width=True):  # Added icon
        st.switch_page("pages/2_Projects.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Market Sector Classifier")  # Added icon
    st.markdown("ML-driven sector analysis using price movement patterns")
    st.markdown("""
    <div style="margin: 15px 0;">
        <span class="tech-tag">Scikit-learn</span>
        <span class="tech-tag">TA-Lib</span>
        <span class="tech-tag">Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 View Project", key="p2", use_container_width=True):
        st.switch_page("pages/2_Projects.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌍 Macroeconomic Dashboard")  # Added icon
    st.markdown("Global economic indicators with forecasting capabilities")
    st.markdown("""
    <div style="margin: 15px 0;">
        <span class="tech-tag">FRED API</span>
        <span class="tech-tag">Prophet</span>
        <span class="tech-tag">Altair</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 View Project", key="p3", use_container_width=True):
        st.switch_page("pages/2_Projects.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================
# CALL TO ACTION (Vibrant colors applied via CSS)
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
# FOOTER SECTION (Improved alignment with flexbox)
# ================
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**© 2025 YS Analytics**")
with footer_cols[1]:
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/wizard5919) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
