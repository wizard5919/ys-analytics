import streamlit as st

# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# Fixed CSS with proper contrast
st.markdown(f"""
<style>
:root {{
    --primary: #0A1F44;
    --accent: #00C2FF;
    --light: #F5F9FC;
    --dark: #0A1F44;
    --gray: #6c757d;
    --text-light: #FFFFFF;  /* Added light text color */
    --text-dark: #0A1F44;   /* Dark text color */
}}

.stApp {{
    background-color: #0A1F44;  /* Dark blue background */
    color: var(--text-light) !important;  /* Light text color */
    padding-top: 1rem;
}}

.header-container {{
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}}

h1, h2, h3, h4 {{
    color: var(--text-light) !important;  /* Light text for headings */
}}

.card {{
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    padding: 1.5rem;
    margin: 1.5rem 0;
    background-color: rgba(255, 255, 255, 0.1);  /* Semi-transparent white */
    border-top: 3px solid var(--accent);
    color: var(--text-light) !important;  /* Light text in cards */
}}

/* Make all text light */
p, li, div {{
    color: var(--text-light) !important;
}}

/* Style links to be visible */
a {{
    color: var(--accent) !important;
}}
</style>
""", unsafe_allow_html=True)

# Page header
header_container = st.container()
with header_container:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(LOGO_URL, width=120)
    with col2:
        st.title("YS Analytics")
        st.markdown("**Data-Driven Market Intelligence**")
    st.divider()

# Mission statement
with st.container():
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

# Featured projects section
st.header("Featured Analytics Projects")
st.caption("Select case studies demonstrating our financial analytics capabilities")

# Project 1 - Options Analytics Suite
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Options Analytics Suite")
    st.markdown("Real-time Greeks calculation and volatility surface visualization")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">Python</span>
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">Streamlit</span>
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">QuantLib</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation button
    if st.button("View Project", key="p1", use_container_width=True):
        st.switch_page("pages/6_Options_Analyzer.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Market Sector Classifier")
    st.markdown("ML-driven sector analysis using price movement patterns")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">Scikit-learn</span>
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">TA-Lib</span>
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p2", use_container_width=True):
        st.switch_page("pages/7_Sector_Classifier.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Macroeconomic Dashboard")
    st.markdown("Global economic indicators with forecasting capabilities")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">FRED API</span>
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">Prophet</span>
        <span style="display: inline-block; background-color: #00C2FF; color: #0A1F44; border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.8em; font-weight: 500;">Altair</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p3", use_container_width=True):
        st.switch_page("pages/8_Macro_Dashboard.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Call to action section
st.divider()
st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# Footer section
st.divider()
st.markdown("**© 2024 YS Analytics** • [GitHub](https://github.com/wizard5919) • [LinkedIn](https://linkedin.com)")
st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
