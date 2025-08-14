import streamlit as st

# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# MUST be the first command
st.set_page_config(
    page_title="YS Analytics",
    page_icon=LOGO_URL,
    layout="wide",
    initial_sidebar_state="auto"
)

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Projects", "Dashboard", "Contact"])

if page == "Home":
    # Home page content will be shown in main area
    pass
elif page == "Projects":
    st.switch_page("pages/2_Projects.py")
elif page == "Dashboard":
    st.switch_page("pages/3_Dashboard.py")
elif page == "Contact":
    st.switch_page("pages/5_Contact.py")

# Home Page Content
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# Mission statement
st.markdown("""
<div style='
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08);
    padding: 1.5rem;
    margin: 1.5rem 0;
    background-color: white;
    border-top: 3px solid #00C2FF;
'>
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

# Featured projects
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

col1, col2, col3 = st.columns(3)

# Project 1 - Options Analytics Suite
with col1:
    st.markdown('<div style="border-radius: 12px; box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08); padding: 1.5rem; margin: 1.5rem 0; background-color: white; border-top: 3px solid #00C2FF;">', unsafe_allow_html=True)
    st.subheader("Options Analytics Suite")
    st.markdown("Real-time Greeks calculation and volatility surface visualization")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">Python</span>
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">Streamlit</span>
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">QuantLib</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p1", use_container_width=True):
        st.switch_page("pages/6_Options_Analyzer.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
with col2:
    st.markdown('<div style="border-radius: 12px; box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08); padding: 1.5rem; margin: 1.5rem 0; background-color: white; border-top: 3px solid #00C2FF;">', unsafe_allow_html=True)
    st.subheader("Market Sector Classifier")
    st.markdown("ML-driven sector analysis using price movement patterns")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">Scikit-learn</span>
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">TA-Lib</span>
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p2", use_container_width=True):
        st.switch_page("pages/7_Sector_Classifier.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
with col3:
    st.markdown('<div style="border-radius: 12px; box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08); padding: 1.5rem; margin: 1.5rem 0; background-color: white; border-top: 3px solid #00C2FF;">', unsafe_allow_html=True)
    st.subheader("Macroeconomic Dashboard")
    st.markdown("Global economic indicators with forecasting capabilities")
    st.markdown("""
    <div style="margin: 10px 0;">
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">FRED API</span>
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">Prophet</span>
        <span style="display: inline-block; background-color: #0A1F44; color: #00C2FF; border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.8em; font-weight: 500;">Altair</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("View Project", key="p3", use_container_width=True):
        st.switch_page("pages/8_Macro_Dashboard.py")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Call to action
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
    st.markdown("**© 2024 YS Analytics**")
with footer_cols[1]:
    st.markdown("[GitHub](https://github.com/wizard5919) • [LinkedIn](https://linkedin.com)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
