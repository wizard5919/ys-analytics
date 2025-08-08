import streamlit as st
import os
from PIL import Image

# Check environment (local vs cloud)
is_cloud = os.path.exists('/mount/src/ys-analytics')

# Try to load logo
try:
    if is_cloud:
        # Streamlit Cloud path
        logo_path = '/mount/src/ys-analytics/assets/logo.png'
    else:
        # Local path
        logo_path = 'assets/logo.png'
    
    logo = Image.open(logo_path)
    has_logo = True
except Exception as e:
    # Fallback to GitHub URL
    try:
        logo_url = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"
        has_logo = True
    except:
        st.warning("Logo could not be loaded. Please check the file path.")
        has_logo = False

# Page header
if has_logo:
    col1, col2 = st.columns([1, 3])
    with col1:
        # Use different method based on source
        if 'logo_url' in locals():
            st.image(logo_url, use_container_width=True)
        else:
            st.image(logo, use_container_width=True)
    with col2:
        st.title("YS Analytics")
        st.markdown("**Data-Driven Market Intelligence**")
else:
    # Text-based fallback
    st.markdown("""
    <div style="background-color:#0A1F44; padding:20px; border-radius:12px; margin-bottom:20px">
        <h1 style="color:white; margin:0">YS Analytics</h1>
        <p style="color:#00C2FF; margin:0">Data-Driven Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# Mission statement
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

# Featured projects
st.header("Featured Analytics Projects")
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Options Analytics Suite")
    st.markdown("Real-time Greeks calculation and volatility surface visualization")
    st.markdown("`Python` `Streamlit` `QuantLib`")
    st.button("View Project", key="p1", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Market Sector Classifier")
    st.markdown("ML-driven sector analysis using price movement patterns")
    st.markdown("`Scikit-learn` `TA-Lib` `Plotly`")
    st.button("View Project", key="p2", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Macroeconomic Dashboard")
    st.markdown("Global economic indicators with forecasting capabilities")
    st.markdown("`FRED API` `Prophet` `Altair`")
    st.button("View Project", key="p3", use_container_width=True)
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
