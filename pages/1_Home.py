import streamlit as st

# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# Set page config - MUST be first
st.set_page_config(
    page_title="YS Analytics | Home",
    page_icon=LOGO_URL,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "### YS Analytics\nFinancial Analytics Platform\n"
    }
)

# Add custom CSS styling
st.markdown("""
<style>
:root {
    --primary: #0A1F44;
    --accent: #00C2FF;
    --light: #F5F9FC;
}

.stApp {
    background-color: var(--light);
}

h1, h2, h3, h4 {
    color: var(--primary) !important;
}

.card {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08);
    padding: 1.5rem;
    margin: 1rem 0;
    background-color: white;
    border-top: 3px solid var(--accent);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 194, 255, 0.15);
}

.tech-tag {
    display: inline-block;
    background-color: var(--primary);
    color: var(--accent);
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.8em;
    font-weight: 500;
}

.stButton button {
    background-color: var(--accent) !important;
    color: var(--primary) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 194, 255, 0.3);
}

.stPageLink a {
    display: block;
    text-align: center;
    padding: 0.5rem;
    background-color: var(--primary);
    color: var(--accent) !important;
    border-radius: 8px;
    text-decoration: none;
    transition: all 0.3s ease;
}

.stPageLink a:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(10, 31, 68, 0.2);
}
</style>
""", unsafe_allow_html=True)

# Page header
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=120)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

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
st.caption("Select case studies demonstrating our financial analytics capabilities")

col1, col2, col3 = st.columns(3)

# Project 1 - Options Analytics Suite
with col1:
    with st.container():
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
        
        if st.button("View Project", key="p1", use_container_width=True):
            st.switch_page("pages/6_Options_Analyzer.py")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
with col2:
    with st.container():
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
            st.switch_page("pages/7_Sector_Classifier.py")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
with col3:
    with st.container():
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
            st.switch_page("pages/8_Macro_Dashboard.py")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Call to action
st.divider()
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# Footer
st.divider()
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**© 2024 YS Analytics**")
with footer_cols[1]:
    st.markdown("[GitHub](https://github.com/wizard5919) • [LinkedIn](https://linkedin.com)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
