import streamlit as st
from pathlib import Path
import importlib.util

# Page configuration
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# Define correct page paths
PROJECTS_PAGE = "pages/2_Projects.py"
DASHBOARD_PAGE = "pages/3_Dashboard.py"
CONTACT_PAGE = "pages/4_Contact.py"

# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# Custom CSS for tech tags and cards
st.markdown("""
<style>
.tech-tag {
    display: inline-block;
    background-color: #0A1F44;
    color: #00C2FF;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.8em;
    font-weight: 500;
}
.card {
    background-color: #F8F9FA;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.card:hover {
    transform: scale(1.02);
    transition: 0.3s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# Mission statement
st.subheader("Precision Analytics for Financial Markets")
st.write("""
We transform complex market data into actionable intelligence through:
- **Quantitative Research** • Algorithmic market analysis
- **Predictive Modeling** • Machine learning-driven forecasts
- **Strategic Visualization** • Interactive financial dashboards
- **Risk Analytics** • Options pricing and volatility insights
""")

# Featured projects
st.header("Featured Analytics Projects")
st.markdown("Select case studies demonstrating our financial analytics capabilities")

projects = [
    {
        "title": "Options Analytics Suite",
        "description": "Real-time Greeks calculation and volatility surface visualization",
        "tech": ["Python", "Streamlit", "QuantLib"],
        "page": PROJECTS_PAGE
    },
    {
        "title": "Market Sector Classifier",
        "description": "ML-driven sector analysis using price movement patterns",
        "tech": ["Scikit-learn", "TA-Lib", "Plotly"],
        "page": PROJECTS_PAGE
    },
    {
        "title": "Macroeconomic Dashboard",
        "description": "Global economic indicators with forecasting capabilities",
        "tech": ["FRED API", "Prophet", "Altair"],
        "page": DASHBOARD_PAGE
    }
]

cols = st.columns(3)
for col, project in zip(cols, projects):
    with col:
        with st.container():
            st.markdown(f"### {project['title']}")
            st.write(project["description"])
            tech_tags = " ".join([f"<span class='tech-tag'>{t}</span>" for t in project["tech"]])
            st.markdown(tech_tags, unsafe_allow_html=True)
            st.page_link(project["page"], label="View Project", icon="📂", use_container_width=True)
            st.markdown("---")

# Call to action
st.header("Get Started")
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link(PROJECTS_PAGE, label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link(DASHBOARD_PAGE, label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link(CONTACT_PAGE, label="Schedule Consultation", icon="✉️", use_container_width=True)

# Footer
st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[0]:
    st.markdown("**© 2024 YS Analytics**")
with footer_cols[1]:
    st.markdown("[GitHub](https://github.com/wizard5919) • [LinkedIn](https://linkedin.com)")
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
