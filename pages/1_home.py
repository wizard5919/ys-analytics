import streamlit as st
from pathlib import Path

# --- Page config ---
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# --- Logo ---
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# --- Header ---
col1, col2 = st.columns([1, 3])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")

# --- Mission ---
st.subheader("Precision Analytics for Financial Markets")
st.write("""
We transform complex market data into actionable intelligence through:
- **Quantitative Research** • Algorithmic market analysis
- **Predictive Modeling** • Machine learning-driven forecasts
- **Strategic Visualization** • Interactive financial dashboards
- **Risk Analytics** • Options pricing and volatility insights
""")

# --- Projects ---
st.header("Featured Analytics Projects")
projects = [
    {
        "title": "Options Analytics Suite",
        "description": "Real-time Greeks calculation and volatility surface visualization",
        "tech": ["Python", "Streamlit", "QuantLib"],
        "page": "2_Projects"  # <-- Streamlit page name, not file path
    },
    {
        "title": "Market Sector Classifier",
        "description": "ML-driven sector analysis using price movement patterns",
        "tech": ["Scikit-learn", "TA-Lib", "Plotly"],
        "page": "2_Projects"
    },
    {
        "title": "Macroeconomic Dashboard",
        "description": "Global economic indicators with forecasting capabilities",
        "tech": ["FRED API", "Prophet", "Altair"],
        "page": "2_Projects"
    }
]

cols = st.columns(3)
for col, project in zip(cols, projects):
    with col:
        st.markdown(f"### {project['title']}")
        st.write(project["description"])
        tech_tags = " ".join([f"<span class='tech-tag'>{t}</span>" for t in project["tech"]])
        st.markdown(tech_tags, unsafe_allow_html=True)
        st.page_link(project["page"], label="View Project", icon="📂", use_container_width=True)
        st.markdown("---")
