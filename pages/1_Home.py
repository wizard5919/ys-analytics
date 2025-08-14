import streamlit as st

# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# Set page config first
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon=LOGO_URL,
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "### YS Analytics\nFinancial Analytics Platform\n"
    }
)

# Add global CSS - moved to top for better rendering
st.markdown("""
<style>
:root {
    --primary: #0A1F44;
    --accent: #00C2FF;
    --light: #F5F9FC;
    --dark: #0A1F44;
}

.stApp {
    background-color: white;
    background-image: radial-gradient(var(--light) 1px, transparent 1px);
    background-size: 20px 20px;
}

h1, h2, h3, h4 {
    color: var(--primary) !important;
}

.stButton button {
    background-color: var(--accent) !important;
    color: var(--dark) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    border: none !important;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 194, 255, 0.3);
}

.card {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08);
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
    background-color: white;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(10, 31, 68, 0.15) !important;
}

.tech-tag {
    display: inline-block;
    background-color: var(--dark);
    color: var(--accent);
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px 4px 2px 0;
    font-size: 0.8em;
    font-weight: 500;
}

/* Responsive columns */
@media (max-width: 768px) {
    .column-card {
        margin-bottom: 1.5rem;
    }
}

.footer {
    font-size: 0.85rem;
    color: #666;
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Page header with improved responsive design
header_col1, header_col2 = st.columns([1, 3])
with header_col1:
    st.image(LOGO_URL, width=120)
with header_col2:
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

# Responsive columns for projects
col1, col2, col3 = st.columns(3, gap="medium")

projects = [
    {
        "title": "Options Analytics Suite",
        "desc": "Real-time Greeks calculation and volatility surface visualization",
        "tags": ["Python", "Streamlit", "QuantLib"],
        "page": "Options_Analyzer"
    },
    {
        "title": "Market Sector Classifier",
        "desc": "ML-driven sector analysis using price movement patterns",
        "tags": ["Scikit-learn", "TA-Lib", "Plotly"],
        "page": "Sector_Classifier"
    },
    {
        "title": "Macroeconomic Dashboard",
        "desc": "Global economic indicators with forecasting capabilities",
        "tags": ["FRED API", "Prophet", "Altair"],
        "page": "Macro_Dashboard"
    }
]

for i, col in enumerate([col1, col2, col3]):
    with col:
        project = projects[i]
        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.subheader(project["title"])
            st.markdown(project["desc"])
            
            # Tech tags
            tags_html = "".join(
                f'<span class="tech-tag">{tag}</span>' 
                for tag in project["tags"]
            )
            st.markdown(f'<div style="margin: 10px 0;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Navigation button
            if st.button("View Project", key=f"p{i+1}", use_container_width=True):
                st.switch_page(f"pages/6_{project['page']}.py")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Call to action section
st.divider()
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", use_container_width=True)
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", use_container_width=True)
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", use_container_width=True)

# Footer section
st.divider()
footer_cols = st.columns([2, 3, 2])
with footer_cols[0]:
    st.markdown("**© 2024 YS Analytics**", help="Financial Analytics Platform")
with footer_cols[1]:
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/wizard5919)
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com)
    """, unsafe_allow_html=True)
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
