import streamlit as st

# GitHub raw URL for your logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"

# Set page config
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon=LOGO_URL,
    layout="centered",
    initial_sidebar_state="auto",
)

# Enhanced CSS with better spacing and visual hierarchy
st.markdown(f"""
<style>
:root {{
    --primary: #0A1F44;
    --accent: #00C2FF;
    --light: #F5F9FC;
    --dark: #0A1F44;
    --gray: #6c757d;
}}

.stApp {{
    background-color: white;
    background-image: radial-gradient(var(--light) 1px, transparent 1px);
    background-size: 20px 20px;
    padding-top: 1rem;
}}

.header-container {{
    border-bottom: 1px solid rgba(10, 31, 68, 0.1);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}}

h1, h2, h3, h4 {{
    color: var(--primary) !important;
}}

h2 {{
    border-left: 4px solid var(--accent);
    padding-left: 0.75rem;
    margin-top: 1.5rem !important;
}}

.card {{
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(10, 31, 68, 0.08);
    padding: 1.5rem;
    margin: 1.5rem 0;
    transition: all 0.3s ease;
    background-color: white;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    border-top: 3px solid var(--accent);
}}

.card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 194, 255, 0.15) !important;
}}

.tech-tag {{
    display: inline-block;
    background-color: var(--dark);
    color: var(--accent);
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px 4px 2px 0;
    font-size: 0.8em;
    font-weight: 500;
}}

.project-card h3 {{
    margin-top: 0 !important;
    color: var(--primary) !important;
}}

.cta-button {{
    margin-top: 1rem !important;
}}

.footer {{
    font-size: 0.85rem;
    color: var(--gray);
    padding-top: 1.5rem;
    border-top: 1px solid rgba(10, 31, 68, 0.1);
    margin-top: 2rem;
}}

/* Responsive adjustments */
@media (max-width: 768px) {{
    .column-card {{
        margin-bottom: 1.5rem;
    }}
    
    .header-container {{
        flex-direction: column;
        text-align: center;
    }}
    
    .header-container img {{
        margin: 0 auto 1rem;
    }}
}}
</style>
""", unsafe_allow_html=True)

# Page header with improved spacing
header_container = st.container()
with header_container:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(LOGO_URL, width=120)
    with col2:
        st.title("YS Analytics")
        st.markdown("**Data-Driven Market Intelligence**")
    st.divider()

# Mission statement with enhanced typography
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

# Featured projects section with improved layout
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
            st.markdown(f'<div class="card project-card">', unsafe_allow_html=True)
            st.subheader(project["title"])
            st.markdown(f'<p style="color: var(--gray);">{project["desc"]}</p>', unsafe_allow_html=True)
            
            # Tech tags
            tags_html = "".join(
                f'<span class="tech-tag">{tag}</span>' 
                for tag in project["tags"]
            )
            st.markdown(f'<div style="margin: 10px 0;">{tags_html}</div>', unsafe_allow_html=True)
            
            # Navigation button
            if st.button("View Project", key=f"p{i+1}", use_container_width=True, 
                         help=f"Explore the {project['title']}"):
                st.switch_page(f"pages/6_{project['page']}.py")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Call to action section with improved buttons
st.divider()
cta_cols = st.columns(3)
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", 
                icon="📚", use_container_width=True, 
                help="View all our analytics projects")
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", 
                icon="📈", use_container_width=True,
                help="Access real-time market data")
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", 
                icon="✉️", use_container_width=True,
                help="Get in touch with our team")

# Enhanced footer with social links
st.divider()
footer = st.container()
with footer:
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.markdown("**© 2024 YS Analytics**", help="Financial Analytics Platform")
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <a href="https://github.com/wizard5919" target="_blank" style="margin: 0 10px; text-decoration: none; color: var(--primary);">
                <span>GitHub</span>
            </a>
            <span>•</span>
            <a href="https://linkedin.com" target="_blank" style="margin: 0 10px; text-decoration: none; color: var(--primary);">
                <span>LinkedIn</span>
            </a>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")
