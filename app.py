import streamlit as st
from utils import load_global_css, render_sidebar_navigation, render_project_search

# Analytics init
if "analytics_initialized" not in st.session_state:
    st.session_state.analytics_initialized = True
    st.markdown("""<!-- Google Tag Manager and Analytics code -->""", unsafe_allow_html=True)

st.set_page_config(
    page_title="YS Analytics - Financial Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load global CSS
load_global_css()

# Add custom styling for perfect layout
st.markdown("""
<style>
    /* Unified card styling */
    .project-card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        background: #0E1117;
        border: 1px solid #1e3a5f;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
        margin-bottom: 25px;
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 194, 255, 0.15);
        border-color: #00C2FF;
    }
    
    /* Consistent title styling */
    .project-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00C2FF;
        margin-top: 0;
        margin-bottom: 15px;
        min-height: 60px;
        line-height: 1.4;
    }
    
    /* Better tag styling */
    .tech-tag {
        background: rgba(0, 194, 255, 0.15);
        color: #00C2FF;
        border: 1px solid #00C2FF;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0 5px 8px 0;
    }
    
    /* Button styling */
    .project-button {
        background: linear-gradient(135deg, #00C2FF, #0077FF) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        margin-top: auto;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .project-button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(0, 194, 255, 0.4);
    }
    
    /* Description styling */
    .project-desc {
        color: #a0aec0;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 15px 0;
        min-height: 70px;
    }
    
    /* Section spacing */
    .section-header {
        margin-bottom: 30px !important;
    }
    
    .section-subheader {
        margin-bottom: 20px !important;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #1e3a5f;
        font-size: 0.9rem;
    }
    
    /* Image styling */
    .project-image {
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 15px;
        border: 1px solid #1e3a5f;
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .project-image img {
        object-fit: cover;
        width: 100%;
        height: 100%;
    }
    
    /* Tag container */
    .tag-container {
        margin: 15px 0;
        min-height: 40px;
    }
    
    /* CTA button styling */
    .cta-button {
        width: 100%;
        margin: 10px 0;
        text-align: center;
        font-weight: 600 !important;
    }
    
    /* Logo sizing */
    .logo-container {
        margin-bottom: 20px;
    }
    
    /* Search bar styling */
    .search-container {
        margin-bottom: 30px;
    }
    
    /* Content container */
    .content-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Column spacing */
    .stColumn > div {
        padding: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# Render sidebar
render_sidebar_navigation()

# Create main content container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Header with consistent logo
LOGO_URL = "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/logo.png"
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 4])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("YS Analytics")
    st.markdown("**Data-Driven Market Intelligence**")
st.markdown('</div>', unsafe_allow_html=True)

# Mission statement card
st.markdown("""
<div class="card">
    <h2 style="color: #00C2FF; border-bottom: 2px solid #00C2FF; padding-bottom: 12px; margin-top: 0;">Precision Analytics for Financial Markets</h2>
    <p style="font-size: 1.05rem;">We transform complex market data into actionable intelligence through:</p>
    <ul style="font-size: 1.05rem; line-height: 1.8; margin-bottom: 0;">
        <li><strong>Quantitative Research</strong> • Algorithmic market analysis</li>
        <li><strong>Predictive Modeling</strong> • Machine learning-driven forecasts</li>
        <li><strong>Strategic Visualization</strong> • Interactive financial dashboards</li>
        <li><strong>Risk Analytics</strong> • Options pricing and volatility insights</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Featured Projects with search and previews
st.markdown('<div class="section-header">', unsafe_allow_html=True)
st.header("Featured Analytics Projects")
st.markdown('<div class="section-subheader">', unsafe_allow_html=True)
st.markdown("***Select case studies demonstrating our financial analytics capabilities***")
st.markdown('</div>', unsafe_allow_html=True)

# Add search component
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown("**AI-Powered Search: Find Projects**")
st.caption("e.g., 'options' or 'ML'")
st.markdown('</div>', unsafe_allow_html=True)

projects = [
    {"title": "Options Analytics Suite", "icon": "📈", "desc": "Real-time Greeks calculation and volatility surface visualization", "tags": ["Python", "Streamlit", "QuantLib"], "preview_url": "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/options-preview.png", "link": "pages/2_Projects.py"},
    {"title": "Market Sector Classifier", "icon": "🤖", "desc": "ML-driven sector analysis using price movement patterns", "tags": ["Scikit-learn", "TA-Lib", "Plotly"], "preview_url": "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/sector-preview.png", "link": "pages/2_Projects.py"},
    {"title": "Macroeconomic Dashboard", "icon": "🌍", "desc": "Global economic indicators with forecasting capabilities", "tags": ["FRED API", "Prophet", "Altair"], "preview_url": "https://raw.githubusercontent.com/wizard5919/ys-analytics/main/assets/macro-preview.png", "link": "pages/2_Projects.py"}
]

filtered_projects = render_project_search(projects)

cols = st.columns(3, gap="medium")
for i, proj in enumerate(filtered_projects):
    with cols[i]:
        st.markdown('<div class="project-card">', unsafe_allow_html=True)
        
        # Project title with icon
        st.markdown(f'<div class="project-title">{proj["icon"]} {proj["title"]}</div>', unsafe_allow_html=True)
        
        # Project image
        st.markdown('<div class="project-image">', unsafe_allow_html=True)
        st.image(proj["preview_url"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Project description
        st.markdown(f'<div class="project-desc">{proj["desc"]}</div>', unsafe_allow_html=True)
        
        # Technology tags
        st.markdown('<div class="tag-container">', unsafe_allow_html=True)
        st.markdown(" ".join([f'<span class="tech-tag">{tag}</span>' for tag in proj["tags"]]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # View project button
        if st.button("🔍 View Project", key=f"p{i}", use_container_width=True, 
                    help=f"Explore the {proj['title']} project", 
                    type="primary", 
                    kwargs={"class": "project-button"}):
            st.switch_page(proj["link"])
        
        st.markdown("</div>", unsafe_allow_html=True)

# CTA section
st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
st.markdown("---")
cta_cols = st.columns(3, gap="medium")
with cta_cols[0]:
    st.page_link("pages/2_Projects.py", label="Explore Full Portfolio", icon="📚", 
                use_container_width=True, help="View all our analytics projects")
with cta_cols[1]:
    st.page_link("pages/3_Dashboard.py", label="Live Market Dashboard", icon="📈", 
                use_container_width=True, help="Access real-time market data")
with cta_cols[2]:
    st.page_link("pages/5_Contact.py", label="Schedule Consultation", icon="✉️", 
                use_container_width=True, help="Contact our analytics team")

# Footer
st.markdown('<div class="footer"></div>', unsafe_allow_html=True)
st.markdown("---")
footer_cols = st.columns([2, 3, 2])
with footer_cols[0]:
    st.markdown("**© 2025 YS Analytics**")
with footer_cols[1]:
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 20px;">
        <a href="https://github.com/wizard5919" target="_blank" style="color: #00C2FF; text-decoration: none;">
            <i class="fab fa-github"></i> GitHub
        </a>
        <a href="https://linkedin.com" target="_blank" style="color: #00C2FF; text-decoration: none;">
            <i class="fab fa-linkedin"></i> LinkedIn
        </a>
    </div>
    """, unsafe_allow_html=True)
with footer_cols[2]:
    st.markdown("**Data Sources:** FRED • Yahoo Finance • OANDA")

st.markdown('</div>', unsafe_allow_html=True)  # Close content container
