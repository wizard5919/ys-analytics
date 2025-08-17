import streamlit as st

def load_global_css():
    """Load global CSS for consistency across pages."""
    st.markdown("""
    <style>
    /* Base Theme: Warmer text, smoother animations */
    body { color: #E0E0E0; font-family: 'Arial', sans-serif; line-height: 1.8; }
    .card {
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #0A1F44 0%, #152852 100%);
        border: 1px solid #00C2FF;
        margin: 40px 0;
        color: #E0E0E0;
        transition: transform 0.4s ease, box-shadow 0.4s ease, opacity 0.4s ease;
        opacity: 0;  /* Start faded for animation */
        animation: fadeIn 0.5s forwards;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .card:hover { transform: translateY(-8px); box-shadow: 0 12px 24px rgba(0,194,255,0.3); }
    
    /* Tech Tags: Higher contrast */
    .tech-tag {
        background-color: rgba(0, 194, 255, 0.4);
        color: #00C2FF;
        border-radius: 6px;
        padding: 6px 14px;
        margin: 6px 4px;
        font-size: 0.85em;
        font-weight: 600;
        border: 1px solid rgba(0, 194, 255, 0.4);
        transition: all 0.3s ease;
    }
    .tech-tag:hover { background-color: rgba(0, 194, 255, 0.5); transform: scale(1.08); }
    
    /* Buttons: More vibrant */
    .stButton>button {
        background: linear-gradient(to right, #00C2FF, #0077B6);
        color: white !important;
        border: none;
        border-radius: 8px;
        transition: all 0.4s ease;
        font-weight: 700;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #00E5FF, #0099D9);
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,194,255,0.3);
        border: 1px solid #00E5FF;
    }
    
    /* Responsive: Mobile-first */
    @media (max-width: 768px) {
        .card { padding: 30px; }
        h1, h2 { font-size: clamp(1.8rem, 8vw, 2.5rem); }
        p, li { font-size: clamp(1rem, 4vw, 1.2rem); }
        .st-columns > div { flex-direction: column; }
    }
    
    /* Sticky Header */
    .sticky-header {
        position: sticky;
        top: 0;
        background: #0A1F44;
        padding: 10px;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Icons via Font Awesome CDN */
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)

def render_sidebar_navigation():
    """Render grouped sidebar with icons and tooltips."""
    with st.sidebar:
        st.title("Navigation")
        with st.expander("Main Pages", expanded=True):
            st.page_link("pages/1_Home.py", label="Home", icon="🏠")
            st.page_link("pages/2_Projects.py", label="Projects", icon="📚")
            st.page_link("pages/3_Dashboard.py", label="Dashboard", icon="📈")
            st.page_link("pages/4_Insights.py", label="Insights", icon="💡")
            st.page_link("pages/5_Contact.py", label="Contact", icon="✉️")
        with st.expander("Tools & Demos", expanded=False):
            st.page_link("pages/6_Options_Analyzer.py", label="Options Analyzer", icon="📊")
            st.page_link("pages/7_Sector_Classifier.py", label="Sector Classifier", icon="🤖")
            st.page_link("pages/8_Macro_Dashboard.py", label="Macro Dashboard", icon="🌍")

def render_project_search(projects):
    """AI-like search bar for filtering projects."""
    search_query = st.text_input("AI-Powered Search: Find Projects", placeholder="e.g., 'options' or 'ML'")
    if search_query:
        filtered = [p for p in projects if search_query.lower() in p['title'].lower() or search_query.lower() in p['desc'].lower()]
        return filtered
    return projects

def render_breadcrumbs(current_page):
    """Simple breadcrumbs for flow."""
    st.markdown(f"<div class='sticky-header'>Home > {current_page}</div>", unsafe_allow_html=True)
