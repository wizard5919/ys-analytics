import streamlit as st
from pathlib import Path
from pages import home, projects, dashboard, insights, contact, options_analyzer, sector_classifier, macro_dashboard



# Optional: set page config globally
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon="📊",
    layout="wide"
)

# Optional: Add a small welcome / redirect notice
st.markdown(
    """
    <div style="text-align:center; margin-top:50px;">
        <h2>Welcome to YS Analytics</h2>
        <p>Redirecting to the Home page...</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Automatically redirect to Home page
# Streamlit multipage apps detect 'pages/' folder automatically
# So just importing the Home page ensures it is loaded
try:
    from pages import _1_Home  # Make sure the module name matches (rename 1_Home.py to _1_Home.py if needed)
except ModuleNotFoundError:
    st.warning("Home page module not found. Make sure 1_Home.py exists inside the pages/ folder.")

# Optional: list all pages
st.markdown("---")
st.markdown("**Available Pages:**")
pages_folder = Path("pages")
for page_file in sorted(pages_folder.glob("*.py")):
    page_name = page_file.stem.replace("_", " ").replace("1 Home", "Home")
    st.write(f"- {page_name}")
