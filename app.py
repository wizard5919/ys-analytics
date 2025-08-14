import streamlit as st
import Home  # or from Home import main

# Global page configuration
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon="assets/logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
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
    background-color: white;
    background-image: radial-gradient(var(--light) 1px, transparent 1px);
    background-size: 20px 20px;
}

h1, h2, h3, h4 {
    color: var(--primary) !important;
}

.stButton button {
    background-color: var(--accent) !important;
    color: var(--primary) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
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
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(10, 31, 68, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# Route to home page
st.switch_page("pages/1_Home.py")

