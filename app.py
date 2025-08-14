import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="YS Analytics | Data-Driven Market Intelligence",
    page_icon="assets/logo.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "### YS Analytics\nFinancial Analytics Platform\n"
    }
)

# Global CSS - minimal version for now
st.markdown("""
<style>
:root {
    --primary: #0A1F44;
    --accent: #00C2FF;
    --light: #F5F9FC;
}

body {
    color: var(--primary);
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Options Analyzer"])

if page == "Home":
    st.switch_page("pages/1_Home.py")
elif page == "Options Analyzer":
    st.switch_page("pages/6_Options_Analyzer.py")
