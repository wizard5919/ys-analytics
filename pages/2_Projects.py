import streamlit as st
from streamlit.components.v1 import html

# Create anchor targets
st.markdown("""
<style>
.anchor-target {
    position: relative;
    top: -100px;  /* Offset for fixed header */
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Navigation handler
def scroll_to_project(project_id):
    js = f"""
    <script>
        var target = document.querySelector("#{project_id}");
        if (target) {{
            window.scrollTo({{
                top: target.offsetTop - 100,
                behavior: 'smooth'
            }});
        }}
    </script>
    """
    html(js)

# Check if we need to scroll to a specific project
if "navigate_to" in st.session_state:
    project_id = st.session_state["navigate_to"]
    del st.session_state["navigate_to"]
    scroll_to_project(project_id)

# Main content
st.title("Project Portfolio")
st.markdown("""
Explore our financial analytics projects demonstrating expertise in market analysis, 
predictive modeling, and data visualization.
""")

# Project 1 - Options Analytics Suite
st.markdown('<div id="options" class="anchor-target"></div>', unsafe_allow_html=True)
with st.expander("Options Analytics Suite", expanded=True):
    st.markdown("""
    <div class="card">
        <h3>Advanced Options Pricing & Risk Analysis</h3>
        <p><strong>Challenge:</strong> Traders need real-time calculation of options Greeks and volatility surfaces to assess risk.</p>
        <p><strong>Solution:</strong> Streamlit-based application that calculates and visualizes:</p>
        <ul>
            <li>Real-time Greeks (Delta, Gamma, Theta, Vega, Rho)</li>
            <li>Implied volatility surfaces</li>
            <li>Profit/loss probability distributions</li>
            <li>Strategy backtesting capabilities</li>
        </ul>
        <p><strong>Technologies:</strong> Python, QuantLib, Streamlit, Plotly</p>
        <div style="display: flex; gap: 10px; margin-top: 20px;">
            <a href="pages/3_Dashboard.py" class="btn">View Live Demo</a>
            <a href="https://github.com/wizard5919" class="btn">GitHub Repository</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
st.markdown('<div id="sector" class="anchor-target"></div>', unsafe_allow_html=True)
with st.expander("Market Sector Classifier"):
    st.markdown("""
    <div class="card">
        <h3>Machine Learning-Based Sector Classification</h3>
        <p><strong>Challenge:</strong> Automatically classify stocks into sectors based on price movement patterns.</p>
        <p><strong>Solution:</strong> ML pipeline that:</p>
        <ul>
            <li>Processes historical price data from global markets</li>
            <li>Extracts technical indicators as features</li>
            <li>Trains Random Forest classifier for sector prediction</li>
            <li>Provides visual analytics of sector rotation patterns</li>
        </ul>
        <p><strong>Technologies:</strong> Python, Scikit-learn, TA-Lib, Plotly</p>
        <div style="display: flex; gap: 10px; margin-top: 20px;">
            <a href="pages/3_Dashboard.py" class="btn">View Case Study</a>
            <a href="#" class="btn">Research Paper</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
st.markdown('<div id="macro" class="anchor-target"></div>', unsafe_allow_html=True)
with st.expander("Macroeconomic Dashboard"):
    st.markdown("""
    <div class="card">
        <h3>Global Economic Indicators Tracker</h3>
        <p><strong>Challenge:</strong> Investors need consolidated view of macroeconomic trends across countries.</p>
        <p><strong>Solution:</strong> Interactive dashboard that:</p>
        <ul>
            <li>Aggregates data from FRED, World Bank, and IMF</li>
            <li>Provides time-series analysis of key indicators (GDP, CPI, unemployment)</li>
            <li>Includes Prophet-based forecasting models</li>
            <li>Compares economic performance across regions</li>
        </ul>
        <p><strong>Technologies:</strong> Python, FRED API, Prophet, Altair</p>
        <div style="display: flex; gap: 10px; margin-top: 20px;">
            <a href="pages/3_Dashboard.py" class="btn">Explore Dashboard</a>
            <a href="#" class="btn">Methodology</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation back to home
st.markdown("---")
st.page_link("pages/1_Home.py", label="← Back to Home", icon="🏠")

# Button styles
st.markdown("""
<style>
.btn {
    background-color: #00C2FF;
    color: #0A1F44;
    border: none;
    padding: 8px 16px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 14px;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.btn:hover {
    background-color: #0A1F44;
    color: #00C2FF;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 194, 255, 0.3);
}
</style>
""", unsafe_allow_html=True)
