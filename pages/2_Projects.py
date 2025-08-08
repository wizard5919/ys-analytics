import streamlit as st

st.title("Project Portfolio")
st.markdown("""
Explore our financial analytics projects demonstrating expertise in market analysis, 
predictive modeling, and data visualization.
""")

# Project 1
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
        </ul>
        <p><strong>Technologies:</strong> Python, QuantLib, Streamlit, Plotly</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/6_Options_Analyzer.py", label="View Live Demo")
    with col2:
        st.link_button("GitHub Repository", "https://github.com/wizard5919/ys-analytics/tree/main/pages/6_Options_Analyzer.py")

# Project 2
with st.expander("Market Sector Classifier"):
    st.markdown("""
    <div class="card">
        <h3>Machine Learning-Based Sector Classification</h3>
        <p><strong>Challenge:</strong> Automatically classify stocks into sectors based on price movement patterns.</p>
        <p><strong>Solution:</strong> ML pipeline that:</p>
        <ul>
            <li>Processes historical price data</li>
            <li>Extracts technical indicators as features</li>
            <li>Trains Random Forest classifier for sector prediction</li>
        </ul>
        <p><strong>Technologies:</strong> Python, Scikit-learn, TA-Lib, Plotly</p>
        <div style="display: flex; gap: 10px;">
            <button style="flex: 1;">View Case Study</button>
            <button style="flex: 1;">Research Paper</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Project 3
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
        </ul>
        <p><strong>Technologies:</strong> Python, FRED API, Prophet, Altair</p>
        <div style="display: flex; gap: 10px;">
            <button style="flex: 1;">Explore Dashboard</button>
            <button style="flex: 1;">Methodology</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.page_link("pages/1_Home.py", label="← Back to Home", icon="🏠")
