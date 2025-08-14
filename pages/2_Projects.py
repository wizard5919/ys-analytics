import streamlit as st

st.title("Project Portfolio")
st.markdown("""
Explore our financial analytics projects demonstrating expertise in market analysis, 
predictive modeling, and data visualization.
""")

# ---------------- Project 1 ----------------
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
    </div>
    """, unsafe_allow_html=True)

    # Internal page link
    st.page_link("Options Analyzer", label="View Live Demo", icon="📂", use_container_width=True)
    # External GitHub link
    st.markdown("[GitHub Repository](https://github.com/wizard5919/options_analyzerPublic)")

# ---------------- Project 2 ----------------
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
    </div>
    """, unsafe_allow_html=True)

    # Internal page link
    st.page_link("Sector Classifier", label="View Case Study", icon="📂", use_container_width=True)
    # External research paper
    st.markdown("[Research Paper](https://example.com/research-paper.pdf)")

# ---------------- Project 3 ----------------
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
    </div>
    """, unsafe_allow_html=True)

    # Internal page links
    st.page_link("Macro Dashboard", label="Explore Dashboard", icon="📂", use_container_width=True)
    st.page_link("Methodology", label="Methodology", icon="📄", use_container_width=True)

# Back to Home
st.markdown("---")
st.page_link("Home", label="← Back to Home", icon="🏠")
