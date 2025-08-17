import streamlit as st

st.title("Project Portfolio")
st.markdown("""
Explore our financial analytics projects demonstrating expertise in market analysis, 
predictive modeling, and data visualization.
""")

# Project 1 - Options Analytics Suite
with st.expander("Options Analytics Suite"):
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
        <div style="display: flex; gap: 10px;">
            <a href="https://ys-analytics-4yaygtnc8ae6ryootpxhxs.streamlit.app/Options_Analyzer" target="_blank" style="flex: 1; background-color: #00C2FF; color: #0A1F44; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; text-decoration: none;">View Live Demo</a>
            <a href="https://github.com/wizard5919/options_analyzerPublic" target="_blank" style="flex: 1; background-color: #00C2FF; color: #0A1F44; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; text-decoration: none;">GitHub Repository</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Project 2 - Market Sector Classifier
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
        <div style="display: flex; gap: 10px;">
            <a href="https://ys-analytics-4yaygtnc8ae6ryootpxhxs.streamlit.app/Sector_Classifier" target="_blank" style="flex: 1; background-color: #00C2FF; color: #0A1F44; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; text-decoration: none;">View Case Study</a>
            <a href="https://example.com/research-paper.pdf" target="_blank" style="flex: 1; background-color: #00C2FF; color: #0A1F44; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; text-decoration: none;">Research Paper</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Project 3 - Macroeconomic Dashboard
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
        <div style="display: flex; gap: 10px;">
            <a href="https://ys-analytics-4yaygtnc8ae6ryootpxhxs.streamlit.app/Macro_Dashboard" target="_blank" style="flex: 1; background-color: #00C2FF; color: #0A1F44; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; text-decoration: none;">Explore Dashboard</a>
            <a href="pages/9_Methodology.py" style="flex: 1; background-color: #00C2FF; color: #0A1F44; border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; text-decoration: none;">Methodology</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.page_link("app.py", label="← Back to Home", icon="🏠")
