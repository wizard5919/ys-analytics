import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def calculate_option_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

def delta(S, K, T, r, sigma, option_type="call"):
    d_1 = d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d_1)
    else:
        return norm.cdf(d_1) - 1

def gamma(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type="call"):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    term1 = - (S * sigma * norm.pdf(d_1)) / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = - r * K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d_2)
    return term1 + term2

def vega(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d_1)

def rho(S, K, T, r, sigma, option_type="call"):
    d_2 = d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d_2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d_2)

def calculate_implied_vol(market_price, S, K, T, r, option_type="call"):
    def objective(sigma):
        return calculate_option_price(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(objective, 1e-6, 5.0)
    except ValueError:
        return np.nan

st.title("Options Analytics Suite")
st.markdown("Real-time options pricing, risk analysis, and visualizations for informed decision-making.")

tabs = st.tabs(["Greeks Calculator", "Volatility Surface", "P/L Distribution"])

with tabs[0]:
    st.subheader("Greeks Calculator")
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Spot Price (S)", value=100.0, min_value=0.01)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01)
        T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
    with col2:
        r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0)
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01)
        option_type = st.selectbox("Option Type", ["call", "put"])

    if st.button("Calculate Greeks"):
        price = calculate_option_price(S, K, T, r, sigma, option_type)
        delta_val = delta(S, K, T, r, sigma, option_type)
        gamma_val = gamma(S, K, T, r, sigma)
        theta_val = theta(S, K, T, r, sigma, option_type)
        vega_val = vega(S, K, T, r, sigma)
        rho_val = rho(S, K, T, r, sigma, option_type)

        st.metric("Option Price", f"{price:.2f}")
        cols = st.columns(5)
        cols[0].metric("Delta", f"{delta_val:.4f}")
        cols[1].metric("Gamma", f"{gamma_val:.4f}")
        cols[2].metric("Theta", f"{theta_val:.4f}")
        cols[3].metric("Vega", f"{vega_val:.4f}")
        cols[4].metric("Rho", f"{rho_val:.4f}")

with tabs[1]:
    st.subheader("Volatility Surface")
    ticker = st.text_input("Ticker Symbol", "AAPL")
    if ticker:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        expiry = st.selectbox("Select Expiry Date", expiries)
        if expiry:
            chain = stock.option_chain(expiry)
            calls = chain.calls.dropna(subset=['impliedVolatility'])
            puts = chain.puts.dropna(subset=['impliedVolatility'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=calls['strike'], y=calls['impliedVolatility'], mode='lines+markers', name='Calls'))
            fig.add_trace(go.Scatter(x=puts['strike'], y=puts['impliedVolatility'], mode='lines+markers', name='Puts'))
            fig.update_layout(title=f"Implied Volatility Smile for {ticker} (Expiry: {expiry})",
                              xaxis_title="Strike Price",
                              yaxis_title="Implied Volatility",
                              hovermode="x")
            st.plotly_chart(fig)

with tabs[2]:
    st.subheader("Profit/Loss Probability Distribution")
    S = st.number_input("Current Spot Price", value=100.0, min_value=0.01, key="pl_s")
    K = st.number_input("Strike Price", value=100.0, min_value=0.01, key="pl_k")
    T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, key="pl_t")
    r = st.number_input("Risk-free Rate", value=0.05, key="pl_r")
    sigma = st.number_input("Volatility", value=0.2, min_value=0.01, key="pl_sigma")
    option_type = st.selectbox("Option Type", ["call", "put"], key="pl_type")
    num_sim = st.number_input("Number of Simulations", value=10000, min_value=1000)

    if st.button("Simulate P/L"):
        # Monte Carlo simulation for stock price at expiry
        Z = np.random.standard_normal(num_sim)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        pl = payoffs * np.exp(-r * T)  # Discounted

        prob_profit = np.mean(pl > 0) * 100
        avg_pl = np.mean(pl)

        st.metric("Probability of Profit", f"{prob_profit:.2f}%")
        st.metric("Average P/L", f"{avg_pl:.2f}")

        fig = go.Figure(data=[go.Histogram(x=pl, nbinsx=50)])
        fig.update_layout(title="P/L Distribution", xaxis_title="P/L", yaxis_title="Frequency")
        st.plotly_chart(fig)

st.markdown("---")
st.page_link("pages/2_Projects.py", label="← Back to Projects", icon="📚")
