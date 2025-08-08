import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import datetime
import requests
from streamlit_autorefresh import st_autorefresh

st.title("Live Financial Dashboard")
st.markdown("Real-time market data and analytics")

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="dashboard_refresh")

# Sidebar for FRED API key
with st.sidebar:
    fred_api_key = st.text_input("FRED API Key", type="password")

# Market overview
st.header("Market Overview")
tabs = st.tabs(["Equities", "Forex", "Commodities", "Economic Indicators"])

with tabs[0]:
    st.subheader("Stock Market Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Enter Ticker", value="AAPL", max_chars=5).upper()
        
    with col2:
        period = st.selectbox(
            "Time Period", 
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"],
            index=5
        )
    
    # Get historical data
    data = yf.download(ticker, period=period, auto_adjust=False)
    if not data.empty:
        # Flatten multi-index columns
        data.columns = data.columns.droplevel(1)
        
        # Create plot data
        plot_data = data.reset_index()
        plot_data = plot_data.rename(columns={'Date': 'date', 'Close': 'close'})
        
        # Price chart
        fig = px.line(plot_data, x='date', y='close', title=f"{ticker} Price History")
        st.plotly_chart(fig, use_container_width=True)
        
        # Fetch latest price for real-time metric
        latest_info = yf.Ticker(ticker).info
        latest_price = latest_info.get('regularMarketPrice', data['Close'].iloc[-1])
        if len(data) > 1:
            delta = latest_price - data['Close'].iloc[-2]
            st.metric(
                label="Current Price", 
                value=f"${latest_price:.2f}", 
                delta=f"{delta:.2f} ({delta/data['Close'].iloc[-2]*100:.2f}%)"
            )
        else:
            st.metric(label="Current Price", value=f"${latest_price:.2f}")
    else:
        st.warning("No data available for selected ticker")

with tabs[1]:
    st.subheader("Forex Market Analysis")
    forex_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
    pair = st.selectbox("Select Forex Pair", forex_pairs)
    
    # Get historical data
    data = yf.download(pair, period="1mo", auto_adjust=False)
    if not data.empty:
        data.columns = data.columns.droplevel(1)
        plot_data = data.reset_index()
        plot_data = plot_data.rename(columns={'Date': 'date', 'Close': 'rate'})
        
        fig = px.line(plot_data, x='date', y='rate', title=f"{pair} Exchange Rate")
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest rate
        latest_info = yf.Ticker(pair).info
        latest_rate = latest_info.get('regularMarketPrice', data['Close'].iloc[-1])
        if len(data) > 1:
            delta = latest_rate - data['Close'].iloc[-2]
            st.metric(
                label="Current Rate", 
                value=f"{latest_rate:.4f}", 
                delta=f"{delta:.4f} ({delta/data['Close'].iloc[-2]*100:.2f}%)"
            )
        else:
            st.metric(label="Current Rate", value=f"{latest_rate:.4f}")
    else:
        st.warning("No data available for selected pair")

with tabs[2]:
    st.subheader("Commodities Market Analysis")
    commodities = ["GC=F", "SI=F", "CL=F", "NG=F", "ZC=F"]
    commodity = st.selectbox("Select Commodity", commodities, format_func=lambda x: {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil", "NG=F": "Natural Gas", "ZC=F": "Corn"}[x])
    
    # Get historical data
    data = yf.download(commodity, period="1mo", auto_adjust=False)
    if not data.empty:
        data.columns = data.columns.droplevel(1)
        plot_data = data.reset_index()
        plot_data = plot_data.rename(columns={'Date': 'date', 'Close': 'price'})
        
        fig = px.line(plot_data, x='date', y='price', title=f"{commodity} Price")
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest price
        latest_info = yf.Ticker(commodity).info
        latest_price = latest_info.get('regularMarketPrice', data['Close'].iloc[-1])
        if len(data) > 1:
            delta = latest_price - data['Close'].iloc[-2]
            st.metric(
                label="Current Price", 
                value=f"${latest_price:.2f}", 
                delta=f"{delta:.2f} ({delta/data['Close'].iloc[-2]*100:.2f}%)"
            )
        else:
            st.metric(label="Current Price", value=f"${latest_price:.2f}")
    else:
        st.warning("No data available for selected commodity")

with tabs[3]:
    st.subheader("Economic Indicators")
    econ_indicators = {
        "GDP": "Gross Domestic Product",
        "CPIAUCSL": "Consumer Price Index (CPI)",
        "UNRATE": "Unemployment Rate",
        "FEDFUNDS": "Federal Funds Rate",
        "MORTGAGE30US": "30-Year Mortgage Rate"
    }
    
    indicator = st.selectbox("Select Indicator", list(econ_indicators.keys()), 
                            format_func=lambda x: econ_indicators[x])
    
    if not fred_api_key:
        st.warning("Please enter FRED API key in sidebar to fetch real data.")
    else:
        try:
            # Fetch data from FRED API
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={indicator}&api_key={fred_api_key}&file_type=json&limit=100&sort_order=desc"
            response = requests.get(url)
            data = response.json()
            observations = data['observations']
            
            econ_data = pd.DataFrame({
                'Date': [obs['date'] for obs in observations],
                'Value': [float(obs['value']) if obs['value'] != '.' else np.nan for obs in observations]
            })
            econ_data['Date'] = pd.to_datetime(econ_data['Date'])
            econ_data = econ_data.dropna()
            
            fig = px.line(econ_data, x='Date', y='Value', 
                         title=f"{econ_indicators[indicator]} Trend")
            st.plotly_chart(fig, use_container_width=True)
            
            latest_value = econ_data['Value'].iloc[-1]
            latest_date = econ_data['Date'].iloc[-1].strftime("%Y-%m-%d")
            st.metric("Latest Value", f"{latest_value:.2f}", f"as of {latest_date}")
        except Exception as e:
            st.error(f"Error fetching FRED data: {str(e)}. Please check API key.")

st.markdown("---")
st.page_link("pages/1_Home.py", label="← Back to Home", icon="🏠")
