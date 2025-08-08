import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import datetime

st.title("Live Financial Dashboard")
st.markdown("Real-time market data and analytics")

# Market overview
st.header("Market Overview")
tabs = st.tabs(["Equities", "Forex", "Commodities", "Economic Indicators"])

with tabs[0]:
    st.subheader("Stock Market Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.selectbox(
            "Select Ticker", 
            ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"],
            index=0
        )
        
    with col2:
        period = st.selectbox(
            "Time Period", 
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"],
            index=5
        )
    
    # Get stock data
    data = yf.download(ticker, period=period, auto_adjust=False)  # Explicitly set to avoid warning; change to True if adjusted close is preferred
    if not data.empty:
        # Flatten multi-index columns (removes ticker level for single-symbol queries)
        data.columns = data.columns.droplevel(1)
        
        # Create a DataFrame that Plotly can handle
        plot_data = data.reset_index()
        plot_data = plot_data.rename(columns={'Date': 'date', 'Close': 'close'})
        
        # Price chart
        fig = px.line(plot_data, x='date', y='close', title=f"{ticker} Price History")
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        if len(data) > 1:
            delta = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            st.metric(
                label="Current Price", 
                value=f"${data['Close'].iloc[-1]:.2f}", 
                delta=f"{delta:.2f} ({delta/data['Close'].iloc[-2]*100:.2f}%)"
            )
        else:
            st.metric(label="Current Price", value=f"${data['Close'].iloc[-1]:.2f}")
    else:
        st.warning("No data available for selected ticker")

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
    
    # In a real implementation, you would fetch from FRED API
    st.info("Economic indicators integration requires FRED API key. This is a placeholder.")
    
    # Sample data
    dates = pd.date_range(end=datetime.datetime.today(), periods=12, freq='M')
    values = [100 + i*2 + (i-6)**2 for i in range(12)]
    econ_data = pd.DataFrame({'Date': dates, 'Value': values})
    
    fig = px.line(econ_data, x='Date', y='Value', 
                 title=f"{econ_indicators[indicator]} Trend")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.page_link("pages/1_Home.py", label="← Back to Home", icon="🏠")
