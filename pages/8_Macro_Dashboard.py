import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
import plotly.express as px
from datetime import datetime
import numpy as np

st.title("Macroeconomic Dashboard Demo")
st.markdown("Interactive dashboard that aggregates data from FRED, provides time-series analysis of key indicators (GDP, CPI, unemployment), includes Prophet-based forecasting models, and compares economic performance across regions.")

# Sidebar for FRED API key
fred_key = st.sidebar.text_input("FRED API Key", type="password")

indicators = {
    "GDP": "Gross Domestic Product (US)",
    "CPALTT01EZM659G": "CPI (EU)",
    "CNGDPYOY": "GDP YoY (China)",
    "CPIAUCSL": "Consumer Price Index (CPI) (US)",
    "UNRATE": "Unemployment Rate (US)"
}

indicator = st.selectbox("Select Indicator", list(indicators.keys()), format_func=lambda x: indicators[x])

if fred_key:
    try:
        # Fetch FRED data
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={indicator}&api_key={fred_key}&file_type=json&sort_order=desc"
        response = requests.get(url)
        data = response.json()['observations']
        
        df = pd.DataFrame({
            'Date': [obs['date'] for obs in data],
            'Value': [float(obs['value']) if obs['value'] != '.' else np.nan for obs in data]
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna().sort_values('Date')
        
        # Plot time-series
        fig = px.line(df, x='Date', y='Value', title=f"{indicators[indicator]} Time Series")
        st.plotly_chart(fig)
        
        latest = df.iloc[-1]
        st.metric("Latest Value", latest['Value'], f"as of {latest['Date'].strftime('%Y-%m-%d')}")
        
        # Prophet forecast
        if st.button("Generate Forecast"):
            prophet_df = df.rename(columns={'Date': 'ds', 'Value': 'y'})
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)
            
            fig_forecast = px.line(forecast, x='ds', y='yhat', title="Prophet Forecast")
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Bound")
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Bound")
            st.plotly_chart(fig_forecast)
        
        # Comparison across regions
        st.subheader("Comparison Across Regions")
        region_data = pd.DataFrame()
        for ind in ["GDP", "CPALTT01EZM659G", "CNGDPYOY"]:
            url_region = f"https://api.stlouisfed.org/fred/series/observations?series_id={ind}&api_key={fred_key}&file_type=json&sort_order=desc&limit=12"
            resp = requests.get(url_region)
            dat = resp.json()['observations']
            region_df = pd.DataFrame({
                'Date': [obs['date'] for obs in dat],
                'Value': [float(obs['value']) if obs['value'] != '.' else np.nan for obs in dat],
                'Region': ind
            })
            region_data = pd.concat([region_data, region_df])
        
        fig_compare = px.line(region_data, x='Date', y='Value', color='Region', title="Economic Performance Comparison")
        st.plotly_chart(fig_compare)
    else:
        st.warning("Please enter FRED API key.")
else:
    st.warning("Enter FRED API key to fetch data.")

st.markdown("---")
st.page_link("pages/2_Projects.py", label="← Back to Projects", icon="📚")
