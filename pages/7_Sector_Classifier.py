import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import traceback

st.title("Market Sector Classifier Demo")
st.markdown("ML pipeline that processes historical price data from global markets, extracts technical indicators as features, trains Random Forest classifier for sector prediction, and provides visual analytics of sector rotation patterns.")

# Expanded stock list with more sectors and companies
stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',  # Tech
    'JPM', 'BAC', 'WFC', 'GS', 'C',           # Finance
    'XOM', 'CVX', 'SLB', 'COP', 'EOG',        # Energy
    'PG', 'KO', 'PEP', 'WMT', 'COST',         # Consumer Goods
    'UNH', 'JNJ', 'PFE', 'MRK', 'ABT'         # Healthcare
]
sectors = [
    'Tech', 'Tech', 'Tech', 'Tech', 'Tech',
    'Finance', 'Finance', 'Finance', 'Finance', 'Finance',
    'Energy', 'Energy', 'Energy', 'Energy', 'Energy',
    'Consumer', 'Consumer', 'Consumer', 'Consumer', 'Consumer',
    'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare'
]

# Create stock-sector mapping
stock_sector_map = {stock: sector for stock, sector in zip(stocks, sectors)}

# Fetch data with progress and error handling
data = []
with st.spinner("Fetching stock data..."):
    for i, stock in enumerate(stocks):
        try:
            # Fetch data with 1 year lookback
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            df = yf.download(stock, start=start_date, end=end_date, progress=False)
            
            # Properly check if DataFrame is empty
            if df is None or df.empty:
                st.warning(f"No data available for {stock}, skipping...")
                continue
                
            # Check length properly
            if len(df) < 10:
                st.warning(f"Insufficient data points for {stock} (only {len(df)} rows), skipping...")
                continue
                
            # Calculate features
            df['Return'] = df['Close'].pct_change()
            returns = df['Return'].dropna()
            
            # Check for valid returns
            if returns.empty:
                st.warning(f"No valid returns for {stock}, skipping...")
                continue
                
            # Calculate features and ensure they are scalars
            mean_return = returns.mean()
            std_return = returns.std()
            volume_mean = df['Volume'].mean()
            
            # Convert to scalars if needed
            if isinstance(mean_return, pd.Series):
                mean_return = mean_return.iloc[0]
            if isinstance(std_return, pd.Series):
                std_return = std_return.iloc[0]
            if isinstance(volume_mean, pd.Series):
                volume_mean = volume_mean.iloc[0]
            
            # Skip if features are NaN
            if (pd.isna(mean_return) or pd.isna(std_return) or pd.isna(volume_mean)):
                st.warning(f"Invalid features for {stock}, skipping...")
                continue
                
            features = {
                'stock': stock,
                'mean_return': mean_return,
                'std_return': std_return,
                'volume_mean': volume_mean
            }
            features['sector'] = stock_sector_map[stock]
            data.append(features)
            
            st.success(f"Processed {stock} successfully")
            
        except Exception as e:
            st.error(f"Error processing {stock}: {str(e)}")
            st.text(traceback.format_exc())  # Show full traceback for debugging

if not data:
    st.error("Failed to fetch data for any stocks. Please check your internet connection or try again later.")
    st.stop()

df_features = pd.DataFrame(data)

# Show the data we have
st.subheader("Training Data")
st.dataframe(df_features[['stock', 'sector', 'mean_return', 'std_return', 'volume_mean']])

# Only proceed if we have enough data
if len(df_features) < 2:
    st.error(f"Insufficient data to train model. Only {len(df_features)} valid samples available. Need at least 2.")
    st.stop()

X = df_features[['mean_return', 'std_return', 'volume_mean']]
y = df_features['sector']

# Handle small datasets by adjusting test size
test_size = 0.2 if len(df_features) > 5 else 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)

# Evaluate model
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
st.metric("Model Accuracy", f"{acc*100:.2f}%")

# Feature importance
st.subheader("Feature Importance")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)
fig_imp = px.bar(importance, x='Feature', y='Importance', title="Random Forest Feature Importance")
st.plotly_chart(fig_imp)

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, pred)
fig_cm, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig_cm)

# Predict for a new stock
st.subheader("Predict Sector for New Stock")
new_ticker = st.text_input("Enter Stock Ticker", "TSLA").upper()
if new_ticker:
    try:
        with st.spinner(f"Fetching data for {new_ticker}..."):
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            new_df = yf.download(new_ticker, start=start_date, end=end_date, progress=False)
            
            # Properly check if DataFrame is empty
            if new_df is None or new_df.empty:
                st.warning(f"No data available for {new_ticker}")
            elif len(new_df) < 10:
                st.warning(f"Insufficient data points for {new_ticker} (only {len(new_df)} rows)")
            else:
                # Calculate features
                new_df['Return'] = new_df['Close'].pct_change()
                returns = new_df['Return'].dropna()
                
                if returns.empty:
                    st.warning(f"No valid returns for {new_ticker}")
                else:
                    # Calculate features and ensure they are scalars
                    mean_return = returns.mean()
                    std_return = returns.std()
                    volume_mean = new_df['Volume'].mean()
                    
                    # Convert to scalars if needed
                    if isinstance(mean_return, pd.Series):
                        mean_return = mean_return.iloc[0]
                    if isinstance(std_return, pd.Series):
                        std_return = std_return.iloc[0]
                    if isinstance(volume_mean, pd.Series):
                        volume_mean = volume_mean.iloc[0]
                    
                    if (pd.isna(mean_return) or pd.isna(std_return) or pd.isna(volume_mean)):
                        st.warning(f"Invalid features for {new_ticker}")
                    else:
                        new_features = {
                            'mean_return': mean_return,
                            'std_return': std_return,
                            'volume_mean': volume_mean
                        }
                        new_X = pd.DataFrame([new_features])
                        
                        # Predict and display
                        pred_sector = clf.predict(new_X)[0]
                        st.success(f"Predicted sector for {new_ticker}: **{pred_sector}**")
                        
                        # Show feature values
                        st.write("Feature Values:")
                        st.dataframe(new_X)
    except Exception as e:
        st.error(f"Error processing {new_ticker}: {str(e)}")

# Visual analytics
st.subheader("Sector Rotation Analytics")

# Sector returns
sector_returns = df_features.groupby('sector')['mean_return'].mean().sort_values()
fig = px.bar(sector_returns, x=sector_returns.index, y=sector_returns.values, 
             title="Average Daily Returns by Sector", color=sector_returns.values,
             color_continuous_scale='RdYlGn')
st.plotly_chart(fig)

# Sector volume
sector_volume = df_features.groupby('sector')['volume_mean'].mean().sort_values()
fig_vol = px.bar(sector_volume, x=sector_volume.index, y=sector_volume.values, 
                 title="Average Daily Volume by Sector", color=sector_volume.index)
st.plotly_chart(fig_vol)

# Sector volatility
sector_std = df_features.groupby('sector')['std_return'].mean().sort_values()
fig_std = px.bar(sector_std, x=sector_std.index, y=sector_std.values, 
                 title="Average Daily Volatility by Sector", color=sector_std.values,
                 color_continuous_scale='RdYlGn_r')
st.plotly_chart(fig_std)

# 3D Scatter plot
st.subheader("Sector Feature Space")
fig_3d = px.scatter_3d(
    df_features, 
    x='mean_return', 
    y='std_return', 
    z='volume_mean',
    color='sector',
    symbol='sector',
    hover_name='stock',
    title="3D Sector Visualization"
)
st.plotly_chart(fig_3d)

st.markdown("---")
st.page_link("pages/2_Projects.py", label="← Back to Projects", icon="📚")
