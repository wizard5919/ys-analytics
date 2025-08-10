import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Market Sector Classifier Demo")
st.markdown("ML pipeline that processes historical price data from global markets, extracts technical indicators as features, trains Random Forest classifier for sector prediction, and provides visual analytics of sector rotation patterns.")

# Sample stocks and sectors
stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'WFC', 'XOM', 'CVX', 'SLB']
sectors = ['Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Finance', 'Energy', 'Energy', 'Energy']

# Fetch data
data = []
for stock in stocks:
    df = yf.download(stock, period="1y", progress=False)
    if df.empty:
        continue  # Skip if no data
    df['Return'] = df['Close'].pct_change()
    features = {
        'stock': stock,
        'mean_return': df['Return'].mean(),
        'std_return': df['Return'].std(),
        'volume_mean': df['Volume'].mean()
    }
    data.append(features)

df_features = pd.DataFrame(data)
df_features['sector'] = sectors

# Convert to numeric, handle NaN
df_features['mean_return'] = pd.to_numeric(df_features['mean_return'], errors='coerce')
df_features['std_return'] = pd.to_numeric(df_features['std_return'], errors='coerce')
df_features['volume_mean'] = pd.to_numeric(df_features['volume_mean'], errors='coerce')

# Drop rows with NaN
df_features = df_features.dropna()

# Train model
X = df_features[['mean_return', 'std_return', 'volume_mean']]
y = df_features['sector']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
st.metric("Model Accuracy", f"{acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig_cm)

# Predict for a new stock
new_ticker = st.text_input("Enter Stock to Predict Sector", "AMZN")
if new_ticker:
    new_df = yf.download(new_ticker, period="1y", progress=False)
    new_df['Return'] = new_df['Close'].pct_change()
    new_features = {
        'mean_return': new_df['Return'].mean(),
        'std_return': new_df['Return'].std(),
        'volume_mean': new_df['Volume'].mean()
    }
    new_X = pd.DataFrame([new_features])
    new_X = new_X.apply(pd.to_numeric, errors='coerce')
    pred_sector = clf.predict(new_X)[0]
    st.success(f"Predicted sector for {new_ticker}: {pred_sector}")

# Visual analytics (simple bar chart of sector returns)
st.subheader("Sector Rotation Analytics")
sector_returns = df_features.groupby('sector')['mean_return'].mean()
fig = px.bar(sector_returns, x=sector_returns.index, y=sector_returns.values, title="Average Returns by Sector")
st.plotly_chart(fig)

# Additional visualization: Sector Volume
sector_volume = df_features.groupby('sector')['volume_mean'].mean()
fig_vol = px.bar(sector_volume, x=sector_volume.index, y=sector_volume.values, title="Average Volume by Sector")
st.plotly_chart(fig_vol)

# Additional visualization: Sector Std Returns
sector_std = df_features.groupby('sector')['std_return'].mean()
fig_std = px.bar(sector_std, x=sector_std.index, y=sector_std.values, title="Average Std Returns by Sector")
st.plotly_chart(fig_std)

st.markdown("---")
st.page_link("pages/2_Projects.py", label="← Back to Projects", icon="📚")
