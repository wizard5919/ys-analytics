import streamlit as st

# Set page title for multi-page navigation
st.set_page_config(page_title="Insights")

st.title("Market Insights & Research")
st.markdown("""
## Coming Soon: YS Analytics Insights Hub

We're developing a comprehensive resource for financial market analysis, including:

- **Weekly Market Reports**: Sector performance and trend analysis
- **Economic Forecasts**: Predictive modeling of key indicators
- **Technical Analysis**: Chart pattern recognition and strategy backtesting
- **Quantitative Research**: Algorithmic trading insights

Sign up to be notified when we launch:
""")

# Email input
email = st.text_input("Your Email Address")
if st.button("Notify Me"):
    if "@" in email and "." in email:
        st.success("You'll be notified when we launch our insights hub!")
    else:
        st.warning("Please enter a valid email address")

st.markdown("---")

# Navigation back to Home page
st.page_link("Home", label="← Back to Home", icon="🏠")
