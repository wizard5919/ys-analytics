import streamlit as st
import pandas as pd

st.title("Contact YS Analytics")
st.markdown("Get in touch for consulting, collaboration, or inquiries")

# Contact form
with st.form("contact_form"):
    st.subheader("Send us a message")
    name = st.text_input("Name")
    email = st.text_input("Email")
    subject = st.selectbox("Subject", [
        "Consulting Inquiry", 
        "Project Collaboration",
        "Data Analysis Request",
        "Other"
    ])
    message = st.text_area("Message", height=150)
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        if name and email and message:
            st.success("Message sent successfully! We'll respond within 24 hours.")
            # In real implementation: Send email or save to database
        else:
            st.error("Please fill in all required fields")

# Contact info
st.markdown("---")
st.subheader("Connect With Us")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Email:** contact@ysanalytics.com  
    **Phone:** +1 (404) 561-9812  
    **Location:** ATLANTA, GA
    """)

with col2:
    st.markdown("""
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com)  
    [![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/wizard5919)
    """)

# Map placeholder
st.markdown("---")
st.subheader("Our Location")
st.map(pd.DataFrame({'lat': [33.7537], 'lon': [-84.3863]}), zoom=12)

st.markdown("---")
st.page_link("pages/1_Home.py", label="← Back to Home", icon="🏠")
