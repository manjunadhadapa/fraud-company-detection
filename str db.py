import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configure the page
st.set_page_config(page_title="Entity Resolution AI", page_icon="🕵️‍♂️", layout="centered")

# Load the saved model and scaler
@st.cache_resource
def load_components():
    with open('best_fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_components()

# UI Header
st.title("🏢 Fraudulent Shell Company Detector")
st.markdown("### B2B Entity Resolution Challenge")
st.write("Enter the extracted numerical features of the newly registered business below to assess fraud risk.")

# Create input fields in two columns for a clean UI
col1, col2 = st.columns(2)

with col1:
    ip_distance = st.number_input("IP Distance Metric (miles)", min_value=0.0, max_value=5000.0, value=15.0)
    shared_directors = st.number_input("Shared Director Count (Banned Entities)", min_value=0, max_value=20, value=0)
    industry_overlap = st.slider("Industry Code Overlap Score", 0.0, 1.0, 0.1)

with col2:
    transactions = st.number_input("Transaction Frequency (per week)", min_value=0.0, max_value=1000.0, value=50.0)
    days_registered = st.number_input("Days Since Registration", min_value=1, max_value=10000, value=1000)

# Prediction Button
if st.button("Run Entity Resolution Analysis", type="primary"):
    
    # Structure the input data exactly how the model expects it
    input_data = pd.DataFrame({
        'ip_distance_metric': [ip_distance],
        'shared_director_count': [shared_directors],
        'transaction_frequency': [transactions],
        'days_since_registration': [days_registered],
        'industry_code_overlap': [industry_overlap]
    })
    
    # Scale the inputs
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100
    
    # Display Results
    st.divider()
    if prediction == 1:
        st.error(f"🚨 **HIGH RISK ALERT: Fraudulent Shell Clone Detected!**")
        st.write(f"The model calculates a **{probability:.1f}%** probability that this is a banned entity operating under a new name.")
    else:
        st.success(f"✅ **LOW RISK: Legitimate Business Verified.**")
        st.write(f"The model calculates only a **{probability:.1f}%** probability of fraud. Entity resolved safely.")