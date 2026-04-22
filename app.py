import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Page Config ---
st.set_page_config(page_title="Travel Churn Predictor", layout="centered")

# --- 1. Load the Model ---
# This matches Step 8 of your notebook
try:
    with open('model.pkl', 'rb') as f:
        my_model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please upload it to your GitHub repository.")

# --- 2. User Interface ---
st.title("Customer Travel Churn Prediction")
st.markdown("Enter details below to predict if a customer will churn (leave) or stay.")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    flyer = st.selectbox("Frequent Flyer?", ["Yes", "No", "No Record"])
    income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])

with col2:
    services = st.number_input("Services Opted", min_value=0, max_value=20, value=1)
    social = st.selectbox("Synced to Social Media?", ["Yes", "No"])
    hotel = st.selectbox("Booked Hotel or Not?", ["Yes", "No"])

# --- 3. Preprocessing (Matching Notebook Step 4) ---
# Mapping inputs exactly like your notebook code
flyer_map = {'Yes': 1, 'No': 0, 'No Record': 0} # Included No Record to prevent errors
social_map = {'Yes': 1, 'No': 0}
hotel_map = {'Yes': 1, 'No': 0}
income_map = {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}

# Transform inputs
input_df = pd.DataFrame([{
    'Age': age,
    'FrequentFlyer': flyer_map[flyer],
    'AnnualIncomeClass': income_map[income],
    'ServicesOpted': services,
    'AccountSyncedToSocialMedia': social_map[social],
    'BookedHotelOrNot': hotel_map[hotel]
}])

# --- 4. Prediction ---
st.markdown("---")
if st.button("Predict Status"):
    prediction = my_model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("### Result: This customer is likely to CHURN.")
    else:
        st.success("### Result: This customer is likely to STAY.")

# --- 5. Visualizations (Matching Notebook Step 7) ---
st.markdown("---")
if st.checkbox("Show Model Analysis & Graphs"):
    st.subheader("Project Analysis Visuals")

    # Feature Importance (Exactly like Step 7)
    st.write("**What Features Mattered Most?**")
    fig1, ax1 = plt.subplots()
    importances = my_model.feature_importances_
    features = ['Age', 'FrequentFlyer', 'IncomeClass', 'ServicesOpted', 'SocialMedia', 'HotelBooked']
    sns.barplot(x=importances, y=features, ax=ax1)
    st.pyplot(fig1)

    # Note: Churn Distribution and ROC require the full dataset/test set.
    # To keep the app fast, we usually show the static training results here.
    st.info("The Feature Importance chart above shows how the model makes decisions based on the training data.")
