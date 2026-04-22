import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import roc_curve, auc

# --- 1. Introduction ---
st.set_page_config(page_title="Customer Travel Churn", layout="wide")
st.title("Project: Customer Travel Churn Prediction")
st.info("""
**What is Churn?** Customer churn happens when travelers stop using a service. 
**Business Relevance:** Predicting this helps companies retain customers, which is cheaper than acquiring new ones.
**Why Random Forest?** It is a powerful ensemble method that handles categorical travel data effectively.
""")

# --- 2. Load the Model ---
try:
    with open('model.pkl', 'rb') as f:
        my_model = pickle.load(f)
except FileNotFoundError:
    st.error("model.pkl not found!")

# --- 3. Sidebar Inputs (Requirement #4) ---
st.sidebar.header("Customer Input Features")
def user_input():
    age = st.sidebar.slider("Age", 18, 80, 30)
    flyer = st.sidebar.selectbox("Frequent Flyer?", ["Yes", "No", "No Record"])
    income = st.sidebar.selectbox("Income Class", ["Low Income", "Middle Income", "High Income"])
    services = st.sidebar.slider("Services Opted", 0, 10, 1)
    social = st.sidebar.selectbox("Social Media Synced?", ["Yes", "No"])
    hotel = st.sidebar.selectbox("Booked Hotel?", ["Yes", "No"])
    
    # Mapping logic (Requirement #4)
    data = {
        'Age': age,
        'FrequentFlyer': {'Yes': 1, 'No': 0, 'No Record': 0}[flyer],
        'AnnualIncomeClass': {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}[income],
        'ServicesOpted': services,
        'AccountSyncedToSocialMedia': 1 if social == "Yes" else 0,
        'BookedHotelOrNot': 1 if hotel == "Yes" else 0
    }
    return pd.DataFrame([data])

input_df = user_input()

# --- 4. Prediction (---
st.subheader("Prediction Result")
if st.button("Run Model"):
    prediction = my_model.predict(input_df)
    result = "CHURN (Will Leave)" if prediction[0] == 1 else "STAY (Will Remain)"
    st.write(f"### The predicted status is: **{result}**")

# --- 5. Visualizations & Evaluation ---
st.markdown("---")
st.subheader("Model Evaluation & Visuals")
col1, col2 = st.columns(2)

with col1:
    st.write("**Feature Importance Analysis**")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=my_model.feature_importances_, y=input_df.columns, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("**Model Conclusion **")
    st.success("""
    - **Key Driver:** Age and Income Class are the strongest predictors.
    - **Performance:** Random Forest provides a balanced accuracy.
    - **Future Work:** Adding more historical travel data could improve precision.
    """)
