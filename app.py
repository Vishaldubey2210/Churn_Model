import streamlit as st
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("model/churn_model.pkl")

# ===============================
# TITLE
# ===============================
st.title("📉 Customer Churn Prediction App")
st.markdown("""
**Internship Project – IBM Telco Customer Churn**

This app predicts whether a customer is likely to churn
based on customer behavior and billing details.
""")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("🧾 Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
married = st.sidebar.selectbox("Married", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

internet = st.sidebar.selectbox(
    "Internet Service", ["No", "DSL", "Fiber Optic"]
)

contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

monthly_charge = st.sidebar.number_input(
    "Monthly Charge", min_value=0.0, value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges", min_value=0.0, value=1500.0
)

satisfaction = st.sidebar.slider(
    "Satisfaction Score", 1, 5, 3
)

# ===============================
# PREPROCESS INPUT
# ===============================
def preprocess():
    data = {
        "Gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "TenureinMonths": tenure,
        "InternetService": {"No": 0, "DSL": 1, "Fiber Optic": 2}[internet],
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "MonthlyCharge": monthly_charge,
        "TotalCharges": total_charges,
        "SatisfactionScore": satisfaction
    }
    return pd.DataFrame([data])

# ===============================
# PREDICTION
# ===============================
st.markdown("## 🔮 Prediction")

if st.button("Predict Churn"):
    input_df = preprocess()

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer is likely to churn\n\n**Probability:** {prob:.2%}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\n**Probability:** {prob:.2%}")

# ===============================
# BUSINESS INSIGHT
# ===============================
st.markdown("## 📊 Business Insight")
st.info("""
Customers with:
• Low tenure  
• Month-to-month contracts  
• High monthly charges  
• Low satisfaction score  

are more likely to churn.

👉 Retention offers should target these users.
""")
