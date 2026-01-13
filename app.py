import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ===============================
# LOAD MODEL & FEATURE SCHEMA
# ===============================
MODEL_PATH = "model/churn_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("❌ Model or feature schema not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <h1 style='text-align: center;'>📉 Customer Churn Prediction</h1>
    <p style='text-align: center; color: gray;'>
    End-to-End ML Project using Ensemble Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("🧾 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
married = st.sidebar.selectbox("Married", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

st.sidebar.markdown("### 📊 Account Details")

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["No", "DSL", "Fiber Optic"]
)

monthly_charge = st.sidebar.number_input(
    "Monthly Charge (₹)",
    min_value=0.0,
    value=70.0
)

satisfaction = st.sidebar.slider(
    "Satisfaction Score (1 = Worst, 5 = Best)",
    1, 5, 3
)

# ===============================
# PREPROCESS INPUT (🔥 MOST IMPORTANT FIX)
# ===============================
def preprocess_input():
    # Base input features (only what user provides)
    data = {
        "Gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "TenureinMonths": tenure,
        "Contract": {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }[contract],
        "InternetService": {
            "No": 0,
            "DSL": 1,
            "Fiber Optic": 2
        }[internet],
        "MonthlyCharge": monthly_charge,
        "SatisfactionScore": satisfaction
    }

    input_df = pd.DataFrame([data])

    # 🔐 Add missing features (used during training but not in UI)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # 🔐 Ensure correct column order
    input_df = input_df[feature_names]

    return input_df

# ===============================
# MAIN CONTENT
# ===============================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🔮 Churn Prediction")

    st.markdown(
        """
        This model predicts **customer churn risk** using:
        - Customer tenure
        - Contract type
        - Service usage
        - Billing behavior
        - Satisfaction score
        """
    )

    if st.button("🚀 Predict Churn", use_container_width=True):
        input_df = preprocess_input()

        prediction = model.predict(input_df)[0]

        # 🛡️ Safe probability handling
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = 0.5

        st.markdown("---")

        if prediction == 1:
            st.error(
                f"""
                ⚠️ **High Churn Risk Detected**

                **Churn Probability:** `{probability:.2%}`

                👉 Immediate retention action recommended.
                """
            )
        else:
            st.success(
                f"""
                ✅ **Low Churn Risk**

                **Churn Probability:** `{probability:.2%}`

                👍 Customer is likely to stay.
                """
            )

with col2:
    st.markdown("## 📊 Business Insight")

    st.info(
        """
        **Customers are more likely to churn if they have:**
        - Low tenure
        - Month-to-month contracts
        - High monthly charges
        - Low satisfaction score

        **Recommended Actions:**
        - Offer loyalty discounts
        - Upsell long-term contracts
        - Improve service experience
        """
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built by Vishal Kumar • Internship-Ready ML Project</p>",
    unsafe_allow_html=True
)
