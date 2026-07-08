import streamlit as st
import pandas as pd
import joblib
import os
import sys

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# ===============================
# LOAD MODEL & FEATURES
# ===============================
MODEL_PATH = "model/churn_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("❌ Model or feature schema not found. Please train the model first.")
    st.stop()

try:
    # Load model with error handling
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
except ModuleNotFoundError as e:
    st.error(f"❌ Error loading model: Missing module {str(e)}")
    st.info("Please ensure all dependencies in requirements.txt are installed.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("The model file may be corrupted. Please retrain the model.")
    st.stop()

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
# PREPROCESS INPUT (FIXED)
# ===============================
def preprocess_input():
    data = {
        "Gender": gender,
        "SeniorCitizen": senior,
        "Married": married,
        "Dependents": dependents,
        "TenureinMonths": tenure,
        "Contract": contract,
        "InternetService": internet,
        "MonthlyCharge": monthly_charge,
        "SatisfactionScore": satisfaction
    }

    input_df = pd.DataFrame([data])

    # 🔥 SAME encoding as training
    input_df = pd.get_dummies(input_df)

    # 🔐 match training columns
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    return input_df

# ===============================
# MAIN LAYOUT
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
        try:
            input_df = preprocess_input()

            with st.spinner("Analyzing customer data..."):
                prediction = model.predict(input_df)[0]

                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(input_df)[0][1]
                else:
                    probability = 0.5

            st.markdown("---")

            # 🎯 RESULT DISPLAY
            if prediction == 1:
                st.error("⚠️ High Churn Risk Detected")
            else:
                st.success("✅ Low Churn Risk")

            # 📊 Probability
            st.write(f"### 📊 Churn Probability: {probability:.2%}")
            st.progress(float(probability))

            # 🎯 Risk Level
            if probability > 0.7:
                st.error("🔴 High Risk Customer")
            elif probability > 0.4:
                st.warning("🟡 Medium Risk Customer")
            else:
                st.success("🟢 Low Risk Customer")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please try again or contact support.")

# ===============================
# SIDE PANEL
# ===============================
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
        - Promote long-term plans
        - Improve customer experience
        """
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built by Vishal Kumar • ML Project</p>",
    unsafe_allow_html=True
)