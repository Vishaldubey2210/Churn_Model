# 📉 Customer Churn Prediction using Ensemble Learning

An **end-to-end, production-ready Machine Learning system** designed to predict customer churn in subscription-based businesses such as **Telecom, SaaS, and Banking**.

This project demonstrates a **real-world ML workflow** — from data understanding and feature engineering to model deployment — with a strong focus on **business impact, interpretability, and reliability**.

---

## 🚀 Project Overview

Customer churn directly impacts business revenue.  
Failing to identify churn-prone customers can lead to **significant financial loss**, while timely detection enables **proactive retention strategies**.

### 🎯 Objective
Predict whether a customer will **churn (1) or stay (0)**, enabling businesses to take **data-driven retention actions**.

---

## 🧠 Key Highlights

- ✅ End-to-end ML pipeline (EDA → Deployment)  
- ✅ Strong focus on **data leakage prevention**  
- ✅ Handles **class imbalance (~26% churn)**  
- ✅ Optimized for **recall of churned customers** (business-critical)  
- ✅ Advanced **Ensemble Learning (Voting Classifier)**  
- ✅ Feature importance & model interpretability  
- ✅ Production-ready **Streamlit web application**  
- ✅ Modular & scalable project structure  

---

## 🧩 Solution Strategy

The project follows a structured, industry-aligned ML workflow:

1. Exploratory Data Analysis (EDA)  
2. Business-driven churn hypothesis validation  
3. Correlation analysis & feature redundancy removal  
4. Feature engineering & encoding  
5. Baseline modeling (Logistic Regression)  
6. Advanced modeling (Random Forest, Gradient Boosting)  
7. Ensemble learning (Soft Voting Classifier)  
8. Threshold tuning for business optimization  
9. Model evaluation & comparison  
10. Deployment via Streamlit  

---

## 📂 Dataset Description

- **Dataset Size:** ~7,000+ customers  
- **Domain:** Telecom customer behavior  

### 🎯 Target Variable
- `ChurnLabel`
  - `1` → Customer churned  
  - `0` → Customer retained  

---

## 🧾 Feature Categories

### 👤 Demographics
- Age  
- Gender  
- Senior Citizen  
- Dependents  

### 🌍 Geographic
- City  
- State  

### 📊 Account & Contract
- Contract Type  
- Tenure in Months  
- Payment Method  

### 💰 Usage & Billing
- Monthly Charges  
- Data Usage  
- Long Distance Charges  

### ⭐ Customer Value
- CLTV  
- Satisfaction Score  

---

## 📊 Exploratory Data Analysis (EDA)

Key insights:

- 🔹 **Class imbalance (~26% churn)**  
- 🔹 Customers with **low tenure** churn more  
- 🔹 **Month-to-month contracts** have highest churn  
- 🔹 **High monthly charges** increase churn probability  
- 🔹 **Low satisfaction score** is a strong churn indicator  

---

## 🔍 Feature Selection & Data Leakage Handling

To ensure model generalization:

### ❌ Removed Features
- `TotalCharges`, `TotalRevenue` → high correlation (leakage risk)  
- `ZipCode`, `Latitude`, `Longitude` → noise, no behavioral value  

### ✅ Retained Features
- Behavioral and service-level features available **before churn occurs**

---

## ⚙️ Feature Engineering

- One-hot encoding using `pd.get_dummies()`  
- Target encoding:
  - `Yes → 1`, `No → 0`  
- Stratified train-test split to preserve class distribution  

---

## 🤖 Models Used

### 1️⃣ Logistic Regression (Baseline)
- Interpretable model  
- Balanced class weights  

### 2️⃣ Random Forest
- Handles non-linearity  
- Robust to noise  

### 3️⃣ Gradient Boosting (Strong Performer)
- High predictive performance  
- Excellent ROC-AUC  

### 4️⃣ Final Model — Voting Classifier (Ensemble)
- Combines:
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
- Improves **recall + overall robustness**

---

## 📈 Model Performance

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|------|---------|---------------|--------|
| Logistic Regression | ~0.88 | Good | ~0.96 |
| Random Forest | ~0.88 | Moderate | ~0.95 |
| Gradient Boosting | ~0.95 | High | ~0.99 |
| **Voting Classifier (Final)** | **~0.95** | **Highest 🔥** | **~0.98** |

---

## 🧠 Business Insight

- Missing a churned customer is **more costly than a false alarm**  
- Therefore, **recall is prioritized over raw accuracy**  
- The final model is optimized to **capture maximum churn cases**

---

## 🖥️ Deployment

The model is deployed using **Streamlit**.

### 🔗 Features
- Interactive UI  
- Real-time predictions  
- Probability-based churn risk  
- Business recommendations  

---

## 📁 Project Structure
Churn_Model/
│
├── data/
├── notebooks/
├── model/
│ ├── churn_model.pkl
│ └── feature_names.pkl
├── app.py
├── requirements.txt
└── README.md


---

## ⚡ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

🚀 Future Improvements
SHAP explainability for model interpretation
Hyperparameter tuning (Optuna/GridSearch)
API deployment (FastAPI)
Real-time data pipeline integration


👨‍💻 Author

Vishal Kumar

Aspiring ML Engineer
Focused on real-world, production-ready ML systems
