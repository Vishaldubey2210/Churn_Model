
---

```markdown
# 📊 Customer Churn Prediction using Ensemble Learning

## 📌 Project Overview
Customer churn is one of the most critical business problems in subscription-based industries such as telecom, SaaS, and banking.  
This project builds an **end-to-end, industry-grade machine learning pipeline** to predict customer churn using **ensemble learning techniques**.

The solution focuses on:
- Understanding churn drivers through EDA
- Preventing data leakage
- Handling class imbalance
- Prioritizing **recall for churned customers**
- Delivering a **production-ready model**

---

## 🎯 Business Problem
Churned customers represent **direct revenue loss**.  
Missing a churned customer is **far more costly** than incorrectly flagging a loyal one.

**Objective:**  
Predict whether a customer will churn (`Yes / No`) so that the business can take **proactive retention actions**.

---

## 🧠 Solution Strategy
We follow a structured ML workflow aligned with real-world data science practices:

1. Exploratory Data Analysis (EDA)
2. Churn driver validation
3. Correlation & redundancy analysis
4. Feature engineering & encoding
5. Baseline model benchmarking
6. Ensemble learning for final model
7. Business-focused evaluation
8. Model serialization for deployment

---

## 📂 Dataset Description
The dataset contains **7,000+ telecom customers** with demographic, geographic, service usage, billing, and satisfaction-related features.

### Target Variable
- **ChurnLabel**
  - `Yes` → Customer churned
  - `No` → Customer retained

### Feature Categories
- **Demographics:** Age, Gender, Dependents
- **Geographic:** City, State, ZipCode, Latitude, Longitude
- **Account Info:** Contract Type, Tenure, Payment Method
- **Usage:** Monthly Charges, Data Usage, Long Distance Charges
- **Customer Value:** Revenue, CLTV, Satisfaction Score

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights derived during EDA:

- **Class Imbalance:** ~26% customers churned
- **Strong churn drivers identified visually**
- Low tenure and month-to-month contracts show higher churn
- Lower satisfaction scores strongly correlate with churn
- High monthly charges increase churn risk when value perception is low

> _“Churn drivers were validated visually before modeling.”_

---

## 📈 Correlation Analysis & Feature Selection
Highly correlated or leakage-prone features were removed to improve generalization.

### Dropped Features
- `TotalCharges`, `TotalRevenue` (highly correlated with tenure)
- `ZipCode`, `Latitude`, `Longitude` (geographic noise)

### Retained Features
- `TenureinMonths`
- `MonthlyCharge`
- Behavioral & service-level features available **pre-churn**

---

## ⚙️ Feature Engineering
- Categorical features encoded using **Label Encoding**
- ChurnLabel converted to binary (Yes → 1, No → 0)
- Stratified train-test split to preserve churn ratio

---

## 🤖 Models Used

### 1️⃣ Baseline Model – Logistic Regression
Used to establish a transparent benchmark.

- Class weight = `balanced`
- Stratified split
- ROC-AUC based evaluation

**Baseline ROC-AUC:** ~0.95

---

### 2️⃣ Ensemble Models (Final)

#### 🔹 Random Forest
- Captures non-linear churn behavior
- Robust to noise
- Handles feature interactions well

#### 🔹 Gradient Boosting
- Focuses on hard-to-predict churn cases
- Improves recall for minority class

#### 🔹 Soft Voting Classifier (Final Model)
Combines:
- Logistic Regression
- Random Forest
- Gradient Boosting

**Why Voting?**
- Reduces bias & variance
- Improves stability
- Delivers best business-aligned performance

---

## 📊 Evaluation Metrics
Accuracy alone is misleading for churn problems.

We prioritize:
- **Recall (Churn Class)**
- **ROC-AUC Score**
- Confusion Matrix
- Precision-Recall balance

### Final Model Performance
- **ROC-AUC:** ~0.96
- **Strong recall for churned customers**
- Balanced false positives vs false negatives

---

## 💼 Business Interpretation
- Catching a churned customer early enables **retention campaigns**
- Model outputs probabilities → can be used for **risk-based targeting**
- Ensemble model offers **high performance without sacrificing reliability**

---

## 💾 Model Saving
The final ensemble model is serialized using `joblib` for deployment:

```

model/churn_model.pkl

```

This makes the solution **production-ready**.

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📁 Project Structure
```

Churn_Model/
│
├── churn_analysis.ipynb   # Complete ML pipeline
├── app.py                # (Optional) Streamlit app
├── model/
│   └── churn_model.pkl   # Final trained ensemble model
├── requirements.txt
└── README.md

```

---

## 🚀 Future Improvements
- SHAP explainability for feature importance
- Threshold tuning for business-specific recall targets
- Model deployment (Streamlit / API)
- Cost-sensitive learning

---

## 👤 Author
**Vishal Kumar**  
Aspiring Data Scientist | Machine Learning Enthusiast

---

## ⭐ Final Note
This project demonstrates:
- Strong ML fundamentals
- Business-oriented thinking
- Industry-grade modeling practices

✅ **Internship ready**  
✅ **Placement ready**  
✅ **Production ready**
```

---
