# Customer Churn Prediction & Analytics Platform

An end-to-end Data Science project that predicts which customers will leave a telecom/banking company, explains why they might leave, and provides actionable retention recommendations.

![Python](https://img.shields.io/badge/Python-3.14+-blue?logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-F2C811?logo=powerbi)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## Overview

Customer churn is one of the most critical business problems across industries. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a system that:

- Predicts which customers are likely to leave — **before they actually leave**
- Explains **why** a customer is at risk using key risk factors
- Gives the retention team **actionable recommendations** to intervene in time

**Datasets:** IBM Telco Churn + Bank Customer Churn (Kaggle)  
**Target Variable:** `Churn` (0 = Stay, 1 = Will Churn)  
**Class Distribution:** ~73% Stay, ~27% Churn

---

## Current Status

| Component | Status |
|---|---|
| Project structure & documentation | ✅ Complete |
| Streamlit prediction app (rule-based) | ✅ Complete |
| SQL schema & analytics queries | ✅ Complete |
| Power BI dashboard | ✅ Complete |
| EDA notebooks | 🔄 In Progress |
| Feature engineering | 🔄 In Progress |
| XGBoost model training | ⏳ Pending |
| Model connected to Streamlit app | ⏳ Pending |

---

## Project Structure

```
customer-churn-prediction/
├── app/
│   └── streamlit_app.py            # Live prediction web app ✅
├── data/
│   └── raw/
│       └── telco_churn.csv         # IBM Telco dataset (Kaggle)
├── src/
│   ├── preprocess.py               # Data cleaning functions
│   ├── features.py                 # Feature engineering pipeline
│   ├── train.py                    # Model training (LR → RF → XGBoost)
│   └── predict.py                  # Inference & SHAP explanation
├── notebooks/
│   ├── 01_eda_telco.py             # Exploratory Data Analysis
│   ├── 03_feature_engineering.py   # Feature creation walkthrough
│   ├── 04_model_building.py        # Model training & evaluation
│   └── 05_shap_explainability.py   # SHAP feature importance
├── sql/
│   ├── schema.sql                  # PostgreSQL table definitions
│   ├── cohort_analysis.sql         # Retention cohort queries
│   ├── rfm_segmentation.sql        # RFM scoring with window functions
│   └── churn_kpis.sql              # Business KPI queries
├── models/                         # Trained model files (.pkl)
├── dashboard/
│   ├── churn_dashboard.pbix        # Power BI dashboard ✅
│   └── dashboard_preview.png       # Dashboard screenshot ✅
├── reports/                        # EDA charts & model reports
├── requirements.txt
├── config.yaml
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14+ |
| Database | PostgreSQL 15 |
| Data Processing | Pandas 2.2+, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn 1.4+, XGBoost 2.0+ |
| Explainability | SHAP 0.44+ |
| Web App | Streamlit 1.30+ |
| Dashboard | Power BI |

---

## Power BI Dashboard

Built on the IBM Telco dataset (7,043 customers) with 6 visuals:

- **KPI Cards** — Total Customers, Churn Rate (26.5%), Monthly Revenue at Risk (₹1.39L), Churned Customers
- **Churn by Contract Type** — Month-to-month shows highest churn at ~42%
- **Churn by Payment Method** — Electronic check users churn at 45%
- **Churn by Tenure** — Downward trend confirms new customers are highest risk
- **Churn by Internet Service** — Fiber optic users dominate churned segment
- **At-Risk Customers Table** — Filtered to Churn = Yes, sorted by monthly charges

> 📁 See `dashboard/churn_dashboard.pbix`

---

## Streamlit App (Current)

The app currently uses a **rule-based scoring system** derived directly from EDA insights. It will be upgraded to use the trained XGBoost model once training is complete.

**Input fields:**
- Tenure (months), Contract Type, Internet Service
- Monthly Charges (₹), Payment Method, Senior Citizen

**Output:**
- Churn Probability (%)
- Risk Level: 🔴 High / 🟡 Medium / 🟢 Low
- Key risk factors driving the prediction
- Business recommendation for the retention team

**Scoring logic (EDA-based):**
```
Score = contract_type + payment_method + internet_type + tenure + charges + age
Churn Probability = 10% + (score × 9%)
```

---

## Key Business Insights (From Dashboard & EDA)

1. Month-to-month contracts show **3x higher churn** than yearly contracts
2. Electronic check users churn at **45%** — highest of all payment methods
3. Customers with tenure **under 12 months** are the highest risk segment
4. Fiber optic users churn more **despite paying a premium**
5. Monthly charges above **₹1,500** correlate with increased churn probability
6. **₹1.39L monthly revenue** is at risk from churned customers

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/phantom074/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app/streamlit_app.py

# Access at: http://localhost:8501
```

**To set up PostgreSQL (optional):**
```bash
createdb churn_db
psql -d churn_db -f sql/schema.sql
```


## Author

**Mukul**  
Data Science Enthusiast  
[GitHub](https://github.com/phantom074)

---

⭐ If you found this useful, consider giving it a star!
