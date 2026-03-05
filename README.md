# Customer Churn Prediction & Analytics Platform

An end-to-end Data Science project covering SQL analytics, exploratory data analysis, machine learning, model explainability (SHAP), and an interactive business dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-F2C811?logo=powerbi)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Overview

Customer churn is one of the most critical business problems across industries. This project builds a churn prediction system that identifies customers likely to leave, explains why using SHAP values, and presents insights through a Power BI dashboard and a live Streamlit app.

**Datasets:** IBM Telco Churn (7,043 customers) + Bank Customer Churn (10,000 customers)  
**Target Variable:** `Churn` (Binary: Yes / No)

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                        # Original CSVs
│   └── processed/                  # Cleaned data
├── sql/
│   ├── schema.sql
│   ├── cohort_analysis.sql
│   └── rfm_segmentation.sql
├── notebooks/
│   ├── 01_eda_telco.ipynb
│   ├── 02_eda_bank.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_building.ipynb
│   └── 05_shap_explainability.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── models/                         # Saved model files (.pkl)
├── app/
│   └── streamlit_app.py
├── dashboard/
│   └── churn_dashboard.pbix
├── requirements.txt
├── config.yaml
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL 15 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Web App | Streamlit |
| Dashboard | Power BI |

---

## Models & Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | ~80% | ~0.84 |
| Random Forest | ~82% | ~0.87 |
| XGBoost ✅ | ~85% | ~0.91 |

XGBoost was selected as the final model. Class imbalance was handled using SMOTE.

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/phantom074/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up PostgreSQL
createdb churn_db
psql -d churn_db -f sql/schema.sql

# 4. Run notebooks in order (01 → 05)

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Key Insights

- Month-to-month contract customers churn **3x more** than yearly contract customers
- Customers with tenure under 12 months are the **highest risk segment**
- Customers using **3+ services churn 40% less** — upselling reduces churn
- Electronic check payment users show significantly higher churn rates

---

## Author

**Mukul**  
B.Tech Final Year | Data Science Enthusiast  
[GitHub](https://github.com/phantom074)

---

⭐ If you found this useful, consider giving it a star!
