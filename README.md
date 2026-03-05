# 🔄 Customer Churn Prediction & Analytics Platform

> An end-to-end Data Science project covering SQL analytics, exploratory data analysis, machine learning, model explainability (SHAP), and an interactive business dashboard — built on real-world telecom & banking datasets.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-F2C811?logo=powerbi)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Project Overview

Customer churn is one of the most critical business problems across industries. This project builds a **production-style churn prediction system** that:

- Identifies customers likely to churn using ML models
- Explains **why** a customer is at risk using SHAP values
- Provides business stakeholders with an interactive Power BI dashboard
- Serves real-time churn probability via a Streamlit web app

**Domain:** Telecom + Banking  
**Dataset:** IBM Telco Churn (7,043 customers) + Bank Customer Churn (10,000 customers)  
**Target Variable:** `Churn` (Binary: Yes / No)

---

## 🏗️ Architecture

```
Raw Data (CSV)
      ↓
PostgreSQL Database  ←──  SQL Analysis & Cohort Queries
      ↓
Python EDA Pipeline  ──→  Insights & Visualizations
      ↓
Feature Engineering
      ↓
ML Model Training  (Logistic Regression → Random Forest → XGBoost)
      ↓
SHAP Explainability  ──→  Business Insights
      ↓
Streamlit App  ──→  Live Prediction Demo
      ↑
Power BI Dashboard  ←──  Connected to PostgreSQL
```

---

## 📁 Repository Structure

```
customer-churn-prediction/
│
├── 📂 data/
│   ├── raw/                          # Original downloaded CSVs
│   │   ├── telco_churn.csv
│   │   └── bank_churn.csv
│   └── processed/                    # Cleaned & feature-engineered data
│       ├── telco_processed.csv
│       └── bank_processed.csv
│
├── 📂 sql/
│   ├── schema.sql                    # PostgreSQL table definitions
│   ├── data_ingestion.sql            # Data loading scripts
│   ├── cohort_analysis.sql           # Monthly retention cohorts
│   ├── rfm_segmentation.sql          # RFM scoring with window functions
│   ├── churn_kpis.sql                # Business KPI queries
│   └── exploratory_queries.sql       # Ad-hoc analysis queries
│
├── 📂 notebooks/
│   ├── 01_eda_telco.ipynb            # EDA on Telco dataset
│   ├── 02_eda_bank.ipynb             # EDA on Bank dataset
│   ├── 03_feature_engineering.ipynb  # Feature creation & encoding
│   ├── 04_model_building.ipynb       # Model training & evaluation
│   ├── 05_shap_explainability.ipynb  # SHAP analysis & plots
│   └── 06_model_comparison.ipynb     # Cross-model benchmarking
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocess.py                 # Reusable data cleaning functions
│   ├── features.py                   # Feature engineering pipeline
│   ├── train.py                      # Model training script
│   ├── evaluate.py                   # Metrics & evaluation utilities
│   ├── predict.py                    # Inference / scoring logic
│   └── db_connect.py                 # PostgreSQL connection helper
│
├── 📂 models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl                    # Saved StandardScaler
│
├── 📂 app/
│   ├── streamlit_app.py              # Main Streamlit application
│   ├── components/
│   │   ├── prediction_form.py
│   │   └── shap_plot.py
│   └── assets/
│       └── style.css
│
├── 📂 dashboard/
│   ├── churn_dashboard.pbix          # Power BI dashboard file
│   └── dashboard_preview.png         # Screenshot for README
│
├── 📂 reports/
│   ├── eda_summary.pdf               # Auto-generated EDA report
│   └── model_performance.md          # Model metrics documentation
│
├── 📂 tests/
│   ├── test_preprocess.py
│   └── test_predict.py
│
├── .gitignore
├── requirements.txt
├── environment.yml                   # Conda environment file
├── config.yaml                       # Project configuration
├── setup.py
└── README.md
```

---

## 📊 Datasets Used

| Dataset | Source | Rows | Features | Domain |
|---|---|---|---|---|
| IBM Telco Churn | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | 7,043 | 21 | Telecom |
| Bank Customer Churn | [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset) | 10,000 | 12 | Banking |

---

## 🔍 EDA Highlights

- **Churn Rate:** ~26.5% in Telco, ~20.4% in Banking
- **Key Finding 1:** Month-to-month contract customers churn **3x more** than yearly contract customers
- **Key Finding 2:** Customers with tenure < 12 months are the highest risk segment
- **Key Finding 3:** Electronic check payment users show significantly higher churn
- **Key Finding 4:** Fiber optic internet users churn more despite higher service quality expectations

> 📓 See `notebooks/01_eda_telco.ipynb` for full analysis with visualizations

---

## 🤖 Machine Learning Pipeline

### Models Trained

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | ~80% | ~0.84 | ~0.65 | ~0.57 | ~0.61 |
| Random Forest | ~82% | ~0.87 | ~0.70 | ~0.62 | ~0.66 |
| **XGBoost** ✅ | **~85%** | **~0.91** | **~0.74** | **~0.68** | **~0.71** |

### Techniques Used
- **Class Imbalance Handling:** SMOTE + class_weight balancing
- **Feature Selection:** Correlation analysis + SHAP importance
- **Hyperparameter Tuning:** GridSearchCV / Optuna
- **Cross Validation:** Stratified K-Fold (k=5)

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to explain individual predictions:

- **Global explanation:** Which features matter most overall?
- **Local explanation:** Why did this specific customer get flagged?
- **Business translation:** "Customer X is at risk primarily due to month-to-month contract + high monthly charges"

> 📓 See `notebooks/05_shap_explainability.ipynb`

---

## 🗄️ SQL Analysis (PostgreSQL)

Key analyses performed using SQL:

```sql
-- Example: Cohort Retention Analysis
WITH cohort AS (
    SELECT customer_id,
           DATE_TRUNC('month', join_date) AS cohort_month,
           DATE_TRUNC('month', churn_date) AS churn_month
    FROM customers
),
retention AS (
    SELECT cohort_month,
           COUNT(customer_id) AS total_customers,
           COUNT(CASE WHEN churn_month IS NULL THEN 1 END) AS retained
    FROM cohort
    GROUP BY cohort_month
)
SELECT cohort_month,
       total_customers,
       retained,
       ROUND(100.0 * retained / total_customers, 2) AS retention_rate
FROM retention
ORDER BY cohort_month;
```

Other SQL analyses: RFM segmentation, churn by service tier, revenue at risk, lifetime value estimation.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Set Up Environment
```bash
# Using pip
pip install -r requirements.txt

# OR using conda
conda env create -f environment.yml
conda activate churn-env
```

### 3. Set Up PostgreSQL
```bash
# Create database
createdb churn_db

# Run schema
psql -d churn_db -f sql/schema.sql

# Load data
psql -d churn_db -f sql/data_ingestion.sql
```

### 4. Configure Project
```yaml
# config.yaml
database:
  host: localhost
  port: 5432
  name: churn_db
  user: your_username

model:
  best_model: xgboost
  threshold: 0.45
```

### 5. Run Notebooks in Order
```
notebooks/01_eda_telco.ipynb
notebooks/02_eda_bank.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_model_building.ipynb
notebooks/05_shap_explainability.ipynb
```

### 6. Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Power BI Dashboard

The dashboard connects directly to PostgreSQL and includes:

- 📊 **Overview Page:** Total customers, churn rate, revenue at risk
- 👥 **Segment Analysis:** Churn by contract, payment method, tenure
- 🔥 **Risk Heatmap:** At-risk customer segments
- 💰 **Financial Impact:** Revenue loss projection & retention ROI

> 📁 See `dashboard/churn_dashboard.pbix`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL 15 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, Imbalanced-learn |
| Explainability | SHAP |
| Web App | Streamlit |
| BI Dashboard | Power BI |
| Version Control | Git & GitHub |

---

## 📦 Requirements

```
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
shap==0.43.0
imbalanced-learn==0.11.0
streamlit==1.28.0
plotly==5.17.0
matplotlib==3.7.0
seaborn==0.12.0
psycopg2-binary==2.9.7
sqlalchemy==2.0.0
pyyaml==6.0
joblib==1.3.0
optuna==3.3.0
```

---

## 🎯 Key Business Insights

1. **Highest Risk Segment:** Month-to-month contract + Fiber optic + Electronic check payment → 55%+ churn probability
2. **Retention Lever:** Customers who adopt 3+ services churn 40% less — upselling reduces churn
3. **Critical Window:** First 12 months is when 60% of churned customers leave — early intervention is key
4. **Revenue at Risk:** ~$2.8M annual revenue at risk from high-probability churners (Telco dataset)

---

## 📬 Contact

**Your Name**  
B.Tech Final Year | Data Science Enthusiast  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)  
💼 [Portfolio](https://yourportfolio.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project useful, please consider giving it a star!**
