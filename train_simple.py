import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading data...")
df = pd.read_csv('data/raw/telco_churn.csv')
logger.info(f"Loaded {len(df)} rows")

df['Churn'] = df['Churn'].map({0: 0, 1: 1, 'Yes': 1, 'No': 0})
df.fillna(0, inplace=True)

feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 
                'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
if 'InternetService' in df.columns:
    df['InternetService_num'] = df['InternetService'].map({'DSL': 0, 'Fiber optic': 1, 'No': 2}).fillna(2)
    feature_cols.append('InternetService_num')

if 'PaymentMethod' in df.columns:
    df['PaymentMethod_num'] = df['PaymentMethod'].map({
        'Electronic check': 0, 'Mailed check': 1, 
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    }).fillna(3)
    feature_cols.append('PaymentMethod_num')

X = df[feature_cols].values
y = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logger.info("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

logger.info("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)
logger.info(f"LR AUC: {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.4f}")

logger.info("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
logger.info(f"RF AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")

logger.info("Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='auc')
xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)
logger.info(f"XGB AUC: {roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]):.4f}")

joblib.dump(xgb, 'models/xgboost_model.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
logger.info("Model saved to models/xgboost_model.pkl")

print("\n✅ Training complete! Ready to run the Streamlit app.")
