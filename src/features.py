import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import logging

logger = logging.getLogger(__name__)


def engineer_telco_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing Telco columns."""
    df = df.copy()

    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72, 100],
        labels=["0-12m", "13-24m", "25-48m", "49-72m", "72m+"]
    )

    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_addons"] = df[addon_cols].sum(axis=1)

    df["is_high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    df["is_e_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    df["risk_score"] = df["is_month_to_month"] + df["is_e_check"] + df["is_fiber"]

    logger.info(f"Engineered {df.shape[1]} total features")
    return df


def engineer_bank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing Bank Churn columns."""
    df = df.copy()

    df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    df["age_bucket"] = pd.cut(
        df["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"]
    )

    df["products_per_year"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    df["has_zero_balance"] = (df["Balance"] == 0).astype(int)

    df["engaged_customer"] = (
        (df["IsActiveMember"] == 1) & (df["HasCrCard"] == 1)
    ).astype(int)

    logger.info(f"Engineered {df.shape[1]} total features")
    return df


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ])
    return preprocessor
def save_preprocessor(preprocessor, path: str = "models/preprocessor.pkl"):
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor saved to {path}")
def load_preprocessor(path: str = "models/preprocessor.pkl"):
    return joblib.load(path)
