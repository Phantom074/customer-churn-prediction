import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_telco(filepath: str) -> pd.DataFrame:
    """Load and do initial cleaning of the IBM Telco dataset."""
    logger.info(f"Loading Telco data from {filepath}")
    df = pd.read_csv(filepath)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df
def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for Telco dataset."""
    df = df.copy()

    before = len(df)
    df.drop_duplicates(subset="customerID", inplace=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    three_level_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in three_level_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    logger.info("Cleaned binary and three-level columns")
    return df
def get_telco_feature_types(df: pd.DataFrame) -> Tuple[List, List]:
    """Return lists of numeric and categorical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "customerID"]
    return numeric_cols, categorical_cols


def load_bank(filepath: str) -> pd.DataFrame:
    """Load and do initial cleaning of the Bank Churn dataset."""
    logger.info(f"Loading Bank data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df
def clean_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for Bank Churn dataset."""
    df = df.copy()

    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        df.fillna(df.median(numeric_only=True), inplace=True)

    logger.info("Bank dataset cleaned")
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a missing value summary report."""
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
def outlier_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """IQR-based outlier detection for numeric columns."""
    rows = []
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        rows.append({
            "column": col,
            "outlier_count": len(outliers),
            "outlier_pct": round(len(outliers) / len(df) * 100, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2)
        })
    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)
