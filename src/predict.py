import pandas as pd
import numpy as np
import joblib
import shap
import logging

logger = logging.getLogger(__name__)


def load_model(model_path: str = "models/xgboost_model.pkl"):
    """Load a saved model from disk."""
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def predict_churn(model, X: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    predicted = (proba >= threshold).astype(int)

    results = pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_predicted":   predicted,
        "risk_level":        pd.cut(
            proba,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"]
        )
    })
    return results


def get_shap_explanation(model, X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    rows = []
    for i in range(len(X)):
        row_shap = pd.Series(sv[i], index=X.columns)
        top_features = row_shap.abs().nlargest(top_n)
        rows.append({
            f"top_{j+1}_feature": feat for j, feat in enumerate(top_features.index)
        })

    return pd.DataFrame(rows)


def predict_single_customer(model, customer_data: dict, feature_columns: list, threshold: float = 0.45) -> dict:
    X = pd.DataFrame([customer_data])[feature_columns]
    proba = model.predict_proba(X)[0][1]
    predicted = int(proba >= threshold)

    risk = "🔴 High Risk" if proba >= 0.6 else "🟡 Medium Risk" if proba >= 0.3 else "🟢 Low Risk"

    return {
        "churn_probability": round(float(proba), 4),
        "churn_predicted":   predicted,
        "risk_level":        risk
    }
