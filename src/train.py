import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def apply_smote(X_train, y_train, random_state: int = 42):
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }

    logger.info(f"\n{'='*40}")
    logger.info(f"Model: {model_name}")
    for k, v in metrics.items():
        if k != "model":
            logger.info(f"  {k:<12}: {v}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return metrics


def train_all_models(X_train, y_train, X_test, y_test, random_state: int = 42) -> dict:
    results = {}

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    lr.fit(X_train, y_train)
    results["logistic_regression"] = {"model": lr, "metrics": evaluate_model(lr, X_test, y_test, "Logistic Regression")}
    joblib.dump(lr, "models/logistic_regression.pkl")

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = {"model": rf, "metrics": evaluate_model(rf, X_test, y_test, "Random Forest")}
    joblib.dump(rf, "models/random_forest.pkl")

    logger.info("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="auc",
        verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    results["xgboost"] = {"model": xgb, "metrics": evaluate_model(xgb, X_test, y_test, "XGBoost")}
    joblib.dump(xgb, "models/xgboost_model.pkl")

    logger.info("All models trained and saved to models/")
    return results


def get_best_model(results: dict) -> tuple:
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    logger.info(f"Best model: {best_name} (AUC: {results[best_name]['metrics']['roc_auc']})")
    return best_name, results[best_name]["model"]


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_telco, clean_telco
    from features import engineer_telco_features, build_preprocessor

    df = load_telco("../data/raw/telco_churn.csv")
    df = clean_telco(df)
    df = engineer_telco_features(df)

    TARGET = "Churn"
    DROP_COLS = ["customerID", "tenure_bucket", "age_bucket"]
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    X_train, X_test, y_train, y_test = split_data(df, target=TARGET)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    results = train_all_models(X_train_res, y_train_res, X_test, y_test)
    best_name, best_model = get_best_model(results)
    print(f"\nBest model: {best_name}")
