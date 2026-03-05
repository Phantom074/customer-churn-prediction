# %% [markdown]
# # 📐 Notebook 03 — Feature Engineering
# **Author:** Mukul (github.com/phantom074)
#
# Building new features that improve model performance.

# %% Imports
import pandas as pd
import numpy as np
import sys
sys.path.append("../src")
from preprocess import load_telco, clean_telco
from features import engineer_telco_features

# %% Load & Clean
df = load_telco("../data/raw/telco_churn.csv")
df = clean_telco(df)
print(f"After cleaning: {df.shape}")

# %% Engineer Features
df = engineer_telco_features(df)

# %% Review new features
new_features = [
    "charges_per_month", "tenure_bucket", "num_addons",
    "is_high_value", "is_month_to_month", "is_e_check",
    "is_fiber", "risk_score"
]

print("New features created:")
print(df[new_features + ["Churn"]].head(10).to_string())

# %% Churn rate by risk_score
print("\nChurn rate by risk_score:")
print(df.groupby("risk_score")["Churn"].mean().apply(lambda x: f"{x:.1%}"))

# %% Save processed data
DROP_COLS = ["customerID", "tenure_bucket", "age_bucket"]
df_processed = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
df_processed.to_csv("../data/processed/telco_processed.csv", index=False)
print(f"\nSaved processed data: {df_processed.shape}")
