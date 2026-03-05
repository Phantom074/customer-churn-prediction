# %% [markdown]
# # 🧠 Notebook 05 — SHAP Explainability
# **Author:** Mukul (github.com/phantom074)
#
# Understanding WHY the model predicts churn for each customer.

# %% Imports
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append("../src")
import warnings
warnings.filterwarnings("ignore")
shap.initjs()

# %% Load Model & Data
model   = joblib.load("../models/xgboost_model.pkl")
df      = pd.read_csv("../data/processed/telco_processed.csv")
CAT_COLS = ["gender", "InternetService", "Contract", "PaymentMethod"]
df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Use a sample for SHAP (faster)
X_sample = X.sample(500, random_state=42)

# %% SHAP TreeExplainer
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# %% Global Feature Importance — Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/shap_feature_importance.png", bbox_inches="tight")
plt.show()

# %% SHAP Beeswarm (Detailed)
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary — Feature Impact on Churn", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/shap_beeswarm.png", bbox_inches="tight")
plt.show()

# %% Individual Customer Explanation
print("="*50)
print("INDIVIDUAL CUSTOMER EXPLANATION")
print("="*50)

customer_idx = 0  # Change this to inspect different customers
customer     = X_sample.iloc[[customer_idx]]
prob         = model.predict_proba(customer)[0][1]

print(f"Customer #{customer_idx} — Churn Probability: {prob:.1%}")

# Force plot (save as image)
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[customer_idx],
    customer,
    matplotlib=True,
    show=False
)
plt.savefig("../reports/shap_force_plot.png", bbox_inches="tight", dpi=150)
plt.show()

# %% Top 5 drivers for this customer
row_shap = pd.Series(shap_values[customer_idx], index=X_sample.columns)
print("\nTop 5 features driving this prediction:")
print(row_shap.abs().nlargest(5).to_string())

# %% SHAP Dependence Plot — tenure
shap.dependence_plot("tenure", shap_values, X_sample, show=False)
plt.title("SHAP Dependence: Tenure vs Churn Impact", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/shap_dependence_tenure.png", bbox_inches="tight")
plt.show()
