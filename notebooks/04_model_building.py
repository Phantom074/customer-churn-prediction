# %% [markdown]
# # 🤖 Notebook 04 — Model Building & Evaluation
# **Author:** Mukul (github.com/phantom074)
#
# Training Logistic Regression, Random Forest, and XGBoost.
# Evaluating with AUC-ROC, F1, Precision, Recall.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import sys
sys.path.append("../src")
from train import split_data, apply_smote, train_all_models, get_best_model
import warnings
warnings.filterwarnings("ignore")

# %% Load Processed Data
df = pd.read_csv("../data/processed/telco_processed.csv")
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.1%}")

# %% Prepare Categorical Columns (One-Hot Encode)
CAT_COLS = ["gender", "InternetService", "Contract", "PaymentMethod"]
df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

# %% Split
X_train, X_test, y_train, y_test = split_data(df, target="Churn")
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# %% Apply SMOTE
X_train_res, y_train_res = apply_smote(X_train, y_train)

# %% Train All Models
results = train_all_models(X_train_res, y_train_res, X_test, y_test)
best_name, best_model = get_best_model(results)

# %% Compare Models
metrics_df = pd.DataFrame([r["metrics"] for r in results.values()])
print("\nModel Comparison:")
print(metrics_df.to_string(index=False))

# %% ROC Curve Comparison
fig, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    RocCurveDisplay.from_estimator(res["model"], X_test, y_test, ax=ax, name=name.replace("_", " ").title())
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("../reports/roc_curves.png", bbox_inches="tight")
plt.show()

# %% Best Model Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, colorbar=False,
                                       display_labels=["Stay", "Churn"], cmap="Reds")
ax.set_title(f"Confusion Matrix — {best_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/confusion_matrix.png", bbox_inches="tight")
plt.show()

print(f"\nBest model: {best_name}")
print(f"Best AUC-ROC: {results[best_name]['metrics']['roc_auc']}")
