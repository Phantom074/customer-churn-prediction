# %% [markdown]
# # 📊 Notebook 01 — EDA: IBM Telco Customer Churn
# **Author:** Mukul (github.com/phantom074)
#
# **Goal:** Understand the dataset structure, distributions, and key patterns
# that drive customer churn.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)

# %% Load Data
df = pd.read_csv("../data/raw/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

print(f"Shape: {df.shape}")
df.head()

# %% Basic Info
print("="*50)
print("Dataset Info")
print("="*50)
print(df.dtypes)
print(f"\nNull values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
print(f"\nChurn rate: {df['Churn_Binary'].mean():.1%}")

# %% [markdown]
# ## 1. Churn Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
axes[0].pie(df["Churn"].value_counts(), labels=["No Churn", "Churn"],
            colors=["#2ECC71", "#E74C3C"], autopct="%1.1f%%", startangle=90)
axes[0].set_title("Overall Churn Rate", fontsize=14, fontweight="bold")

# Count bar
sns.countplot(data=df, x="Churn", palette={"No": "#2ECC71", "Yes": "#E74C3C"}, ax=axes[1])
axes[1].set_title("Churn Count", fontsize=14, fontweight="bold")
axes[1].set_xlabel("")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="bottom", fontsize=12)

plt.tight_layout()
plt.savefig("../reports/01_churn_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Churn by Contract Type

# %%
contract_churn = df.groupby("Contract")["Churn_Binary"].agg(["mean", "count"]).reset_index()
contract_churn.columns = ["Contract", "Churn_Rate", "Count"]
contract_churn["Churn_Rate_Pct"] = (contract_churn["Churn_Rate"] * 100).round(1)

fig = px.bar(
    contract_churn, x="Contract", y="Churn_Rate_Pct",
    color="Churn_Rate_Pct", color_continuous_scale="RdYlGn_r",
    text="Churn_Rate_Pct", title="Churn Rate by Contract Type (%)"
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
fig.show()

print(contract_churn.to_string(index=False))

# %% [markdown]
# ## 3. Tenure Distribution — Churned vs Retained

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for label, color in [("Yes", "#E74C3C"), ("No", "#2ECC71")]:
    subset = df[df["Churn"] == label]["tenure"]
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=f"Churn={label}", edgecolor="white")

ax.set_xlabel("Tenure (months)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Tenure Distribution: Churned vs Retained", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.axvline(df[df["Churn"]=="Yes"]["tenure"].median(), color="#E74C3C", linestyle="--", alpha=0.8, label="Churn Median")
plt.tight_layout()
plt.savefig("../reports/02_tenure_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Monthly Charges vs Churn

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette={"No": "#2ECC71", "Yes": "#E74C3C"}, ax=axes[0])
axes[0].set_title("Monthly Charges by Churn", fontsize=13, fontweight="bold")

# KDE plot
for label, color in [("Yes", "#E74C3C"), ("No", "#2ECC71")]:
    df[df["Churn"]==label]["MonthlyCharges"].plot.kde(ax=axes[1], color=color, label=f"Churn={label}", linewidth=2)
axes[1].set_title("Monthly Charges Distribution", fontsize=13, fontweight="bold")
axes[1].legend()
axes[1].set_xlabel("Monthly Charges ($)")

plt.tight_layout()
plt.savefig("../reports/03_monthly_charges.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Churn by Internet Service & Payment Method

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Internet Service
internet_churn = df.groupby("InternetService")["Churn_Binary"].mean().reset_index()
internet_churn.columns = ["InternetService", "ChurnRate"]
sns.barplot(data=internet_churn, x="InternetService", y="ChurnRate",
            palette="Reds_d", ax=axes[0])
axes[0].set_title("Churn Rate by Internet Service", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Churn Rate")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# Payment Method
pay_churn = df.groupby("PaymentMethod")["Churn_Binary"].mean().sort_values(ascending=False).reset_index()
pay_churn.columns = ["PaymentMethod", "ChurnRate"]
sns.barplot(data=pay_churn, x="ChurnRate", y="PaymentMethod",
            palette="Reds_d", ax=axes[1])
axes[1].set_title("Churn Rate by Payment Method", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Churn Rate")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout()
plt.savefig("../reports/04_service_payment_churn.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Correlation Heatmap

# %%
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap — Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/05_correlation_heatmap.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Key Insights Summary

# %%
print("=" * 55)
print("KEY INSIGHTS FROM EDA")
print("=" * 55)

insights = [
    f"Overall churn rate: {df['Churn_Binary'].mean():.1%}",
    f"Month-to-month churn: {df[df['Contract']=='Month-to-month']['Churn_Binary'].mean():.1%}",
    f"Two-year contract churn: {df[df['Contract']=='Two year']['Churn_Binary'].mean():.1%}",
    f"Fiber optic churn: {df[df['InternetService']=='Fiber optic']['Churn_Binary'].mean():.1%}",
    f"E-check churn: {df[df['PaymentMethod']=='Electronic check']['Churn_Binary'].mean():.1%}",
    f"Median tenure (churned): {df[df['Churn']=='Yes']['tenure'].median():.0f} months",
    f"Median tenure (retained): {df[df['Churn']=='No']['tenure'].median():.0f} months",
    f"Avg monthly charge (churned): ${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}",
]

for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")
