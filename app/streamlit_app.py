import streamlit as st
import numpy as np

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.markdown("""
    <style>

        #MainMenu {visibility: hidden;}

        footer {visibility: hidden;}

        header {visibility: hidden;}

        .st-emotion-cache-scp8yw {
            display: none !important;
        }

        .st-emotion-cache-1s6ol36 {
            padding-right: 0px !important;
        }

        .st-emotion-cache-14vh5up button {
            display: none !important;
        }

        [data-testid="stToolbar"] {
            visibility: hidden;
        }

        .block-container.st-emotion-cache-1w723zb {
            padding-top: 1rem !important;
            padding-right: 1rem !important;
            padding-left: 1rem !important;
            padding-bottom: 1rem !important;
            margin-top: 0px !important;
            margin-right: 0px !important;
            margin-left: 0px !important;
            max-width: 90% !important;
        }

        .stVerticalBlock.st-emotion-cache-tn0cau {
            gap: 0.5rem !important;
        }

        h1, h3 {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }

        .element-container.st-emotion-cache-3pwa5w {
            margin-bottom: 0.5rem !important;
        }

        [data-testid="stWidgetLabel"] p {
            margin-bottom: 0.25rem !important;
        }

        .stHorizontalBlock.st-emotion-cache-1permvm {
            gap: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔄 Customer Churn Predictor")
st.markdown("Fill in the customer details and click **Predict** to see the churn risk.")
st.markdown("---")
# --- Input Form ---
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet        = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

with col2:
    monthly_charges = st.number_input("Monthly Charges (₹)", 0.0, 2000.0, 1200.0, step=50.0)
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    senior_citizen  = st.checkbox("Senior Citizen")

st.markdown("---")
if st.button("🔍 Predict Churn", use_container_width=True):
    score = 0
    if contract == "Month-to-month":        score += 3
    if payment_method == "Electronic check": score += 2
    if internet == "Fiber optic":            score += 1
    if tenure < 12:                          score += 2
    if monthly_charges > 1500:               score += 1
    if senior_citizen:                       score += 1

    churn_prob = round(min(0.10 + score * 0.09, 0.95), 2)

    st.markdown("---")
    st.subheader("Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Churn Probability", f"{churn_prob:.0%}")
    col_b.metric("Prediction", "⚠️ Will Churn" if churn_prob >= 0.45 else "✅ Will Stay")
    col_c.metric("Risk Level",
                 "🔴 High"   if churn_prob >= 0.6 else
                 "🟡 Medium" if churn_prob >= 0.3 else
                 "🟢 Low")

    st.progress(churn_prob)

    st.subheader("Key Risk Factors")
    risks = []
    if contract        == "Month-to-month":   risks.append("Month-to-month contract")
    if payment_method  == "Electronic check": risks.append("Electronic check payment")
    if internet        == "Fiber optic":      risks.append("Fiber optic internet user")
    if tenure          <  12:                 risks.append("New customer (tenure < 12 months)")
    if monthly_charges >  1500:               risks.append("High monthly charges (> ₹1,500)")

    if risks:
        for r in risks:
            st.warning(f"⚠️ {r}")
    else:
        st.success("✅ No major risk factors found.")
    st.subheader("Recommendation")
    if churn_prob >= 0.6:
        st.error("🔴 High Priority — Offer a discount or upgrade to a yearly contract immediately.")
    elif churn_prob >= 0.3:
        st.warning("🟡 Medium Priority — Send a satisfaction survey and a loyalty offer.")
    else:
        st.success("🟢 Low Risk — Continue normal engagement. Consider upselling services.")