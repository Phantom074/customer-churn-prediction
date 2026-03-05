"""
streamlit_app.py
================
Customer Churn Prediction Web App
Author: Mukul (github.com/phantom074)

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Churn Predictor", layout="centered")

# Hide all Streamlit default UI elements and reduce margins
st.markdown("""
    <style>
        /* Hide main menu */
        #MainMenu {visibility: hidden;}
        
        /* Hide footer */
        footer {visibility: hidden;}
        
        /* Hide header */
        header {visibility: hidden;}
        
        /* Hide deploy button and three dots */
        .st-emotion-cache-scp8yw {
            display: none !important;
        }
        
        /* Hide any remaining toolbar elements */
        .st-emotion-cache-1s6ol36 {
            padding-right: 0px !important;
        }
        
        /* Force hide all buttons in header */
        .st-emotion-cache-14vh5up button {
            display: none !important;
        }
        
        /* Additional cleanup for top bar */
        [data-testid="stToolbar"] {
            visibility: hidden;
        }
        
        /* Reduce main container padding */
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
        
        /* Reduce vertical block spacing */
        .stVerticalBlock.st-emotion-cache-tn0cau {
            gap: 0.5rem !important;
        }
        
        /* Reduce heading margins */
        h1, h3 {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Reduce element container margins */
        .element-container.st-emotion-cache-3pwa5w {
            margin-bottom: 0.5rem !important;
        }
        
        /* Minimize widget label spacing */
        [data-testid="stWidgetLabel"] p {
            margin-bottom: 0.25rem !important;
        }
        
        /* Reduce horizontal block gap */
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

# --- Predict Button ---
if st.button("🔍 Predict Churn", use_container_width=True):

    # Simple scoring logic based on known churn patterns from EDA
    score = 0
    if contract == "Month-to-month":        score += 3   # highest churn contract
    if payment_method == "Electronic check": score += 2  # highest churn payment
    if internet == "Fiber optic":            score += 1  # fiber users churn more
    if tenure < 12:                          score += 2  # new customers churn most
    if monthly_charges > 1500:               score += 1  # high charges = churn risk (>₹1500)
    if senior_citizen:                       score += 1  # seniors churn slightly more

    # Convert score to probability (max score = 10)
    churn_prob = round(min(0.10 + score * 0.09, 0.95), 2)

    # --- Show Results ---
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

    # --- Risk Factors ---
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

    # --- Business Recommendation ---
    st.subheader("Recommendation")
    if churn_prob >= 0.6:
        st.error("🔴 High Priority — Offer a discount or upgrade to a yearly contract immediately.")
    elif churn_prob >= 0.3:
        st.warning("🟡 Medium Priority — Send a satisfaction survey and a loyalty offer.")
    else:
        st.success("🟢 Low Risk — Continue normal engagement. Consider upselling services.")