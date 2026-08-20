import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from datetime import date

REFERENCE_DATE = pd.Timestamp("2025-03-11")  # same reference used at training time


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"], errors="coerce")

    df["age_invalid"] = ((df["age"] < 18) | (df["age"] > 100)).astype(int)
    df.loc[df["age_invalid"] == 1, "age"] = np.nan

    df["chronology_invalid"] = (df["signup_date"] > df["last_purchase_date"]).astype(int)

    df["tenure_days"] = (df["last_purchase_date"] - df["signup_date"]).dt.days
    df.loc[df["tenure_days"] < 0, "tenure_days"] = np.nan

    df["recency_days"] = (REFERENCE_DATE - df["last_purchase_date"]).dt.days
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["last_purchase_month"] = df["last_purchase_date"].dt.month

    df["spend_per_visit"] = df["total_spent"] / df["total_visits"].replace(0, np.nan)
    df["session_depth"] = df["avg_session_time"] * df["pages_per_session"]
    df["spend_to_marketing_ratio"] = (
        df["total_spent"] / df["marketing_spend_per_user"].replace(0, np.nan)
    )
    df["email_engagement"] = df["email_open_rate"] * df["email_click_rate"]
    df["lifetime_value_per_tenure_month"] = (
        df["lifetime_value"] / (df["tenure_days"] / 30.44).clip(lower=1)
    )

    df = df.drop(
        columns=[
            "customer_id", "signup_date", "last_purchase_date", "churn",
            "last_3_month_purchase_freq", "support_tickets", "refund_requested",
            "satisfaction_score", "nps_score",
        ],
        errors="ignore",
    )
    return df

MODEL_PATH = Path(__file__).parent / "models" / "final_model.joblib"
BEST_THRESHOLD = 0.3413225316528488  # from the notebook's threshold-tuning step (cell 98)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer's profile to estimate their probability of churning.")

with st.form("churn_form"):
    st.subheader("Demographics")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female", "Other"])
    age = c2.number_input("Age", min_value=0, max_value=100, value=35)
    country = c3.selectbox("Country", ["India", "Germany", "USA", "UK", "Bangladesh"])
    city = st.selectbox("City", ["Berlin", "Mumbai", "London", "Hamburg", "New York", "Delhi", "Dhaka"])

    st.subheader("Account")
    c1, c2, c3 = st.columns(3)
    signup_date = c1.date_input("Signup date", value=date(2023, 6, 1))
    last_purchase_date = c2.date_input("Last purchase date", value=date(2024, 12, 1))
    subscription_type = c3.selectbox("Subscription type", ["Monthly", "Annual"])

    c1, c2, c3 = st.columns(3)
    acquisition_channel = c1.selectbox(
        "Acquisition channel", ["Email", "Organic", "Facebook Ads", "Referral", "Google Ads"]
    )
    device_type = c2.selectbox("Device type", ["Tablet", "Desktop", "Mobile"])
    is_premium_user = c3.selectbox("Premium user?", ["No", "Yes"]) == "Yes"

    st.subheader("Engagement")
    c1, c2, c3 = st.columns(3)
    total_visits = c1.number_input("Total visits", min_value=0, value=15)
    avg_session_time = c2.number_input("Avg. session time (min)", min_value=0.0, value=8.0)
    pages_per_session = c3.number_input("Pages per session", min_value=0.0, value=4.0)

    c1, c2 = st.columns(2)
    email_open_rate = c1.slider("Email open rate", 0.0, 1.0, 0.5)
    email_click_rate = c2.slider("Email click rate", 0.0, 1.0, 0.25)

    st.subheader("Spending")
    c1, c2, c3 = st.columns(3)
    total_spent = c1.number_input("Total spent ($)", min_value=0.0, value=500.0)
    avg_order_value = c2.number_input("Avg. order value ($)", min_value=0.0, value=60.0)
    lifetime_value = c3.number_input("Lifetime value ($)", min_value=0.0, value=1200.0)

    c1, c2, c3 = st.columns(3)
    discount_used = c1.selectbox("Used a discount?", ["No", "Yes"]) == "Yes"
    coupon_code = c2.selectbox("Coupon code", ["None", "NEW20", "SALE15", "REF10"])
    payment_method = c3.selectbox("Payment method", ["Card", "PayPal", "UPI", "BKash", "SEPA"])

    st.subheader("Service")
    c1, c2 = st.columns(2)
    delivery_delay_days = c1.number_input("Avg. delivery delay (days)", min_value=0, value=3)
    marketing_spend_per_user = c2.number_input("Marketing spend per user ($)", min_value=0.0, value=17.6)

    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    row = pd.DataFrame([{
        "customer_id": 0,
        "gender": gender,
        "age": age,
        "country": country,
        "city": city,
        "signup_date": signup_date.isoformat(),
        "last_purchase_date": last_purchase_date.isoformat(),
        "acquisition_channel": acquisition_channel,
        "device_type": device_type,
        "subscription_type": subscription_type,
        "is_premium_user": int(is_premium_user),
        "total_visits": total_visits,
        "avg_session_time": avg_session_time,
        "pages_per_session": pages_per_session,
        "email_open_rate": email_open_rate,
        "email_click_rate": email_click_rate,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "discount_used": int(discount_used),
        "coupon_code": None if coupon_code == "None" else coupon_code,
        "delivery_delay_days": delivery_delay_days,
        "payment_method": payment_method,
        "marketing_spend_per_user": marketing_spend_per_user,
        "lifetime_value": lifetime_value,
    }])

    X = engineer_features(row)
    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability >= BEST_THRESHOLD)

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High churn risk — probability: {probability:.1%}")
    else:
        st.success(f"✅ Low churn risk — probability: {probability:.1%}")

    st.progress(min(float(probability), 1.0))
    st.caption(f"Decision threshold: {BEST_THRESHOLD:.3f} (tuned for best F1 on validation data)")