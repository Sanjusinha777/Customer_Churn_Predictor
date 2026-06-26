import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS (DESIGN)
# -----------------------------
st.markdown("""
<style>

/* ===== GLOBAL ===== */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}

/* Remove top clutter */
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    display: none;
}

/* ===== MAIN CARD ===== */
.main-card {
    max-width: 560px;
    margin: auto;
    padding: 32px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
}

/* ===== TITLES ===== */
h1 {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
}
h3 {
    text-align: center;
    color: #d1d5db;
    margin-bottom: 24px;
}

/* ===== LABELS ===== */
label,
div[data-testid="stWidgetLabel"] > label {
    color: #e5e7eb !important;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 6px;
}

/* ===== INPUTS ===== */
input, select {
    background: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"] {
    background: transparent !important;
    padding: 0px !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 14px;
    color: #ffffff;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
}

/* ===== RESULT ===== */
.result-box {
    margin-top: 22px;
    padding: 18px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    text-align: center;
}
.success {
    background: rgba(34,197,94,0.2);
    color: #bbf7d0;
}
.error {
    background: rgba(239,68,68,0.2);
    color: #fecaca;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===== REMOVE TOP BLANK / GREY BAR ===== */
div[data-testid="stAppViewContainer"] > section:first-child {
    padding-top: 0rem !important;
}

div[data-testid="stAppViewContainer"] header {
    display: none !important;
}

/* extra safety (Streamlit updates ke liye) */
div[data-testid="stToolbar"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# BUILD MODEL (IMPORTANT)
# -----------------------------
def build_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(11,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    return model

model = build_model()
model.load_weights("churn.weights.h5")

# -----------------------------
# LOAD SCALER
# -----------------------------
scaler = joblib.load("scaler.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h1>📉 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3>Predict whether a customer will leave the bank</h3>", unsafe_allow_html=True)

# -----------------------------
# INPUTS
# -----------------------------
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 40)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", 0.0, 300000.0, 60000.0)
num_products = st.selectbox("Number of Products", [1,2,3,4])
has_card = st.selectbox("Has Credit Card", [0,1])
is_active = st.selectbox("Is Active Member", [0,1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geo = st.selectbox("Geography", ["France","Germany","Spain"])
gender = st.selectbox("Gender", ["Female","Male"])

geo_germany = 1 if geo=="Germany" else 0
geo_spain = 1 if geo=="Spain" else 0
gender_male = 1 if gender=="Male" else 0

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Churn"):
    X = pd.DataFrame([[credit_score, age, tenure, balance, num_products,
                       has_card, is_active, salary,
                       geo_germany, geo_spain, gender_male]])

    Xs = scaler.transform(X)
    prob = model.predict(Xs)[0][0]

    if prob > 0.5:
        st.markdown(
            f"<div class='result-box error'>⚠️ Customer is likely to CHURN<br>Probability: {prob:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-box success'>✅ Customer will NOT churn<br>Probability: {prob:.2f}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
