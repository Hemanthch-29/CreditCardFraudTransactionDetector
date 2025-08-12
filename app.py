import streamlit as st
import pandas as pd
import pickle

# --- Authentication ---
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state["authenticated"] = True
        else:
            st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("🔍 Credit Card Fraud Detection App")
st.markdown("Enter transaction details below to predict if it is fraudulent.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
transaction_type = st.selectbox("Transaction Type", ['Bank Transfer', 'ATM Withdrawal', 'Online', 'POS'])
device_type = st.selectbox("Device Type", ['Mobile', 'Laptop', 'Tablet'])
is_weekend = st.selectbox("Is Weekend?", [0, 1])
previous_fraud = st.selectbox("Previous Fraudulent Activity", [0, 1])
daily_txn_count = st.number_input("Daily Transaction Count", min_value=0)
avg_txn_7d = st.number_input("Avg Transaction Amount (7d)", min_value=0.0)
failed_txn_7d = st.number_input("Failed Transaction Count (7d)", min_value=0)
card_type = st.selectbox("Card Type", ['Amex', 'Mastercard', 'Discover', 'Visa'])
card_age = st.number_input("Card Age (months)", min_value=0)
txn_distance = st.number_input("Transaction Distance (km)", min_value=0.0)
auth_method = st.selectbox("Authentication Method", ['OTP', 'Biometric', 'Password', 'PIN'])
risk_score = st.number_input("Risk Score", min_value=0.0, max_value=1.0, step=0.01)

# Create input DataFrame
input_data = pd.DataFrame([{
    'Transaction_Amount': amount,
    'Transaction_Type': transaction_type,
    'Device_Type': device_type,
    'Is_Weekend': is_weekend,
    'Previous_Fraudulent_Activity': previous_fraud,
    'Daily_Transaction_Count': daily_txn_count,
    'Avg_Transaction_Amount_7d': avg_txn_7d,
    'Failed_Transaction_Count_7d': failed_txn_7d,
    'Card_Type': card_type,
    'Card_Age': card_age,
    'Transaction_Distance': txn_distance,
    'Authentication_Method': auth_method,
    'Risk_Score': risk_score
}])

# One-hot encoding (same as training)
input_data = pd.get_dummies(input_data)
# Align with model input columns
model_input_columns = pickle.load(open('model_columns.pkl', 'rb'))
input_data = input_data.reindex(columns=model_input_columns, fill_value=0)

# Predict
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Transaction is Not Fraudulent (Confidence: {1 - prob:.2%})")
