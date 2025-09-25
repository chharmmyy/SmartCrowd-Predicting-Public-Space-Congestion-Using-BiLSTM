import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# === Load model & scalers ===
@st.cache_resource
def load_artifacts():
    model = load_model("smartcrowd_bilstm_saved.keras")
    feature_scaler = joblib.load("feature_scaler.pkl")
    target_scaler = joblib.load("target_scaler.pkl")
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_artifacts()

st.title("🚦 SmartCrowd: Predicting Public Space Congestion Using BiLSTM")
st.write("Predict the next hour's crowd level using past 24 hours of data.")

# === User input ===
st.subheader("📊 Enter Last 24 Hours Crowd Counts")
last_counts = []
for i in range(24):
    val = st.number_input(f"Hour -{24-i}", min_value=0, step=1, value=50)
    last_counts.append(val)

# === Prediction ===
if st.button("🔮 Predict Next Hour"):
    X = np.array(last_counts).reshape(-1, 1)
    X_scaled = feature_scaler.transform(X)
    X_scaled = X_scaled.reshape((1, 24, 1))

    y_pred = model.predict(X_scaled)
    y_pred_inv = target_scaler.inverse_transform(y_pred)[0, 0]

    # Categorize
    if y_pred_inv < 30:
        category = "🟢 Low"
    elif y_pred_inv < 70:
        category = "🟡 Medium"
    else:
        category = "🔴 High"

    st.success(f"Predicted Crowd Count: **{y_pred_inv:.0f}** → {category}")
