import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# -----------------------------
# Load model + scalers
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("smartcrowd_bilstm.h5")
    scalers = joblib.load("scalers.pkl")
    feature_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_resources()

# -----------------------------
# Streamlit App
# -----------------------------
st.title("📊 SmartCrowd: Predicting Public Space Congestion Using BiLSTM")

st.markdown("""
Upload your recent crowd data (CSV with a `timestamp` column and features).  
The app will predict the **next congestion level**.
""")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

    st.write("### 📝 Uploaded Data (first rows)", df.head())

    if "timestamp" not in df.columns:
        st.error("❌ CSV must contain a 'timestamp' column.")
    else:
        # Drop timestamp for prediction
        features = df.drop(columns=['timestamp'])

        # Scale features
        scaled_features = feature_scaler.transform(features)

        # Reshape for BiLSTM: (1, timesteps, features)
        X_input = np.expand_dims(scaled_features, axis=0)

        # Predict
        pred_scaled = model.predict(X_input)
        pred = target_scaler.inverse_transform(pred_scaled)

        # Show result
        st.success(f"📈 Predicted Crowd Count: **{int(pred[0][0])}** people")

        # Convert to congestion level
        if pred[0][0] < 50:
            level = "🟢 Low"
        elif pred[0][0] < 150:
            level = "🟡 Medium"
        else:
            level = "🔴 High"

        st.info(f"Predicted Congestion Level: **{level}**")
