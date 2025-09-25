import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model & scalers
model = load_model("smartcrowd_bilstm_saved")
feature_scaler = joblib.load("feature_scaler.save")
target_scaler = joblib.load("target_scaler.save")

SEQ_LEN = 24  # must match training
FEATURES = feature_scaler.feature_names_in_

st.title("🧠 SmartCrowd: BiLSTM Public Space Congestion Predictor")

st.write("This app predicts future crowd levels (people count) using past data and contextual features.")

# Input options (demo: last 24 hours of crowd counts)
crowd_input = []
for i in range(SEQ_LEN):
    crowd_input.append(st.number_input(f"Crowd count at t-{SEQ_LEN-i}", min_value=0, value=50))

temperature = st.number_input("Current Temperature (°C)", value=25)
pollution = st.number_input("Pollution Index", value=50)
is_holiday = st.selectbox("Holiday?", [0, 1])
special_event = st.selectbox("Special Event?", [0, 1])

# Build feature array
last_window = pd.DataFrame([{
    "crowd_count": c,
    "temperature": temperature,
    "pollution_index": pollution,
    "is_holiday": is_holiday,
    "special_event": special_event
} for c in crowd_input])

# Scale and reshape
scaled = feature_scaler.transform(last_window[FEATURES])
X = scaled.reshape((1, SEQ_LEN, len(FEATURES)))

# Prediction
if st.button("Predict Next Hour Crowd"):
    pred_scaled = model.predict(X)
    pred = target_scaler.inverse_transform(pred_scaled)[0, 0]

    st.success(f"Predicted crowd count: {pred:.0f}")

    # Convert to Low/Medium/High
    if pred < 30:
        level = "Low 🟢"
    elif pred < 70:
        level = "Medium 🟡"
    else:
        level = "High 🔴"
    st.write("Congestion Level:", level)
