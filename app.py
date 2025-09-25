import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ======================
# Safe TensorFlow Import
# ======================
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError as e:
    st.error(f"❌ TensorFlow could not be imported. Error: {e}")
    st.stop()

# ======================
# Load Model & Scalers
# ======================
@st.cache_resource
def load_artifacts():
    model = load_model("smartcrowd_bilstm_saved.h5")  # your trained BiLSTM
    scaler = joblib.load("scaler.pkl")                # MinMaxScaler
    return model, scaler

model, scaler = load_artifacts()

# ======================
# Streamlit UI
# ======================
st.title("🧠 SmartCrowd: Predicting Public Space Congestion Using BiLSTM")
st.write("Upload past crowd data or enter recent counts to predict the next step congestion level.")

# Sidebar for input window
st.sidebar.header("Input Parameters")
n_steps = 24  # same as during training

# Allow manual input of last 24 counts
st.sidebar.write(f"Enter last {n_steps} crowd counts (people/hour)")
default_values = [50] * n_steps
user_values = []
for i in range(n_steps):
    val = st.sidebar.number_input(f"Hour -{n_steps - i}", min_value=0, max_value=10000, value=default_values[i])
    user_values.append(val)

# Convert to numpy array
last_window = np.array(user_values).reshape(-1, 1)

# ======================
# Prediction
# ======================
if st.button("🔮 Predict Congestion"):
    try:
        # Scale input
        scaled_input = scaler.transform(last_window).reshape((1, n_steps, 1))
        # Predict
        pred_scaled = model.predict(scaled_input)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]

        # Categorize crowd level
        if pred < 30:
            level = "🟢 Low"
        elif pred < 70:
            level = "🟡 Medium"
        else:
            level = "🔴 High"

        # Show result
        st.success(f"Predicted next hour crowd count: **{pred:.0f} people**")
        st.info(f"Crowd level: {level}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
