# app.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# Try importing joblib
try:
    import joblib
    joblib_import_error = None
except Exception as e:
    joblib = None
    joblib_import_error = str(e)

# Safe TensorFlow import
def try_import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        return tf, load_model, None
    except Exception as e:
        return None, None, str(e)

st.set_page_config(page_title="SmartCrowd — BiLSTM", layout="centered")
st.title("🧠 SmartCrowd — Predicting Public Space Congestion (BiLSTM)")

st.markdown(
    """
This app predicts the next crowd count using a trained BiLSTM model.

Required files:
- `smartcrowd_bilstm.h5` (trained model)
- `scalers.pkl` (saved scaler(s))

If they are missing, upload them below.
"""
)

# --- Helper to load scalers ---
def load_scalers_from_path(scaler_path):
    if not Path(scaler_path).exists():
        return None, "Scaler file not found."
    if joblib is not None:
        try:
            return joblib.load(scaler_path), None
        except Exception:
            pass
    try:
        import pickle
        with open(scaler_path, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, "Failed to load scalers: " + str(e)

# --- Load artifacts ---
def load_artifacts(model_path="smartcrowd_bilstm.h5", scaler_path="scalers.pkl"):
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        return None, None, None, "Model or scaler file missing."

    if joblib is None:
        return None, None, None, "joblib import failed: " + joblib_import_error

    tf, load_model, tf_err = try_import_tensorflow()
    if tf_err is not None:
        return None, None, None, "TensorFlow import failed: " + tf_err

    try:
        model = load_model(model_path)
    except Exception as e:
        return None, None, None, "Failed to load model: " + str(e)

    scalers, err = load_scalers_from_path(scaler_path)
    if err:
        return None, None, None, err

    if isinstance(scalers, dict):
        feature_scaler = scalers.get("feature_scaler") or scalers.get("scaler")
        target_scaler = scalers.get("target_scaler") or feature_scaler
    else:
        feature_scaler = target_scaler = scalers

    return model, feature_scaler, target_scaler, None

# --- Try loading defaults ---
model, feature_scaler, target_scaler, load_err = load_artifacts()

if load_err:
    st.warning(load_err)
    st.info(
        "Make sure you included `smartcrowd_bilstm.h5` and `scalers.pkl` "
        "in your repo or upload them below."
    )

# --- Upload fallback ---
uploaded_model = st.file_uploader("Upload model (.h5)", type=["h5"])
uploaded_scaler = st.file_uploader("Upload scalers (.pkl)", type=["pkl", "joblib"])

if uploaded_model and uploaded_scaler:
    with open("uploaded_model.h5", "wb") as f:
        f.write(uploaded_model.getbuffer())
    with open("uploaded_scalers.pkl", "wb") as f:
        f.write(uploaded_scaler.getbuffer())
    model, feature_scaler, target_scaler, load_err = load_artifacts(
        "uploaded_model.h5", "uploaded_scalers.pkl"
    )
    if load_err:
        st.error(load_err)
    else:
        st.success("Uploaded model and scalers loaded successfully.")

if model is None:
    st.stop()

# --- Get input shape ---
try:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    _, n_steps, n_features = input_shape
except Exception:
    n_steps, n_features = 24, 1

st.sidebar.write(f"Model expects: timesteps={n_steps}, features={n_features}")

# --- Prediction UI ---
if n_features == 1:
    st.subheader(f"Enter last {n_steps} values")
    values = []
    cols = st.columns(4)
    for i in range(n_steps):
        with cols[i % 4]:
            val = st.number_input(f"t-{n_steps-i}", min_value=0, value=50, key=f"v{i}")
            values.append(val)

    if st.button("Predict"):
        arr = np.array(values).reshape(-1, 1)
        scaled = feature_scaler.transform(arr)
        X = scaled.reshape((1, n_steps, n_features))
        pred_scaled = model.predict(X)
        try:
            pred = target_scaler.inverse_transform(pred_scaled)[0, 0]
        except Exception:
            pred = float(pred_scaled.ravel()[0])
        st.success(f"Predicted next crowd count: {pred:.0f}")
else:
    st.subheader("Upload CSV with last window of values")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        arr = df.values.astype(float)
        scaled = feature_scaler.transform(arr)
        X = scaled.reshape((1, n_steps, n_features))
        pred_scaled = model.predict(X)
        try:
            pred = target_scaler.inverse_transform(pred_scaled)[0, 0]
        except Exception:
            pred = float(pred_scaled.ravel()[0])
        st.success(f"Predicted next crowd count: {pred:.0f}")
