# app.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Try to import joblib
try:
    import joblib
    joblib_import_error = None
except Exception as e:
    joblib = None
    joblib_import_error = e

# Do NOT import tensorflow at top-level
def try_import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        return tf, load_model, None
    except Exception as e:
        return None, None, e

st.set_page_config(page_title="SmartCrowd — BiLSTM", layout="centered")
st.title("🧠 SmartCrowd — Predicting Public Space Congestion (BiLSTM)")

st.markdown(
    """
This app predicts the next crowd count using a trained BiLSTM model.

Expected files:
- `smartcrowd_bilstm.h5` (model)
- `scalers.pkl` (scalers with feature + target)

If these are missing or imports fail, you can upload them below.
"""
)

RECOMMENDED_REQUIREMENTS = """\
streamlit
tensorflow-cpu==2.11.0
scikit-learn
pandas
numpy
joblib
"""

def load_scalers_from_path(scaler_path):
    if not Path(scaler_path).exists():
        return None, f"Scaler file not found at: {scaler_path}"
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
        return None, f"Failed to load scalers: {e}"

def load_artifacts(model_path="smartcrowd_bilstm.h5", scaler_path="scalers.pkl"):
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        return None, None, None, "Missing files. Upload them below."
    if joblib is None:
        return None, None, None, f"joblib import failed: {joblib_import_error}"

    tf, load_model, tf_err = try_import_tensorflow()
    if tf_err is not None:
        return None, None, None, f"TensorFlow import failed: {tf_err}"

    try:
        model = load_model(model_path)
    except Exception as e:
        return None, None, None, f"Failed to load mode
