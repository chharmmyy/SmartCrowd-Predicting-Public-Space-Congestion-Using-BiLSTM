# app.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import os
import io

# Try to import joblib (may be missing in some envs)
try:
    import joblib
    joblib_import_error = None
except Exception as e:
    joblib = None
    joblib_import_error = e

# Do NOT import tensorflow at top-level to avoid import-time failures.
# We'll attempt to import it lazily when needed.

st.set_page_config(page_title="SmartCrowd — BiLSTM", layout="centered")
st.title("🧠 SmartCrowd — Predicting Public Space Congestion (BiLSTM)")

st.markdown(
    """
This app predicts the next crowd count using a trained BiLSTM model.
It will try to load local artifacts:
- `smartcrowd_bilstm.h5` (Keras HDF5 model)
- `scalers.pkl` (scalers saved via joblib.dump or pickle)

If imports (TensorFlow/joblib) fail, the app will show instructions to fix your `requirements.txt`,
or you can upload the model and scalers directly below.
"""
)

# recommended requirements snippet for the user
RECOMMENDED_REQUIREMENTS = """\
# recommended minimal requirements.txt for Streamlit Cloud
streamlit
tensorflow-cpu==2.11.0
scikit-learn
pandas
numpy
joblib
"""

def try_import_tensorflow():
    """Try importing tensorflow and return (tf, load_model_func, error_msg)."""
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
        return tf, load_model, None
    except Exception as e:
        return None, None, e

def load_scalers_from_path(scaler_path):
    """Load scalers with joblib if available, else fallback to pickle."""
    if not Path(scaler_path).exists():
        return None, f"Scaler file not found at: {scaler_path}"
    # Try joblib first
    if joblib is not None:
        try:
            scalers = joblib.load(scaler_path)
            return scalers, None
        except Exception as e:
            # try pickle fallback
            pass
    # Fallback to pickle
    try:
        import pickle
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        return scalers, None
    except Exception as e:
        return None, f"Failed to load scalers using joblib/pickle: {e}"

def load_artifacts(model_path="smartcrowd_bilstm.h5", scaler_path="scalers.pkl"):
    """
    Attempt to load model and scalers.
    Returns: (model, feature_scaler, target_scaler, error_message_or_None)
    """
    # Check files exist
    if not Path(model_path).exists() or not Path(scaler_path).exists():
        missing = []
        if not Path(model_path).exists():
            missing.append(model_path)
        if not Path(scaler_path).exists():
            missing.append(scaler_path)
        return None, None, None, f"Missing files: {', '.join(missing)}. Upload them below or add them to the repo."

    # Check joblib
    if joblib is None:
        return None, None, None, f"'joblib' import failed: {joblib_import_error}. Add 'joblib' to requirements.txt."

    # Import tensorflow lazily
    tf, load_model, tf_err = try_import_tensorflow()
    if tf_err is not None:
        return None, None, None, f"TensorFlow import failed: {tf_err}. Recommended requirements: 'tensorflow-cpu==2.11.0'."

    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        return None, None, None, f"Failed to load model '{model_path}': {e}"

    # Load scalers
    scalers, err = load_scalers_from_path(scaler_path)
    if err:
        return None, None, None, err

    # normalize scalers
    if isinstance(scalers, dict):
        # try common keys
        feature_scaler = scalers.get("feature_scaler") or scalers.get("scaler") or scalers.get("feature_scaler")
        target_scaler = scalers.get("target_scaler") or scalers.get("target") or scalers.get("target_scaler")
        # if dictionary contains just one scaler, assign both
        if feature_scaler is None and target_scaler is None and len(scalers) == 1:
            only = list(scalers.values())[0]
            feature_scaler = only
            target_scaler = only
    else:
        # scalers is a single scaler object (likely for univariate)
        feature_scaler = scalers
        target_scaler = scalers

    if feature_scaler is None or target_scaler is None:
        # fallback: assign the loaded object to both
        feature_scaler = scalers
        target_scaler = scalers

    return model, feature_scaler, target_scaler, None

# Try loading default artifacts from repo
model_path_default = "smartcrowd_bilstm.h5"
scaler_path_default = "scalers.pkl"
model, feature_scaler, target_scaler, load_err = load_artifacts(model_path_default, scaler_path_default)

if load_err:
    st.warning(load_err)
    with st.expander("How to fix"):
        st.markdown(
            f"""
**1)** Ensure files are in the app folder (next to `app.py`):
- `{model_path_default}`
- `{scaler_path_default}`

**2)** Make sure `requirements.txt` includes:
