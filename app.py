# app.py
import os
import json
import numpy as np
import joblib
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import xgboost as xgb

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(layout="wide", page_title="Fracture Detection — CNN vs XGBoost")

# ---------------------------------
# Paths (adjust if needed)
# ---------------------------------
KERAS_MODEL_PATH = "fracture_detection_model.h5"
FEATURE_EXTRACTOR_PATH = "feature_extractor.keras"
XGB_META = "xgb_model_meta.joblib"
CLASS_INDICES_PATH = "class_indices.json"

# ---------------------------------
# Load models and mappings
# ---------------------------------
@st.cache_resource
def load_resources():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)

    meta = joblib.load(XGB_META)
    booster = xgb.Booster()
    booster.load_model(meta["booster_path"])

    return idx_to_class, keras_model, feature_extractor, booster

try:
    idx_to_class, keras_model, feature_extractor, booster = load_resources()
except Exception as e:
    st.error(f"❌ Failed to load models/resources: {e}")
    st.stop()

# ---------------------------------
# UI
# ---------------------------------
st.title("Fracture Detection — CNN vs XGBoost")

uploaded = st.file_uploader("📤 Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Please upload an image to begin inference.")
    st.stop()

# show input image
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Input X-ray", width=300)

# ---------------------------------
# Helper: CNN prediction
# ---------------------------------
def keras_predict(img_pil):
    img_resized = img_pil.resize((128, 128))
    x = np.array(img_resized).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)
    prob = float(keras_model.predict(x, verbose=0)[0].ravel()[0])
    pred = int(prob > 0.5)
    return pred, prob

# ---------------------------------
# Helper: XGBoost prediction
# ---------------------------------
def xgb_predict(img_pil):
    img_resized = img_pil.resize((224, 224))
    x = img_to_array(img_resized)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)
    features = feature_extractor.predict(x, verbose=0)
    dfeat = xgb.DMatrix(features)
    prob = float(booster.predict(dfeat)[0])
    pred = int(prob > 0.5)
    return pred, prob

# ---------------------------------
# Run both predictions
# ---------------------------------
keras_pred, keras_prob = keras_predict(image)
xgb_pred, xgb_prob = xgb_predict(image)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 CNN (Convolutional Neural Network)")
    st.write("Prediction:", idx_to_class[keras_pred])
    st.write(f"Probability (fracture): **{keras_prob:.4f}**")

with col2:
    st.subheader("🌲 XGBoost (on deep features)")
    st.write("Prediction:", idx_to_class[xgb_pred])
    st.write(f"Probability (fracture): **{xgb_prob:.4f}**")

# ---------------------------------
# Footer / Notes
# ---------------------------------
st.markdown("---")
st.caption("⚠️ These predictions are experimental and for educational use only. Validate with domain experts before clinical or diagnostic application.")
