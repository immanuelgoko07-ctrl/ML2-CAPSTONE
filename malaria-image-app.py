# ==========================================
# Malaria Mosquito Species Classification
# ==========================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Mosquito Species Classifier",
    layout="wide"
)

st.title("🦟 Malaria Mosquito Species Classification System")

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "mosquito_species_model.h5"  # Change if needed
IMAGE_SIZE = 224  # Change to match training size

CLASS_NAMES = [
    "Aedes aegypti",
    "Anopheles gambiae",
    "Culex quinquefasciatus",
    "Aedes albopictus",
    "Anopheles arabiensis",
    "Culex pipiens"
]

# ==========================================
# ROBUST MODEL LOADER
# ==========================================

def load_trained_model(model_path: str):
    """
    Loads the trained Keras model safely.
    Returns:
        model if successful
        None if failed
    """

    # Check if file exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at path: {model_path}")
        return None

    try:
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully.")
        return model

    except Exception as e:
        st.error("❌ Error while loading the model.")
        st.error(str(e))
        return None


# Cache model so it loads once
@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


model = get_model()

# Stop app if model failed
if model is None:
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Upload & Predict", "Model Details"]
)

# ==========================================
# HOME
# ==========================================

if menu == "Home":

    st.markdown("""
    ### System Overview

    This AI system classifies mosquito images into 6 species relevant to malaria research and vector surveillance.

    **Pipeline:**

    Image → Resize → Normalize → CNN → Softmax → Prediction
    """)

# ==========================================
# PREDICTION PAGE
# ==========================================

elif menu == "Upload & Predict":

    st.header("Upload Mosquito Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.write("Processed Image Shape:", img_array.shape)

        # Prediction
        try:
            predictions = model.predict(img_array)
            probabilities = predictions[0]

            predicted_index = np.argmax(probabilities)
            predicted_species = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index] * 100

            st.success(f"🎯 Predicted Species: {predicted_species}")
            st.write(f"Confidence: {confidence:.2f}%")

            # Probability chart
            prob_df = pd.DataFrame({
                "Species": CLASS_NAMES,
                "Probability": probabilities
            })

            fig, ax = plt.subplots()
            ax.barh(prob_df["Species"], prob_df["Probability"])
            ax.set_xlabel("Probability")
            ax.set_xlim([0, 1])
            st.pyplot(fig)

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.error(str(e))

# ==========================================
# MODEL DETAILS
# ==========================================

elif menu == "Model Details":

    st.header("Model Architecture Summary")

    try:
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary_text = "\n".join(summary_lines)
        st.text(summary_text)
    except Exception as e:
        st.error("Unable to display model summary.")
        st.error(str(e))
