# ================================
# Mosquito Species Classification
# System Architecture Web App
# ================================

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Mosquito Species Classification System",
    layout="wide"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_trained_model():
    model = load_model("mosquito_species_model.h5")
    return model

# ================================
# CLASS LABELS (EDIT TO MATCH YOUR DATASET)
# ================================
CLASS_NAMES = [
    "Species_1",
    "Species_2",
    "Species_3",
    "Species_4",
    "Species_5",
    "Species_6"
]

# ================================
# SIDEBAR NAVIGATION
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "System Overview",
        "Data Preparation",
        "Model Architecture",
        "Make Prediction",
        "Model Evaluation"
    ]
)

# ================================
# SYSTEM OVERVIEW
# ================================
if page == "System Overview":

    st.title("🦟 Mosquito Species Image Classification System")

    st.markdown("""
    ## Project Objective
    This system performs **multi-class classification** of mosquito images 
    into **6 distinct species** using a deep learning model built in Python.

    ### High-Level Pipeline
    1. Image Input
    2. Preprocessing & Augmentation
    3. CNN Backbone Feature Extraction
    4. Classification Head
    5. Softmax Probability Output
    """)

    st.markdown("---")

    st.subheader("System Architecture Flow")

    st.code("""
    Raw Image
        ↓
    Resize & Normalize
        ↓
    Convolutional Neural Network (Backbone)
        ↓
    Global Average Pooling
        ↓
    Dense Layers
        ↓
    Softmax (6 Classes)
        ↓
    Predicted Species
    """)

# ================================
# DATA PREPARATION
# ================================
elif page == "Data Preparation":

    st.title("📊 Data Preparation Pipeline")

    st.markdown("""
    ### Dataset Structure
    - 6 folders representing 6 mosquito species
    - Images resized to 224x224
    - Normalized pixel values (0–1)

    ### Preprocessing Steps
    - Resize images
    - Normalize pixel values
    - Train/Validation/Test split
    - Data augmentation:
        - Rotation
        - Horizontal flipping
        - Zoom
        - Brightness adjustment
    """)

    st.subheader("Why Augmentation?")
    st.write("""
    Data augmentation increases model generalization and reduces overfitting 
    by exposing the network to varied image transformations.
    """)

# ================================
# MODEL ARCHITECTURE
# ================================
elif page == "Model Architecture":

    st.title("🧠 Deep Learning Architecture")

    st.markdown("""
    ### Backbone Network
    The model uses a Convolutional Neural Network (CNN) to extract 
    spatial features from mosquito images.

    Core Components:
    - Convolution Layers
    - ReLU Activation
    - Max Pooling
    - Batch Normalization
    - Dropout Regularization
    """)

    st.subheader("Classification Head")

    st.markdown("""
    - Global Average Pooling
    - Fully Connected Dense Layer
    - Output Layer (6 neurons)
    - Softmax Activation
    """)

    st.subheader("Model Summary")
    st.text(model.summary())

# ================================
# IMAGE PREDICTION
# ================================
elif page == "Make Prediction":

    st.title("🔍 Upload Image for Prediction")

    uploaded_file = st.file_uploader(
        "Upload a mosquito image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        probabilities = prediction[0]

        predicted_class = CLASS_NAMES[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        st.success(f"Predicted Species: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Probability Visualization
        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Species": CLASS_NAMES,
            "Probability": probabilities
        })

        fig, ax = plt.subplots()
        sns.barplot(x="Probability", y="Species", data=prob_df, ax=ax)
        st.pyplot(fig)

# ================================
# MODEL EVALUATION
# ================================
elif page == "Model Evaluation":

    st.title("📈 Model Performance Metrics")

    st.markdown("""
    Evaluation metrics used:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    """)

    st.info("""
    To display real evaluation metrics, connect this section 
    to your saved training history or evaluation results.
    """)




import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------------------
# App Configuration
# ---------------------------
st.set_page_config(
    page_title="Mosquito Species Classification",
    page_icon="🦟",
    layout="centered"
)

st.title("🦟 Mosquito Species Classification App")
st.write("Upload an image of a mosquito and the app will predict its species.")

# ---------------------------
# Load the trained model
# ---------------------------
@st.cache_resource
def load_mosquito_model(model_path):
    model = load_model(model_path)
    return model

model_path = "mosquito_species_model.h5"
model = load_mosquito_model(model_path)

# ---------------------------
# Image upload
# ---------------------------
uploaded_file = st.file_uploader("Choose a mosquito image...", type=["jpg", "jpeg", "png"])

# ---------------------------
# Prediction function
# ---------------------------
def predict_species(img, model):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if model trained on 0-1 range

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)[0]
    
    # Replace with your actual class labels
    class_labels = ["Aedes", "Anopheles", "Culex"]
    predicted_class = class_labels[class_index]
    confidence = predictions[0][class_index]
    
    return predicted_class, confidence

# ---------------------------
# Display prediction
# ---------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    species, confidence = predict_species(img, model)
    st.success(f"Predicted Species: {species}")
    st.info(f"Confidence: {confidence*100:.2f}%")




