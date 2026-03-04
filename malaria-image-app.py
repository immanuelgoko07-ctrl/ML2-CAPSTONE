# ==========================================
# Malaria Mosquito Species Classification App
# ==========================================

import streamlit as st
import numpy as np

from PIL import Image
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Malaria Mosquito Species Classifier",
    layout="wide"
)

st.title("🦟 Malaria Mosquito Species Classification System")

# ==========================================
# CLASS LABELS (EDIT TO MATCH YOUR TRAINING ORDER)
# ==========================================

CLASS_NAMES = [
    "Aedes aegypti",
    "Anopheles gambiae",
    "Culex quinquefasciatus",
    "Aedes albopictus",
    "Anopheles arabiensis",
    "Culex pipiens"
]

IMAGE_SIZE = 224  # Change if your model uses different size


# ==========================================
# LOAD MODEL WITH ERROR HANDLING
# ==========================================
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("mosquito_species_model.h5")
        return model
    except Exception as e:
        st.error("❌ Model failed to load.")
        st.error(str(e))
        return None

model = load_trained_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully.")


# ==========================================
# SIDEBAR
# ==========================================

menu = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Upload & Predict", "Model Info"]
)

# ==========================================
# HOME PAGE
# ==========================================

if menu == "Home":

    st.markdown("""
    This system classifies mosquito images into 6 species relevant to malaria and vector surveillance.

    ### Pipeline:
    Image → Resize → Normalize → CNN → Softmax → Species Prediction
    """)

# ==========================================
# UPLOAD & PREDICT PAGE
# ==========================================

elif menu == "Upload & Predict":

    st.header("Upload Mosquito Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.write("Image shape after preprocessing:", img_array.shape)

        # Predict
        try:
            prediction = model.predict(img_array)
            probabilities = prediction[0]

            predicted_index = np.argmax(probabilities)
            predicted_species = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index] * 100

            st.success(f"🎯 Predicted Species: {predicted_species}")
            st.write(f"Confidence: {confidence:.2f}%")

            # Probability Chart
            st.subheader("Prediction Probabilities")

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
# MODEL INFO PAGE
# ==========================================

elif menu == "Model Info":

    st.header("Model Architecture Summary")

    try:
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        summary_string = "\n".join(stringlist)
        st.text(summary_string)
    except Exception as e:
        st.error("Could not display model summary.")
        st.error(str(e))





