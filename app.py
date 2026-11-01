
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "water_pollution_model.keras"

# Class names — keep this order the same as the training code
LABELS = [
    "Algae Pollution",
    "Chemical Pollution",
    "Mixed Pollution",
    "Natural / No Visible Pollution",
    "Plastic Pollution",
    "Polythene Pollution"
]

st.set_page_config(
    page_title="AI Water Pollution Detection",
    page_icon="🌊",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("🌊 AI-Based Lake and River Water Pollution Detection")
st.write(
    "Upload a lake or river image to identify visible pollution patterns "
    "using a deep learning model."
)

st.info("This is an experimental image-classification system.")

uploaded_file = st.file_uploader(
    "Upload a lake or river image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Water Pollution"):

        try:
            model = load_model()

            # Image preprocessing
            processed_image = image.resize((160, 160))
            image_array = np.array(
                processed_image, dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(image_array, axis=0)

            # Model prediction
            predictions = model.predict(
                image_array, verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_label = LABELS[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            st.subheader("Prediction Report")

            st.success(f"Detected Category: {predicted_label}")

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Category-wise Prediction Scores")
            for label, score in zip(LABELS, predictions):
                percentage = float(score) * 100

                st.write(f"**{label}: {percentage:.2f}%**")
                st.progress(int(percentage))

            

            st.warning(
                "Important: This model identifies visible image patterns "
                "only. It cannot measure chemical concentration, pH, "
                "dissolved oxygen, toxins, or drinking-water safety. "
                "Its predictions are experimental and must not replace "
                "laboratory water testing."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "AI-Based Lake and River Water Pollution Detection | "
    "Deep Learning Project"
)
