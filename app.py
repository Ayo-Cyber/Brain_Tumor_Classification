import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from src.utils import generate_explanation
from src.utils import CLASS_NAMES, grad_cam_overlay , load_and_preprocess_image , load_selected_model
import tempfile

# Page Configuration
st.set_page_config(page_title="AI Neuro Diagnosis", page_icon="🩺", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Settings")
    st.selectbox("Select Model", ["ResNet50 (Coming Soon)", "VGG16 (Coming Soon)", "InceptionV3 (Coming Soon)"], index=0, disabled=True)
    dark_mode = st.toggle("Dark Mode")

# Dark mode style
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #222; color: white; }
        .stTextInput input { background-color: #444; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Header
st.title("🧠 AI Medical Image Diagnosis For NeuroDegenerative Diseases")
st.write("Upload a brain scan image (JPG, PNG) to get a prediction of possible neurodegenerative disease.")
st.markdown("_____")

# Load model
model = load_selected_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    original_image = Image.open(temp_path)
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_image, caption="Uploaded Image", use_container_width=True)

    img_array_preprocessed, img_for_overlay = load_and_preprocess_image(temp_path)

    if model is None:
        st.warning("⚠️ Model could not be loaded. Please try again later.")
    else:
        try:
            predictions = model.predict(img_array_preprocessed)
            confidence_scores = predictions[0]
            predicted_index = np.argmax(confidence_scores)
            predicted_class = CLASS_NAMES[predicted_index]
            predicted_confidence = confidence_scores[predicted_index] * 100

            with col2:
                st.subheader("🧾 Diagnosis Result")
                st.write(f"**Prediction:** {predicted_class}")
                st.write(f"**Confidence:** {predicted_confidence:.2f}%")

                with st.expander("🔍 See All Class Confidence Scores"):
                    for idx, label in enumerate(CLASS_NAMES):
                        st.write(f"{label}: {confidence_scores[idx]*100:.2f}%")

                grad_cam_button = st.button("Generate Grad-CAM Explanation")

                if grad_cam_button:
                    heatmap_overlay = grad_cam_overlay(model, img_for_overlay, layer_name="conv5_block3_out", alpha=0.4)

                    st.markdown("### 🔍 Grad-CAM Visual Explanation + Medical Insight")
                    cam_col, expl_col = st.columns(2)

                    with cam_col:
                        st.image(heatmap_overlay, caption="Grad-CAM Overlay", use_container_width=True)

                    with expl_col:
                        st.info(generate_explanation(predicted_class, predicted_confidence, confidence_scores))

        except Exception as e:
            st.error(f"🚨 Prediction failed: {e}")

st.markdown("_____")
st.markdown("Made with ❤️ by Atunrase Ayomide (University Of Lagos) .")
