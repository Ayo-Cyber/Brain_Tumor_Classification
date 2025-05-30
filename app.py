import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from src.utils import generate_explanation , CLASS_NAMES, grad_cam_overlay , load_and_preprocess_image , load_selected_model
import tempfile
import os

# Page Configuration
st.set_page_config(page_title="AI Neuro Diagnosis", page_icon="🩺", layout="wide")

# Initialize session state variables if they don't exist
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None
if 'img_array_preprocessed' not in st.session_state:
    st.session_state.img_array_preprocessed = None
if 'img_for_overlay' not in st.session_state:
    st.session_state.img_for_overlay = None
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'predicted_confidence' not in st.session_state:
    st.session_state.predicted_confidence = None
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home" , "Image Diagnosis"])
    dark_mode = st.toggle("Dark Mode")

# Dark mode style
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #222; color: white; }
        .stTextInput input { background-color: #444; color: white; }
        </style>
    """, unsafe_allow_html=True)

if page == "Home":
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

        # Store the temp_path in session state for use in Image Diagnosis page
        st.session_state.temp_path = temp_path

        original_image = Image.open(temp_path)
        # Store the original image in session state
        st.session_state.original_image = original_image
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image, caption="Uploaded Image", use_container_width=True)

        img_array_preprocessed, img_for_overlay = load_and_preprocess_image(temp_path)

        # Store processed images in session state
        st.session_state.img_array_preprocessed = img_array_preprocessed
        st.session_state.img_for_overlay = img_for_overlay

        if model is None:
            st.warning("⚠️ Model could not be loaded. Please try again later.")
        else:
            try:
                predictions = model.predict(img_array_preprocessed)
                confidence_scores = predictions[0]
                predicted_index = np.argmax(confidence_scores)
                predicted_class = CLASS_NAMES[predicted_index]
                predicted_confidence = confidence_scores[predicted_index] * 100

                # Store prediction results in session state
                st.session_state.predictions = predictions
                st.session_state.confidence_scores = confidence_scores
                st.session_state.predicted_class = predicted_class
                st.session_state.predicted_confidence = predicted_confidence

                with col2:
                    st.subheader("🧾 Diagnosis Result")
                    st.write(f"**Prediction:** {predicted_class}")
                    st.write(f"**Confidence:** {predicted_confidence:.2f}%")
                    # stoggle("🔍 See All Class Confidence Scores", 
                    #         "\n".join([f"{label}: {confidence_scores[idx]*100:.2f}%" for idx, label in enumerate(CLASS_NAMES)]),)

                    with st.expander("🔍 See All Class Confidence Scores"):
                        for idx, label in enumerate(CLASS_NAMES):
                            st.write(f"{label}: {confidence_scores[idx]*100:.2f}%")

            except Exception as e:
                st.error(f"🚨 Prediction failed: {e}")
    
    # Show current session state (for debugging/info)
    elif st.session_state.original_image is not None:
        st.info("📷 You have an image already uploaded. You can go to Image Diagnosis page for Grad-CAM analysis.")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.original_image, caption="Previously Uploaded Image", use_container_width=True)
        with col2:
            if st.session_state.predicted_class:
                st.subheader("🧾 Previous Diagnosis Result")
                st.write(f"**Prediction:** {st.session_state.predicted_class}")
                st.write(f"**Confidence:** {st.session_state.predicted_confidence:.2f}%")

                with st.expander("🔍 See All Class Confidence Scores"):
                        for idx, label in enumerate(CLASS_NAMES):
                            st.write(f"{label}: {st.session_state.confidence_scores[idx]*100:.2f}%")

elif page == "Image Diagnosis":
    st.title("🧠 AI Medical Image Diagnosis For NeuroDegenerative Diseases")
    st.write("Perform Explainable Image Diagnosis on Brain Scans")
    st.markdown("_____")

    # Check if we have an image from the Home page
    if (st.session_state.temp_path is not None and 
        st.session_state.predicted_class is not None and 
        st.session_state.original_image is not None and 
        st.session_state.img_for_overlay is not None and st.session_state.confidence_scores is not None):
        
        # Load model
        model = load_selected_model()

        # Grad-CAM Analysis
        st.subheader("🔍 Grad-CAM Visual Explanation")
        grad_cam_button = st.button("Generate Grad-CAM Explanation")

        if grad_cam_button:
            if model is not None:
                try:
                    heatmap_overlay = grad_cam_overlay(model, st.session_state.img_for_overlay, layer_name="conv5_block3_out", alpha=0.4)

                    st.markdown("### 🔍 Grad-CAM Visual Explanation + Medical Insight")
                    cam_col, expl_col = st.columns(2)

                    with cam_col:
                        st.image(heatmap_overlay, caption="Grad-CAM Overlay", use_container_width=True)

                    with expl_col:
                        st.info(generate_explanation(st.session_state.predicted_class, st.session_state.predicted_confidence, st.session_state.confidence_scores))

                except Exception as e:
                    st.error(f"🚨 Grad-CAM generation failed: {e}")
            else:
                st.warning("⚠️ Model could not be loaded. Please try again later.")
    else:
        st.warning("⚠️ Please upload an image on the Home page first to perform Grad-CAM analysis.")
        st.info("💡 Go to the Home page, upload a brain scan image, and then return here for detailed visual explanation.")
        
        # Add a button to clear session state if needed
        if st.button("🔄 Reset Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()


st.markdown("_____")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Made with ❤️ by:</p>
    <p>
        <a href="https://github.com/Ayo-Cyber" target="_blank" style="color: #4CAF50; text-decoration: none; margin-right: 20px;">
            🔗 Atunrase Ayomide
        </a>
        &
        <a href="https://github.com/Techtacles" target="_blank" style="color: #4CAF50; text-decoration: none; margin-left: 20px;">
            🔗 Offisong Emmanuel
        </a>
    </p>
    <p style="color: #666; font-size: 14px;">(Artificial Intelligence And Robotics Laboratory)</p>
</div>
""", unsafe_allow_html=True)