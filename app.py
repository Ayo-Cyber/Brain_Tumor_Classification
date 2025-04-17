import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import tempfile

# Grad-CAM Implementation
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    casted_grads = tf.cast(grads > 0, "float32") * grads
    pooled_grads = tf.reduce_mean(casted_grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return output

def grad_cam_overlay(model, image_array, layer_name, alpha=0.4):
    img_array = np.expand_dims(image_array, axis=0)
    heatmap = grad_cam(model, img_array, layer_name)
    overlayed_image = overlay_heatmap(heatmap, image_array, alpha)
    return overlayed_image

def load_and_preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array_exp)
    return img_array_preprocessed, img_array.astype(np.uint8)

# Class Labels
CLASS_NAMES = ['glioma', 'meningioma', 'no tumor', 'pituitary tumor']

# Medical Explanation Dictionary
CLASS_EXPLANATIONS = {
    "glioma": (
        "Gliomas are tumors that arise from glial cells in the brain or spine. "
        "They are typically aggressive and require early diagnosis and treatment. "
        "Common symptoms include headaches, seizures, and cognitive changes."
    ),
    "meningioma": (
        "Meningiomas are usually benign tumors that develop from the meninges, the protective layers around the brain and spinal cord. "
        "Though often slow-growing, they may cause symptoms by pressing on nearby structures."
    ),
    "no tumor": (
        "The model did not detect any tumor-related abnormalities in the brain scan. "
        "However, this does not fully rule out other neurological issues. Further medical evaluation is advised if symptoms persist."
    ),
    "pituitary tumor": (
        "Pituitary tumors are growths found in the pituitary gland, which regulates hormones. "
        "These tumors can lead to hormonal imbalances and vision problems, and are typically benign but may require treatment."
    )
}

# Generate Explanation
def generate_explanation(predicted_class, confidence):
    explanation = CLASS_EXPLANATIONS.get(predicted_class, "No information available.")
    if confidence >= 85:
        confidence_text = "with high confidence"
    elif confidence >= 60:
        confidence_text = "with moderate confidence"
    else:
        confidence_text = "with low certainty"
    return (
        f"🧠 The AI model predicts **{predicted_class.upper()}** {confidence_text} "
        f"({confidence:.2f}% confidence).\n\n"
        f"**Medical Overview:** {explanation}"
    )

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
@st.cache_resource
def load_selected_model():
    try:
        model_path = "/Users/admin/Documents/GitHub/Brain_Tumor_Classification/notebooks/artefacts/_model.h5"
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None

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
                        st.info(generate_explanation(predicted_class, predicted_confidence))

        except Exception as e:
            st.error(f"🚨 Prediction failed: {e}")

st.markdown("_____")
st.markdown("Made with ❤️ by Atunrase Ayomide (University Of Lagos) .")
