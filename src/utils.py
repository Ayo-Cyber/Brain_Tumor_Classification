import tensorflow as tf
import numpy as np
import os
import cv2
import boto3
import streamlit as st
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from google import genai
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import tempfile
from datetime import datetime
import io
from PIL import Image

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
SEED = 42
BUCKET_NAME = "airlab-brain-tumor-model-artifacts"
s3_client = boto3.client("s3")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
llm_prompt = """
I’m building an ML model to classify brain tumors into four classes: glioma, meningioma, pituitary, and no tumor. For a given MRI image, the model outputs the following probabilities:

Glioma: {}

Meningioma: {}

No tumor: {}

Pituitary: {}

Can you  interpret these results in an easy-to-understand manner?
Give a brief description of the tumor (if there's a tumor).
Please include what the prediction implies, the model’s confidence, and a suggestion to the users.

do not include any introduction like "okay, here's your result"

"""


def get_llm_response(client, prompt: str, 
                     glioma_prob: str, 
                     meningioma_prob: str,
                     no_tumor_prob: str, 
                     pituary_prob: str) -> str:
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt.format(glioma_prob, meningioma_prob, no_tumor_prob, pituary_prob),
    )
    return response.text



def get_latest_model_artifact_from_bucket(client, bucket_name: str):
    response = client.list_objects_v2(
    Bucket=BUCKET_NAME
    )
    if 'Contents' not in response:
        return None  # No objects found

    # Find the object with the latest LastModified timestamp
    latest_object = max(response['Contents'], key=lambda obj: obj['LastModified'])

    return {
        'Key': latest_object['Key'],
        'LastModified': latest_object['LastModified']
    }

def download_artifact_from_bucket(client, bucket_name: str, key: str):
    # download the model artifact from s3
    response = client.download_file(
        Bucket=bucket_name,
        Key=key,
        Filename=f"model_artefacts/{key}"
    )
    return "file downloaded from bucket"


def data_generator(train_dir, test_dir):
    # Training data generator
    train_generator = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        seed=SEED
    )

    # Validation data generator
    val_generator = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        validation_split=0.3,
        subset='validation',
        seed=SEED
    )

    # Test data generator
    test_generator = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        validation_split=0.3,
        subset='training',
        seed=SEED
    )

    return train_generator, val_generator, test_generator

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
def generate_explanation(predicted_class, confidence, confidence_scores):
    explanation = CLASS_EXPLANATIONS.get(predicted_class, "No information available.")
    if confidence >= 85:
        confidence_text = "with high confidence"
    elif confidence >= 60:
        confidence_text = "with moderate confidence"
    else:
        confidence_text = "with low certainty"
    return (
        get_llm_response(gemini_client, llm_prompt, confidence_scores[0], confidence_scores[1], confidence_scores[2], confidence_scores[3])
    )

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

@st.cache_resource
def load_selected_model():
    try:
        get_latest_object_key = get_latest_model_artifact_from_bucket(s3_client, BUCKET_NAME).get("Key")
        print(f"Key is {get_latest_object_key}")
        current_working_directory = os.getcwd()
        print(f"Current working directory is {current_working_directory}")
        path_to_check = f"{current_working_directory}/model_artefacts/{get_latest_object_key}"
        if os.path.isfile(path_to_check):
            print(f"Model found locally... using model {path_to_check}")
            model_path = f"model_artefacts/{get_latest_object_key}"
        else:
            print(f"Downloading artifact {get_latest_object_key} from bucket {BUCKET_NAME} ")
            download_artifact_from_bucket(s3_client, BUCKET_NAME, get_latest_object_key)

        model_path = f"model_artefacts/{get_latest_object_key}"
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None


# artefacts in a pdf

def generate_medical_report_pdf(
    original_image,
    heatmap_overlay, 
    predicted_class,
    predicted_confidence,
    confidence_scores,
    explanation_text,
    patient_name="Patient",
    doctor_name="AI Diagnostic System",
    report_id=None
):
    """
    Generate a comprehensive medical diagnosis report as PDF
    
    Args:
        original_image: PIL Image object of the original brain scan
        heatmap_overlay: PIL Image object of the Grad-CAM overlay
        predicted_class: String of the predicted diagnosis
        predicted_confidence: Float of prediction confidence (0-100)
        confidence_scores: Array of all class confidence scores
        explanation_text: String containing the medical explanation
        patient_name: String of patient name (default: "Patient")
        doctor_name: String of diagnosing doctor/system (default: "AI Diagnostic System")
        report_id: String of report ID (auto-generated if None)
    
    Returns:
        bytes: PDF file content as bytes
    """
    
    # Create a temporary file for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY
    )
    
    # Story to hold all elements
    story = []
    
    # Header
    story.append(Paragraph("NEUROLOGICAL DIAGNOSTIC REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata table
    if report_id is None:
        report_id = f"NDR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    metadata = [
        ['Report ID:', report_id],
        ['Patient Name:', patient_name],
        ['Date & Time:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ['Diagnostic System:', doctor_name],
        ['Report Type:', 'AI-Assisted Neuroimaging Analysis']
    ]
    
    metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 30))
    
    # Primary Diagnosis Section
    story.append(Paragraph("PRIMARY DIAGNOSIS", subtitle_style))
    
    diagnosis_data = [
        ['Predicted Condition:', predicted_class],
        ['Confidence Level:', f"{predicted_confidence:.2f}%"],
        ['Diagnostic Method:', 'Deep Learning CNN Analysis with Grad-CAM Visualization']
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(diagnosis_table)
    story.append(Spacer(1, 20))
    
    # Confidence Scores Section
    story.append(Paragraph("DETAILED CONFIDENCE ANALYSIS", subtitle_style))
    
    # Assuming CLASS_NAMES is available or passed
    # You might need to import CLASS_NAMES or pass it as a parameter
    try:
        from src.utils import CLASS_NAMES
        confidence_data = [['Condition', 'Confidence Score']]
        for idx, class_name in enumerate(CLASS_NAMES):
            confidence_data.append([class_name, f"{confidence_scores[idx]*100:.2f}%"])
    except ImportError:
        # Fallback if CLASS_NAMES not available
        confidence_data = [['Condition', 'Confidence Score']]
        for idx, score in enumerate(confidence_scores):
            confidence_data.append([f"Class {idx+1}", f"{score*100:.2f}%"])
    
    confidence_table = Table(confidence_data, colWidths=[3*inch, 2*inch])
    confidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(confidence_table)
    story.append(Spacer(1, 30))
    
    # Medical Images Section
    story.append(Paragraph("DIAGNOSTIC IMAGING", subtitle_style))
    
    # Save images temporarily for inclusion in PDF
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_orig:
        original_image.save(tmp_orig.name)
        orig_path = tmp_orig.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_heat:
        if isinstance(heatmap_overlay, np.ndarray):
            Image.fromarray(heatmap_overlay).save(tmp_heat.name)
        else:
            heatmap_overlay.save(tmp_heat.name)
        heat_path = tmp_heat.name
    
    # Create image table
    image_table_data = [
        [ReportLabImage(orig_path, width=2.5*inch, height=2.5*inch), 
         ReportLabImage(heat_path, width=2.5*inch, height=2.5*inch)],
        ['Original Brain Scan', 'Grad-CAM Analysis Overlay']
    ]
    
    image_table = Table(image_table_data, colWidths=[3*inch, 3*inch])
    image_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
    ]))
    
    story.append(image_table)
    story.append(Spacer(1, 30))
    
    # Medical Explanation Section
    story.append(Paragraph("CLINICAL INTERPRETATION", subtitle_style))
    story.append(Paragraph(explanation_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Disclaimer Section
    story.append(Paragraph("IMPORTANT DISCLAIMER", subtitle_style))
    disclaimer_text = """
    This report is generated by an AI-assisted diagnostic system and is intended for research and 
    educational purposes only. It should NOT be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions 
    regarding medical conditions. The AI system's predictions are based on pattern recognition from 
    training data and may not account for all clinical factors relevant to individual cases.
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | AI Neuro Diagnosis System"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    story.append(Paragraph(footer_text, footer_style))
    
    # Build PDF
    doc.build(story)
    
    # Cleanup temporary files
    try:
        os.unlink(orig_path)
        os.unlink(heat_path)
    except:
        pass
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

def save_report_to_streamlit(pdf_bytes, filename="neurological_diagnostic_report.pdf"):
    """
    Helper function to provide download button in Streamlit
    
    Args:
        pdf_bytes: PDF content as bytes
        filename: Desired filename for download
    
    Returns:
        Streamlit download button
    """
    if isinstance(pdf_bytes, bytes):
        return st.download_button(
            label="📄 Download Medical Report (PDF)",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            help="Click to download the complete diagnostic report as PDF"
        )
    else:
        st.error("PDF data is not in bytes format. Cannot offer download.")