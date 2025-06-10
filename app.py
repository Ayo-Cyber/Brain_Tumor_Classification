import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from src.utils import (generate_explanation ,
                        CLASS_NAMES, 
                        grad_cam_overlay ,
                          load_and_preprocess_image ,
                            load_selected_model,
                              generate_medical_report_pdf,
                                assess_image_quality, 
                                display_quality_assessment,  
                                should_proceed_with_analysis, 
                                create_dark_theme,
                                create_light_theme) 
from src.help import help_tutorial 
import tempfile
from datetime import datetime
import os

# Page Configuration
st.set_page_config(page_title="AI Neuro Diagnosis", page_icon="🩺", layout="wide")
# if st.button("🔍 Go to Image Diagnosis"):
#     st.session_state.page = "Image Diagnosis"
#     st.rerun()



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
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
# ADD THESE NEW SESSION STATE VARIABLES
if 'quality_results' not in st.session_state:
    st.session_state.quality_results = None
if 'quality_approved' not in st.session_state:
    st.session_state.quality_approved = False

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home" , "Image Diagnosis", "Medical Report", "Help & Tutorial"])

    dark_mode = st.toggle("Dark Mode")
    # Apply theme based on toggle
    if dark_mode:
        create_dark_theme()
        st.info("🌙 Dark mode enabled! Refresh the page to see changes.")
    else:
        create_light_theme()
        if st.session_state.get('was_dark_mode', False):
            st.info("☀️ Light mode enabled! Refresh the page to see changes.")

    # Store the previous state
    st.session_state.was_dark_mode = dark_mode

# Add a refresh button for immediate theme application
    if st.sidebar.button("🔄 Apply Theme"):
        st.rerun()

if st.session_state.page == "Home":
    # session state for home pag
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
        
        # === IMAGE QUALITY ASSESSMENT SECTION ===
        st.markdown("---")
        
        # Convert PIL image to numpy array for quality assessment
        img_array = np.array(original_image)
        
        # Assess image quality
        quality_results = assess_image_quality(img_array, min_resolution=(224, 224))
        st.session_state.quality_results = quality_results
        
        # Display quality assessment
        display_quality_assessment(quality_results)
        
        # Determine if we should proceed with analysis
        can_proceed = should_proceed_with_analysis(quality_results, min_score=30)  # Lower threshold for demo
        
        if not can_proceed:
            st.error("❌ **Image quality is too low for reliable analysis**")
            st.info("Please upload a higher quality image or check the recommendations above.")
            
            # Optional: Allow user to proceed anyway with strong warning
            proceed_anyway = st.checkbox("⚠️ Proceed with analysis anyway (NOT RECOMMENDED)", 
                                       help="Analysis results may be unreliable with poor quality images")
            if proceed_anyway:
                st.session_state.quality_approved = True
                st.warning("⚠️ **DISCLAIMER**: Results may be unreliable due to poor image quality.")
            else:
                st.session_state.quality_approved = False
        else:
            st.session_state.quality_approved = True
            st.success("✅ Image quality is acceptable for analysis")
        
        # === CONTINUE WITH EXISTING LOGIC ONLY IF QUALITY IS APPROVED ===
        if st.session_state.quality_approved:
            st.markdown("---")
            
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
                        
                        # Add quality disclaimer if quality was poor but user proceeded
                        if quality_results['overall_score'] < 60:
                            st.warning("⚠️ **Note**: Results may be affected by image quality")

                        with st.expander("🔍 See All Class Confidence Scores"):
                            for idx, label in enumerate(CLASS_NAMES):
                                st.write(f"{label}: {confidence_scores[idx]*100:.2f}%")

                except Exception as e:
                    st.error(f"🚨 Prediction failed: {e}")
        else:
            # Clear any existing predictions if quality is not approved
            st.session_state.predicted_class = None
            st.session_state.predicted_confidence = None
            st.session_state.confidence_scores = None
    
    # Show current session state (for debugging/info)
    elif st.session_state.original_image is not None:
        st.info("📷 You have an image already uploaded. You can go to Image Diagnosis page for Grad-CAM analysis.")
        
        # Show quality assessment if available
        if st.session_state.quality_results is not None:
            display_quality_assessment(st.session_state.quality_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.original_image, caption="Previously Uploaded Image", use_container_width=True)
        with col2:
            if st.session_state.predicted_class:
                st.subheader("🧾 Previous Diagnosis Result")
                st.write(f"**Prediction:** {st.session_state.predicted_class}")
                st.write(f"**Confidence:** {st.session_state.predicted_confidence:.2f}%")
                
                # Show quality warning if applicable
                if (st.session_state.quality_results and 
                    st.session_state.quality_results['overall_score'] < 60):
                    st.warning("⚠️ **Note**: Results may be affected by image quality")

                with st.expander("🔍 See All Class Confidence Scores"):
                        for idx, label in enumerate(CLASS_NAMES):
                            st.write(f"{label}: {st.session_state.confidence_scores[idx]*100:.2f}%")

elif st.session_state.page == "Image Diagnosis":
    st.title("🧠 AI Medical Image Diagnosis For NeuroDegenerative Diseases")
    st.write("Perform Explainable Image Diagnosis on Brain Scans")
    st.markdown("_____")

    # Check if we have an image from the Home page
    if (st.session_state.temp_path is not None and 
        st.session_state.predicted_class is not None and 
        st.session_state.original_image is not None and 
        st.session_state.img_for_overlay is not None and 
        st.session_state.confidence_scores is not None and
        st.session_state.quality_approved):  # ADD QUALITY CHECK
        
        # Show quality assessment summary
        if st.session_state.quality_results is not None:
            quality_col1, quality_col2, quality_col3 = st.columns(3)
            with quality_col1:
                st.metric("Image Quality", f"{st.session_state.quality_results['overall_score']}/100")
            with quality_col2:
                st.metric("Status", st.session_state.quality_results['status'])
            with quality_col3:
                st.metric("Resolution", st.session_state.quality_results['resolution'])
        
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
                        st.session_state.explanation = generate_explanation(st.session_state.predicted_class,
                                                                             st.session_state.predicted_confidence,
                                                                               st.session_state.confidence_scores)
                        st.info(st.session_state.explanation)
                        
                        # Add quality disclaimer to explanation if needed
                        if (st.session_state.quality_results and 
                            st.session_state.quality_results['overall_score'] < 60):
                            st.warning("⚠️ **Quality Note**: Analysis based on suboptimal image quality. Results should be interpreted with caution.")

                except Exception as e:
                    st.error(f"🚨 Grad-CAM generation failed: {e}")
            else:
                st.warning("⚠️ Model could not be loaded. Please try again later.")
    
    elif not st.session_state.quality_approved:
        st.error("❌ **Image quality assessment required**")
        st.info("💡 Please go to the Home page and upload a high-quality image that passes quality assessment.")
        
    else:
        st.warning("⚠️ Please upload an image on the Home page first to perform Grad-CAM analysis.")
        st.info("💡 Go to the Home page, upload a brain scan image, and then return here for detailed visual explanation.")
        
        # Add a button to clear session state if needed
        if st.button("🔄 Reset Session"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

elif st.session_state.page == "Medical Report":
    st.title("📄 Medical Report Generation")
    st.write("Create and download comprehensive diagnostic reports")
    st.markdown("_____")
    
    # Check if we have all necessary data for PDF generation
    if (st.session_state.original_image is not None and
        st.session_state.predicted_class is not None and
        st.session_state.predicted_confidence is not None and
        st.session_state.confidence_scores is not None and
        st.session_state.explanation is not None and
        st.session_state.img_for_overlay is not None and
        st.session_state.quality_approved):  # ADD QUALITY CHECK
        
        # Display current analysis summary
        st.subheader("📊 Current Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)  # ADD 4th COLUMN
        
        with col1:
            st.metric("Predicted Condition", st.session_state.predicted_class)
        
        with col2:
            st.metric("Confidence Level", f"{st.session_state.predicted_confidence:.2f}%")
        
        with col3:
            st.metric("Analysis Status", "Complete ✅")
            
        with col4:  # ADD QUALITY METRIC
            if st.session_state.quality_results:
                quality_score = st.session_state.quality_results['overall_score']
                st.metric("Image Quality", f"{quality_score}/100")
        
        # Show quality assessment summary
        if st.session_state.quality_results is not None:
            display_quality_assessment(st.session_state.quality_results)
        
        # Show preview of images
        st.subheader("🖼️ Diagnostic Images Preview")
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            st.image(st.session_state.original_image, caption="Original Brain Scan", use_container_width=True)
        
        with preview_col2:
            st.image(st.session_state.img_for_overlay, caption="Grad-CAM Analysis", use_container_width=True)
        
        # Report generation form
        st.markdown("---")
        st.subheader("📝 Report Information")

        with st.form("report_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_name = st.text_input("Patient Name", value="Anonymous Patient", help="Enter patient's name or leave as anonymous")
                doctor_name = st.text_input("Doctor/Institution Name", value="AI Diagnostic System", help="Name of the diagnosing physician or institution")
            
            with col2:
                report_id = st.text_input("Report ID (Optional)", value="", help="Leave empty for auto-generation")
                additional_notes = st.text_area("Additional Notes (Optional)", value="", help="Any additional clinical notes or observations")
            
            # Form submission
            generate_report = st.form_submit_button("🔄 Generate Medical Report", type="primary")

        if generate_report:
            with st.spinner("Generating comprehensive medical report..."):
                try:
                    # Add additional notes to explanation if provided
                    full_explanation = st.session_state.explanation
                    if additional_notes.strip():
                        full_explanation += f"\n\nAdditional Clinical Notes:\n{additional_notes}"
                    
                    # ADD QUALITY DISCLAIMER TO EXPLANATION
                    if (st.session_state.quality_results and 
                        st.session_state.quality_results['overall_score'] < 60):
                        quality_disclaimer = st.session_state.quality_results
                        image_disclaimer = "⚠️ **Quality Note**: Analysis based on suboptimal image quality. Results should be interpreted with caution."
                        full_explanation += f"\n\n{image_disclaimer}"
                    
                    pdf_bytes = generate_medical_report_pdf(
                        original_image=st.session_state.original_image,
                        heatmap_overlay=st.session_state.img_for_overlay,
                        predicted_class=st.session_state.predicted_class,
                        predicted_confidence=st.session_state.predicted_confidence,
                        confidence_scores=st.session_state.confidence_scores,
                        explanation_text=full_explanation,
                        patient_name=patient_name if patient_name else "Anonymous Patient",
                        doctor_name=doctor_name,
                        image_quality_report=st.session_state.quality_results,
                        report_id=report_id if report_id else None
                    )
                    
                    # Store PDF in session state
                    st.session_state.pdf_report = pdf_bytes
                    
                    # Safe file name - STORE IT IN SESSION STATE
                    safe_patient_name = patient_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    safe_condition = st.session_state.predicted_class.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    filename = f"neurological_report_{safe_patient_name}_{safe_condition}.pdf"
                    st.session_state.pdf_filename = filename 
                    
                    st.success("✅ Medical report generated successfully!")

                    # Show report details
                    st.info(f"📄 Report generated for: {patient_name}")
                    st.info(f"🏥 Institution: {doctor_name}")
                    st.info(f"📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

                except Exception as e:
                    st.error(f"🚨 PDF generation failed: {e}")
                    st.error("Please check if all required libraries are installed (reportlab)")

        # Download button - now it will use the stored filename
        if "pdf_report" in st.session_state:
            filename = st.session_state.get("pdf_filename", "neurological_report.pdf")
            st.download_button(
                label="📥 Download PDF Report",
                data=st.session_state.pdf_report,
                file_name=filename,
                mime="application/pdf"
            )
    
    elif not st.session_state.quality_approved:
        st.error("❌ **Quality assessment required for report generation**")
        st.info("💡 Please ensure your uploaded image passes quality assessment on the Home page.")
        
    elif st.session_state.original_image is not None and st.session_state.predicted_class is not None:
        # Have basic analysis but missing Grad-CAM
        st.warning("⚠️ Basic diagnosis available, but Grad-CAM analysis is required for complete report generation.")
        st.info("💡 Please go to the 'Image Diagnosis' page and generate Grad-CAM explanation first.")
        
        # Show what we have
        st.subheader("📊 Available Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(st.session_state.original_image, caption="Uploaded Brain Scan", use_container_width=True)
        
        with col2:
            st.metric("Predicted Condition", st.session_state.predicted_class)
            st.metric("Confidence Level", f"{st.session_state.predicted_confidence:.2f}%")
            
        if st.button("🔍 Go to Image Diagnosis"):
            st.switch_page("Image Diagnosis")  
    
    else:
        st.warning("⚠️ No diagnostic data available for report generation.")
        st.info("💡 Please complete the following steps:")
        st.markdown("""
        1. **Upload Image**: Go to the Home page and upload a brain scan
        2. **Quality Check**: Ensure image passes quality assessment
        3. **Get Diagnosis**: The AI will analyze and provide predictions
        4. **Generate Grad-CAM**: Go to Image Diagnosis page for visual explanation
        5. **Create Report**: Return here to generate and download your report
        """)
        
        if st.button("🏠 Go to Home Page"):
            st.session_state.page = "Home"
            st.rerun()

elif page== "Help & Tutorial":
    help_tutorial()

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