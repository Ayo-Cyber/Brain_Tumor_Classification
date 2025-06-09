import streamlit as st

# Replace your existing help_tutorial section in the sidebar with this:

def help_tutorial():
    st.title("📚 Help & Tutorial")
    st.write("Complete guide to using the AI Medical Image Diagnosis system")
    st.markdown("---")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📋 How to Use", "🔬 Understanding Results", "❓ FAQ", "📞 Support"])
    
    with tab1:
        st.header("🧠 About AI Medical Image Diagnosis")
        
        st.markdown("""
        ### What is this application?
        This is an **AI-powered medical diagnostic tool** specifically designed to analyze brain scan images for **neurodegenerative diseases**. 
        The system uses advanced deep learning models to provide preliminary diagnostic insights.
        
        ### Key Features
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔍 AI-Powered Analysis**
            - Advanced neural network models
            - High accuracy predictions
            - Confidence scoring system
            
            **📊 Quality Assessment**
            - Automatic image quality evaluation
            - Resolution and clarity checks
            - Quality-based recommendations
            
            **🎯 Explainable AI**
            - Grad-CAM visual explanations
            - Highlighted brain regions
            - Medical insights generation
            """)
        
        with col2:
            st.markdown("""
            **📄 Comprehensive Reports**
            - Professional PDF reports
            - Detailed analysis results
            - Downloadable documentation
            
            **🎨 User-Friendly Interface**
            - Intuitive navigation
            - Dark/Light mode support
            - Step-by-step guidance
            
            **⚕️ Medical Integration**
            - Clinical terminology
            - Professional formatting
            - Healthcare workflow support
            """)
        
        st.warning("""
        **⚠️ Important Medical Disclaimer**
        
        This tool is designed for **research and educational purposes** and should **NOT** be used as a substitute for professional medical diagnosis. 
        Always consult with qualified healthcare professionals for medical decisions.
        """)
    
    with tab2:
        st.header("📋 Step-by-Step Usage Guide")
        
        st.markdown("### 🔄 Complete Workflow")
        
        # Step 1
        st.markdown("#### 1️⃣ Upload Your Brain Scan Image")
        st.markdown("""
        **Go to the Home page:**
        - Click on **"Upload an Image"** button
        - Select brain scan files (JPG, PNG, JPEG formats)
        - Supported image types: MRI, CT scans, X-rays
        
        **Image Requirements:**
        - Minimum resolution: 224x224 pixels
        - Clear, high-contrast images work best
        - Avoid blurry or low-quality scans
        """)
        
        # Step 2
        st.markdown("#### 2️⃣ Quality Assessment")
        st.markdown("""
        **Automatic Quality Check:**
        - System evaluates image quality automatically
        - Provides quality score (0-100)
        - Shows resolution and clarity metrics
        
        **Quality Factors:**
        - **Resolution**: Image dimensions and pixel density
        - **Clarity**: Sharpness and focus quality
        - **Contrast**: Brightness and contrast levels
        - **Noise**: Image artifacts and distortions
        """)
        
        # Step 3
        st.markdown("#### 3️⃣ AI Diagnosis")
        st.markdown("""
        **Automated Analysis:**
        - AI model processes the brain scan
        - Generates prediction with confidence score
        - Shows probability for all possible conditions
        
        **Results Include:**
        - **Primary Prediction**: Most likely condition
        - **Confidence Level**: Percentage certainty
        - **All Class Scores**: Probability breakdown
        """)
        
        # Step 4
        st.markdown("#### 4️⃣ Visual Explanation (Image Diagnosis Page)")
        st.markdown("""
        **Grad-CAM Analysis:**
        - Click **"Generate Grad-CAM Explanation"**
        - Visual heatmap shows important brain regions
        - Medical insights explain the findings
        
        **Understanding Grad-CAM:**
        - **Red/Hot areas**: High importance regions
        - **Blue/Cool areas**: Lower importance regions
        - **Overlay**: Highlights decision-making areas
        """)
        
        # Step 5
        st.markdown("#### 5️⃣ Generate Medical Report")
        st.markdown("""
        **Professional Documentation:**
        - Fill in patient and doctor information
        - Add additional clinical notes
        - Generate comprehensive PDF report
        - Download for medical records
        
        **Report Contents:**
        - Patient information
        - Analysis results and confidence
        - Visual explanations
        - Medical recommendations
        - Quality assessment details
        """)
        
        st.success("💡 **Pro Tip**: Complete all steps in order for the most comprehensive analysis!")
    
    with tab3:
        st.header("🔬 Understanding Your Results")
        
        st.markdown("### 📊 Interpreting Predictions")
        
        # Confidence Levels
        st.markdown("#### Confidence Levels")
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            st.markdown("""
            **High Confidence (80-100%)**
            - 🟢 Strong prediction reliability
            - Clear diagnostic indicators
            - High model certainty
            
            **Medium Confidence (60-79%)**
            - 🟡 Moderate prediction reliability
            - Some diagnostic indicators present
            - Further evaluation recommended
            """)
        
        with conf_col2:
            st.markdown("""
            **Low Confidence (40-59%)**
            - 🟠 Limited prediction reliability
            - Weak diagnostic indicators
            - Additional testing advised
            
            **Very Low Confidence (<40%)**
            - 🔴 Unreliable prediction
            - Insufficient diagnostic evidence
            - Professional consultation required
            """)
        
        # Quality Impact
        st.markdown("#### 🎯 Image Quality Impact")
        st.markdown("""
        **Quality Score Interpretation:**
        - **90-100**: Excellent quality - Highly reliable results
        - **70-89**: Good quality - Reliable results
        - **50-69**: Acceptable quality - Results with caution
        - **30-49**: Poor quality - Limited reliability
        - **<30**: Very poor quality - Not recommended for analysis
        
        **Quality Factors Affecting Results:**
        - **Low Resolution**: Reduces diagnostic accuracy
        - **Blurriness**: May miss important details
        - **Poor Contrast**: Affects feature detection
        - **Artifacts**: Can lead to false positives
        """)
        
        # Grad-CAM Explanation
        st.markdown("#### 🔍 Grad-CAM Visual Analysis")
        st.markdown("""
        **Heat Map Colors:**
        - **Red/Orange**: High attention areas (most important for diagnosis)
        - **Yellow**: Moderate attention areas
        - **Green**: Low attention areas
        - **Blue/Purple**: Minimal attention areas
        
        **Medical Interpretation:**
        - Focus areas often correspond to anatomical regions
        - Patterns may indicate specific pathological changes
        - Multiple hot spots suggest distributed abnormalities
        - Concentrated areas may indicate localized conditions
        """)
    
    with tab4:
        st.header("❓ Frequently Asked Questions")
        
        # Create expandable FAQ sections
        with st.expander("🖼️ **What image formats are supported?**"):
            st.markdown("""
            **Supported formats:**
            - JPG/JPEG
            - PNG
            - 
            **Recommended specifications:**
            - Minimum resolution: 224x224 pixels
            - Maximum file size: 10MB
            - High contrast, clear images work best
            """)
        
        with st.expander("🎯 **How accurate are the predictions?**"):
            st.markdown("""
            **Accuracy depends on several factors:**
            - Image quality and resolution
            - Type of brain scan (MRI vs CT)
            - Specific condition being diagnosed
            - Model training data quality
            
            **Important notes:**
            - Results are preliminary assessments only
            - Not a replacement for professional diagnosis
            - Should be used as a screening tool
            - Always consult healthcare professionals
            """)
        
        with st.expander("⚠️ **What if my image quality is poor?**"):
            st.markdown("""
            **Options for poor quality images:**
            - Try to obtain a higher quality scan
            - Check if the image can be enhanced
            - Proceed with caution (results marked as unreliable)
            - Consider retaking the medical scan
            
            **Quality improvement tips:**
            - Ensure proper scanning parameters
            - Avoid motion artifacts
            - Use appropriate contrast settings
            - Maintain proper patient positioning
            """)
        
        with st.expander("📄 **How do I use the medical reports?**"):
            st.markdown("""
            **Medical report features:**
            - Professional PDF format
            - Includes all analysis results
            - Contains visual explanations
            - Suitable for medical records
            
            **Best practices:**
            - Fill in accurate patient information
            - Add relevant clinical notes
            - Share with healthcare providers
            - Keep for medical history records
            """)
        
        with st.expander("🔒 **Is my medical data secure?**"):
            st.markdown("""
            **Data security measures:**
            - Images are processed locally during session
            - No permanent storage of medical images
            - Session data cleared when browser closes
            - No data transmitted to external servers
            
            **Privacy recommendations:**
            - Use secure, private internet connections
            - Clear browser data after use
            - Avoid using on public computers
            - Follow institutional data policies
            """)
        
        with st.expander("🚨 **What should I do in medical emergencies?**"):
            st.markdown("""
            **Important reminder:**
            This tool is NOT for emergency medical situations.
            
            **In case of medical emergency:**
            - Contact emergency services immediately
            - Seek immediate medical attention
            - Do not rely on AI diagnosis for urgent care
            - Follow established emergency protocols
            """)
    
    with tab5:
        st.header("📞 Support & Contact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 👨‍💻 Development Team
            
            **Atunrase Ayomide**
            - 🔗 [GitHub Profile](https://github.com/Ayo-Cyber)
            - 📧 Email: [Contact via GitHub]
            - 🎓 AI & Robotics Laboratory
            
            **Offisong Emmanuel**
            - 🔗 [GitHub Profile](https://github.com/Techtacles)
            - 📧 Email: [Contact via GitHub]
            - 🎓 AI & Robotics Laboratory
            """)
        
        with col2:
            st.markdown("""
            ### 🆘 Getting Help
            
            **Technical Issues:**
            - Check the FAQ section above
            - Review the step-by-step guide
            - Ensure image quality requirements
            - Try refreshing the application
            
            **Feature Requests:**
            - Submit issues on GitHub
            - Contact the development team
            - Provide detailed descriptions
            - Include example use cases
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🔧 Troubleshooting Common Issues
        
        **Upload Problems:**
        - Check file format (JPG, PNG only)
        - Verify file size (<10MB)
        - Ensure stable internet connection
        - Try refreshing the page
        
        **Prediction Errors:**
        - Verify image quality score
        - Check if model loaded successfully
        - Ensure image contains brain scan
        - Try with a different image
        
        **PDF Generation Issues:**
        - Complete all analysis steps first
        - Generate Grad-CAM explanation
        - Fill in required report fields
        - Check browser's download settings
        """)
        
        st.success("""
        **💡 Quick Help Tips:**
        - Use high-quality, clear brain scan images
        - Follow the step-by-step workflow
        - Check quality assessment before proceeding
        - Always interpret results with medical professionals
        """)
        
        st.info("""
        **🎓 Academic Institution:**
        Artificial Intelligence And Robotics Laboratory
        
        This project is developed for educational and research purposes in medical AI applications.
        """)

