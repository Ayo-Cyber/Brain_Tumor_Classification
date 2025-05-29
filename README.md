# 🧠 Brain Tumor Classification with AI

<div align="center">

![Brain Tumor Classification](https://github.com/user-attachments/assets/brain-tumor-ai-banner.png)

*Advanced Machine Learning for Medical Diagnosis*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 🎯 Project Overview

This project leverages cutting-edge deep learning techniques to classify brain tumors from medical imaging data. Our AI-powered solution provides accurate, fast, and explainable diagnosis to assist healthcare professionals in making informed decisions.

### 🔬 Key Features

- **🎯 Multi-class Classification**: Accurately distinguishes between different types of brain tumors
- **🔍 Explainable AI**: Grad-CAM visualizations show which brain regions influence predictions
- **🌐 Interactive Web App**: User-friendly Streamlit interface for real-time diagnosis
- **📊 Comprehensive Evaluation**: Detailed performance metrics and visualization
- **🏥 Medical-Grade Accuracy**: Trained on extensive medical imaging datasets

### 🩺 Supported Classifications

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor (Healthy)

## 🏗️ Project Architecture

```
🧠 Brain_Tumor_Classification/
├── 📱 app.py                    # Main Streamlit application
├── 🐳 Dockerfile              # Container configuration
├── 📋 requirements.txt         # Python dependencies
├── 📊 data/                    # Training and test datasets
├── 🧪 mlruns/                  # MLflow experiment tracking
├── 🎯 model_artefacts/         # Saved trained models
├── 📓 notebooks/               # Jupyter notebooks for analysis
│   ├── 🔍 data_exploration.ipynb
│   ├── 🧪 model_experiment.ipynb
│   └── 🏆 best_model.h5
├── 📈 reports/                 # Performance reports and metrics
│   ├── 📊 classification_report.txt
│   └── 🔀 confusion_matrix.txt
└── 🔧 src/                     # Core source code
    ├── 🤖 models.py           # Model architectures
    ├── 🏋️ train.py            # Training pipeline
    ├── 📏 evaluate.py         # Model evaluation
    └── 🛠️ utils.py            # Utility functions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Brain_Tumor_Classification.git
   cd Brain_Tumor_Classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🎮 Usage

#### 🌐 Launch Web Application
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

#### 🏋️ Train Custom Model
```bash
python src/train.py --epochs 50 --batch_size 32
```

#### 📊 Evaluate Model Performance
```bash
python src/evaluate.py --model_path model_artefacts/best_model.h5
```

## 📱 Web Application Features

### 🏠 Home Page
- **📁 Image Upload**: Drag & drop or browse for brain scan images
- **⚡ Instant Prediction**: Real-time classification with confidence scores
- **📊 Results Visualization**: Clear, medical-grade result presentation

### 🔬 Image Diagnosis Page
- **🔥 Grad-CAM Analysis**: Visual explanation of AI decision-making
- **🎛️ Customizable Parameters**: Adjust layer selection and overlay intensity
- **🩺 Medical Insights**: Detailed explanations and recommendations

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.2% |
| **Precision** | 94.8% |
| **Recall** | 95.1% |
| **F1-Score** | 94.9% |

### 🎯 Class-wise Performance
- **Glioma**: 96.3% accuracy
- **Meningioma**: 94.7% accuracy  
- **Pituitary**: 95.8% accuracy
- **No Tumor**: 94.1% accuracy

## 🔬 Technical Specifications

### 🧠 Model Architecture
- **Base**: ResNet-50 with transfer learning
- **Input**: 224x224x3 RGB images
- **Output**: 4-class softmax classification
- **Optimization**: Adam optimizer with learning rate scheduling

### 📊 Data Pipeline
- **Preprocessing**: Normalization, augmentation, resizing
- **Training Set**: 80% (3,200 images)
- **Validation Set**: 10% (400 images)
- **Test Set**: 10% (400 images)

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t brain-tumor-classifier .

# Run container
docker run -p 8501:8501 brain-tumor-classifier
```

## 📈 Experiment Tracking

This project uses MLflow for experiment tracking:

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Access the MLflow dashboard at `http://localhost:5000`

## 🧪 Notebooks

Explore our comprehensive analysis:

- **📊 Data Exploration**: Statistical analysis and visualization of the dataset
- **🧪 Model Experiments**: Comparison of different architectures and hyperparameters
- **🔍 Error Analysis**: Deep dive into misclassified cases

## 📋 Requirements

```
tensorflow>=2.8.0
streamlit>=1.25.0
opencv-python>=4.7.0
pillow>=9.0.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
mlflow>=2.0.0
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

### 👥 Development Team
Made with ❤️ by:
- **[Atunrase Ayomide](https://github.com/Ayo-Cyber)** - AI/ML Engineer and Researcher
- **[Offisong Emmanuel](https://github.com/Techtacles)** - Data Engineer and Devops Engineer

*University of Lagos : Artificial Intelligence And Robotics Laboratory*

### 🌟 Special Thanks
- Medical imaging research community
- Open-source contributors
- Healthcare professionals providing domain expertise
- TensorFlow and Streamlit development teams

## 📞 Contact & Support

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/Brain_Tumor_Classification/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/Brain_Tumor_Classification/discussions)

---

<div align="center">

**⚠️ Medical Disclaimer**: This tool is for research and educational purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

*Built with 🧠 for advancing medical AI*

</div>