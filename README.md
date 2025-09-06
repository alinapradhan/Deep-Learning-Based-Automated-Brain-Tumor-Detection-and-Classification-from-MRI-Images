# Deep Learning-Based Automated Brain Tumor Detection and Classification from MRI Images

A comprehensive deep learning system for automated detection and classification of brain tumors from MRI images. This system provides robust tumor classification, precise localization through segmentation, and clinical-grade reporting with confidence scores suitable for medical applications.

## 🩺 Features

- **Multi-class Classification**: Distinguishes between No Tumor, Glioma, Meningioma, and Pituitary tumors
- **Tumor Localization**: U-Net based segmentation for precise tumor boundary detection
- **Clinical Reporting**: Generates comprehensive reports with confidence scores and risk assessment
- **Robust Preprocessing**: Handles various MRI formats (DICOM, NIfTI, standard images)
- **Performance Metrics**: Comprehensive evaluation including accuracy, sensitivity, specificity, F1-score
- **Visualization Tools**: Advanced visualization for clinical interpretation
- **Batch Processing**: Efficient processing of multiple images
- **Medical Image Support**: Native support for DICOM and NIfTI formats

## 🏗️ Architecture

### Classification Model
- **Base Architecture**: ResNet50 with pre-trained ImageNet weights
- **Custom Head**: Dense layers with dropout for medical image classification
- **Input Size**: 224×224×3 RGB images
- **Output**: 4-class classification with confidence scores

### Segmentation Model
- **Architecture**: U-Net with encoder-decoder structure
- **Purpose**: Tumor localization and boundary detection
- **Output**: Binary masks and bounding boxes
- **Metrics**: Dice coefficient, IoU score, pixel accuracy

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   ├── cnn_model.py          # ResNet50-based classification model
│   │   └── segmentation_model.py # U-Net segmentation model
│   ├── preprocessing/
│   │   ├── data_loader.py        # Data loading and preprocessing
│   │   └── image_utils.py        # Image preprocessing utilities
│   ├── training/
│   │   ├── train_classifier.py   # Classification training script
│   │   └── train_segmentation.py # Segmentation training script
│   ├── evaluation/
│   │   ├── metrics.py           # Comprehensive evaluation metrics
│   │   └── visualization.py     # Result visualization tools
│   ├── inference/
│   │   └── predict.py          # Inference pipeline
│   └── utils/
│       └── config.py           # Configuration settings
├── main.py                     # Main entry point
├── requirements.txt           # Dependencies
└── README.md                 # Documentation
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/alinapradhan/Deep-Learning-Based-Automated-Brain-Tumor-Detection-and-Classification-from-MRI-Images.git
cd Deep-Learning-Based-Automated-Brain-Tumor-Detection-and-Classification-from-MRI-Images
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python main.py --help
```

## 📊 Data Preparation

### Dataset Structure
Organize your MRI images in the following structure:
```
data/
├── no_tumor/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── glioma/
│   ├── image1.jpg
│   └── ...
├── meningioma/
│   ├── image1.jpg
│   └── ...
└── pituitary/
    ├── image1.jpg
    └── ...
```

### Create Demo Structure
```bash
python main.py create_demo --data_dir ./demo_data
```

### Supported Formats
- **Standard Images**: JPG, PNG, BMP, TIFF
- **Medical Images**: DICOM (.dcm), NIfTI (.nii, .nii.gz)

## 🎯 Usage

### Training Models

**Basic Training**:
```bash
python main.py train --data_dir ./data --output_dir ./models
```

**Advanced Training**:
```bash
python main.py train \
    --data_dir ./data \
    --output_dir ./models \
    --epochs 100 \
    --learning_rate 0.001 \
    --batch_size 32
```

### Running Inference

**Single Image Prediction**:
```bash
python main.py predict --image path/to/mri_scan.jpg --output_dir ./results
```

**Clinical Report Generation**:
```bash
python main.py predict \
    --image path/to/scan.jpg \
    --clinical_report \
    --output_dir ./results
```

**Batch Processing**:
```bash
python main.py predict \
    --batch_dir ./test_images \
    --output_dir ./results
```

**Using Pre-trained Models**:
```bash
python main.py predict \
    --image path/to/scan.jpg \
    --classification_model ./models/classification_final.h5 \
    --segmentation_model ./models/segmentation_final.h5 \
    --output_dir ./results
```

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged precision
- **Recall (Sensitivity)**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve

### Segmentation Metrics
- **Dice Coefficient**: Overlap measure between predicted and true masks
- **IoU Score**: Intersection over Union
- **Pixel Accuracy**: Pixel-wise classification accuracy
- **Sensitivity/Specificity**: For tumor detection

### Clinical Metrics
- **Confidence Scores**: Model uncertainty quantification
- **Risk Assessment**: Low/Medium/High risk stratification
- **Tumor Burden**: Quantitative tumor area analysis

## 🔬 Clinical Applications

### Diagnostic Support
- **Screening**: Early detection of brain tumors
- **Classification**: Differentiation between tumor types
- **Monitoring**: Treatment response assessment
- **Second Opinion**: AI-assisted diagnosis validation

### Report Features
- **Visual Localization**: Tumor boundaries and bounding boxes
- **Quantitative Analysis**: Tumor area and percentage
- **Confidence Assessment**: Reliability indicators
- **Clinical Recommendations**: Risk-based action items

## 🧠 Model Details

### Preprocessing Pipeline
1. **Format Handling**: Automatic detection and loading of various image formats
2. **Intensity Normalization**: Robust normalization for varying image qualities
3. **Contrast Enhancement**: CLAHE for improved tissue contrast
4. **Denoising**: Advanced filtering for noise reduction
5. **Skull Stripping**: Optional brain extraction
6. **Augmentation**: Real-time data augmentation during training

### Training Strategy
1. **Transfer Learning**: Pre-trained ResNet50 initialization
2. **Two-Phase Training**: Initial training + fine-tuning
3. **Data Augmentation**: Rotation, flipping, brightness adjustment
4. **Regularization**: Dropout and batch normalization
5. **Early Stopping**: Prevent overfitting
6. **Learning Rate Scheduling**: Adaptive learning rate reduction

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for training

### Key Dependencies
- **TensorFlow**: 2.10.0+
- **OpenCV**: Image processing
- **SimpleITK**: DICOM support
- **nibabel**: NIfTI support
- **scikit-learn**: Metrics and evaluation
- **matplotlib/seaborn**: Visualization

## 🔧 Configuration

Key parameters can be modified in `src/utils/config.py`:

```python
# Model parameters
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4
BATCH_SIZE = 32

# Training parameters
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Classification threshold
CONFIDENCE_THRESHOLD = 0.5
```

## 📊 Expected Results

### Classification Performance
- **Accuracy**: >90% on balanced datasets
- **Sensitivity**: >85% for tumor detection
- **Specificity**: >95% for normal cases

### Segmentation Performance
- **Dice Coefficient**: >0.8 for tumor regions
- **IoU Score**: >0.7 for boundary detection

## 🚨 Important Notes

### Medical Disclaimer
- This system is designed for **research and educational purposes**
- **NOT** intended for direct clinical diagnosis
- Always consult qualified medical professionals
- Validate results with multiple imaging modalities

### Limitations
- Performance depends on image quality and dataset
- May require fine-tuning for specific populations
- Segmentation accuracy varies with tumor characteristics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{brain_tumor_detection_2024,
  title={Deep Learning-Based Automated Brain Tumor Detection and Classification from MRI Images},
  author={Your Name},
  year={2024},
  url={https://github.com/alinapradhan/Deep-Learning-Based-Automated-Brain-Tumor-Detection-and-Classification-from-MRI-Images}
}
```

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Include error messages and system information

---

**Note**: This system demonstrates state-of-the-art deep learning techniques for medical image analysis. Always validate results in clinical settings and consult medical professionals for diagnostic decisions.