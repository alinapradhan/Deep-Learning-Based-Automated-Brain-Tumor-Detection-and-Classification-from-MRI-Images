#!/usr/bin/env python3
"""
Demo script to showcase the Brain Tumor Detection System functionality
"""

import os
import sys
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_demo_images():
    """Create sample brain scan images for demonstration"""
    print("Creating demo brain scan images...")
    
    os.makedirs('demo_images', exist_ok=True)
    
    # Create realistic-looking brain scan patterns
    for i, tumor_type in enumerate(['normal', 'glioma', 'meningioma', 'pituitary']):
        # Base brain-like pattern
        image = np.random.randint(20, 60, (224, 224, 3), dtype=np.uint8)
        
        # Add brain-like circular structure
        center = (112, 112)
        Y, X = np.ogrid[:224, :224]
        brain_mask = (X - center[0])**2 + (Y - center[1])**2 <= 100**2
        image[brain_mask] += 50
        
        # Add tumor-specific patterns
        if tumor_type != 'normal':
            # Add tumor region
            tumor_center = (100 + i*10, 100 + i*10)
            tumor_mask = (X - tumor_center[0])**2 + (Y - tumor_center[1])**2 <= (15 + i*5)**2
            image[tumor_mask] = [200, 150, 100]  # Tumor-like appearance
        
        # Ensure valid range
        image = np.clip(image, 0, 255)
        
        # Save image
        img = Image.fromarray(image.astype(np.uint8))
        img.save(f'demo_images/{tumor_type}_scan_{i+1}.jpg')
    
    print(f"Created {len(['normal', 'glioma', 'meningioma', 'pituitary'])} demo images in demo_images/")

def demo_classification():
    """Demonstrate classification functionality"""
    print("\n" + "="*60)
    print("BRAIN TUMOR CLASSIFICATION DEMO")
    print("="*60)
    
    from src.models.cnn_model import TumorClassifier
    from src.preprocessing.image_utils import ImagePreprocessor
    
    # Initialize components
    classifier = TumorClassifier()
    classifier.build_model()
    preprocessor = ImagePreprocessor()
    
    print("✓ Classification model built successfully")
    print(f"✓ Model has {classifier.model.count_params():,} parameters")
    
    # Test with demo image
    if os.path.exists('demo_images/glioma_scan_2.jpg'):
        image = preprocessor.preprocess_image('demo_images/glioma_scan_2.jpg')
        result = classifier.predict_with_confidence(image)
        
        print(f"\nSample Prediction Results:")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All probabilities: {[f'{p:.4f}' for p in result['class_probabilities']]}")

def demo_segmentation():
    """Demonstrate segmentation functionality"""
    print("\n" + "="*60)
    print("BRAIN TUMOR SEGMENTATION DEMO")
    print("="*60)
    
    from src.models.segmentation_model import TumorSegmentation
    from src.preprocessing.image_utils import ImagePreprocessor
    
    # Initialize components
    segmentor = TumorSegmentation()
    segmentor.build_model()
    preprocessor = ImagePreprocessor()
    
    print("✓ Segmentation model built successfully")
    print(f"✓ Model has {segmentor.model.count_params():,} parameters")
    
    # Test with demo image
    if os.path.exists('demo_images/meningioma_scan_3.jpg'):
        image = preprocessor.preprocess_image('demo_images/meningioma_scan_3.jpg')
        result = segmentor.predict_segmentation(image)
        
        print(f"\nSample Segmentation Results:")
        print(f"Tumor Detected: {result['has_tumor']}")
        print(f"Tumor Area: {result['tumor_area_pixels']} pixels")
        print(f"Tumor Percentage: {result['tumor_percentage']:.2f}%")
        if result['bounding_box']:
            bbox = result['bounding_box']
            print(f"Bounding Box: ({bbox['x']}, {bbox['y']}, {bbox['width']}, {bbox['height']})")

def demo_evaluation_metrics():
    """Demonstrate evaluation metrics"""
    print("\n" + "="*60)
    print("EVALUATION METRICS DEMO")
    print("="*60)
    
    from src.evaluation.metrics import ModelEvaluator
    
    # Create synthetic predictions for demonstration
    n_samples = 100
    y_true = np.random.randint(0, 4, n_samples)
    y_pred = np.random.randint(0, 4, n_samples)
    y_pred_proba = np.random.rand(n_samples, 4)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    
    print("✓ Comprehensive metrics calculated")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Sensitivity (Macro): {metrics['sensitivity_macro']:.4f}")
    print(f"Specificity (Macro): {metrics['specificity_macro']:.4f}")

def demo_complete_pipeline():
    """Demonstrate complete inference pipeline"""
    print("\n" + "="*60)
    print("COMPLETE INFERENCE PIPELINE DEMO")
    print("="*60)
    
    from src.inference.predict import BrainTumorPredictor
    
    if not os.path.exists('demo_images/pituitary_scan_4.jpg'):
        print("Demo images not found. Creating them...")
        create_demo_images()
    
    # Initialize predictor
    predictor = BrainTumorPredictor()
    print("✓ Complete prediction pipeline initialized")
    
    # Run prediction
    result = predictor.predict_single_image(
        'demo_images/pituitary_scan_4.jpg',
        visualize=False,
        save_results=None
    )
    
    if result:
        print("✓ Prediction completed successfully")
        
        # Classification results
        classification = result['classification']
        print(f"\nClassification Results:")
        print(f"  Predicted Class: {classification['predicted_class']}")
        print(f"  Confidence: {classification['confidence']:.4f}")
        
        # Segmentation results
        segmentation = result['segmentation']
        print(f"\nSegmentation Results:")
        print(f"  Tumor Detected: {segmentation['has_tumor']}")
        print(f"  Tumor Area: {segmentation['tumor_percentage']:.2f}%")
        
        # Clinical assessment
        clinical = result['clinical_assessment']
        print(f"\nClinical Assessment:")
        print(f"  Diagnosis: {clinical['diagnosis']}")
        print(f"  Confidence Level: {clinical['confidence_level']}")
        print(f"  Risk Level: {clinical['risk_level']}")
        print(f"  Recommendation: {clinical['recommendation']}")

def main():
    """Run the complete demonstration"""
    print("Brain Tumor Detection System - Comprehensive Demo")
    print("="*60)
    print("This demo showcases all the key features of the system:")
    print("• Deep learning models for classification and segmentation")
    print("• Comprehensive evaluation metrics")
    print("• Clinical-grade reporting")
    print("• End-to-end inference pipeline")
    
    try:
        # Create demo data
        create_demo_images()
        
        # Run component demos
        demo_classification()
        demo_segmentation()
        demo_evaluation_metrics()
        demo_complete_pipeline()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ ResNet50-based tumor classification")
        print("✓ U-Net tumor segmentation and localization")
        print("✓ Comprehensive clinical metrics")
        print("✓ End-to-end prediction pipeline")
        print("✓ Clinical assessment and reporting")
        print("\nThe system is ready for:")
        print("• Training on real MRI datasets")
        print("• Clinical evaluation and validation")
        print("• Integration with medical imaging workflows")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("This may be due to missing dependencies or environment issues.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())