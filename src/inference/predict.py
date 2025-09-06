"""
Main inference pipeline for brain tumor detection and classification
"""
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_model import TumorClassifier
from models.segmentation_model import TumorSegmentation
from preprocessing.image_utils import ImagePreprocessor
from evaluation.visualization import ResultVisualizer
from utils.config import Config
import tensorflow as tf

class BrainTumorPredictor:
    """
    Complete inference pipeline for brain tumor detection and classification
    """
    
    def __init__(self, classification_model_path=None, segmentation_model_path=None):
        """
        Initialize the predictor with trained models
        
        Args:
            classification_model_path: Path to trained classification model
            segmentation_model_path: Path to trained segmentation model
        """
        self.preprocessor = ImagePreprocessor(target_size=Config.IMAGE_SIZE)
        self.visualizer = ResultVisualizer(class_names=Config.CLASS_NAMES)
        
        # Initialize models
        self.classifier = TumorClassifier(
            input_shape=(*Config.IMAGE_SIZE, 3),
            num_classes=Config.NUM_CLASSES
        )
        
        self.segmentor = TumorSegmentation(
            input_shape=(*Config.IMAGE_SIZE, 3),
            num_classes=2  # Background and tumor
        )
        
        # Load trained models if paths provided
        if classification_model_path and os.path.exists(classification_model_path):
            try:
                self.classifier.load_model(classification_model_path)
                print(f"Loaded classification model from {classification_model_path}")
            except Exception as e:
                print(f"Error loading classification model: {e}")
                print("Building new classification model...")
                self.classifier.build_model()
        else:
            print("No classification model provided. Building new model...")
            self.classifier.build_model()
            
        if segmentation_model_path and os.path.exists(segmentation_model_path):
            try:
                self.segmentor.load_model(segmentation_model_path)
                print(f"Loaded segmentation model from {segmentation_model_path}")
            except Exception as e:
                print(f"Error loading segmentation model: {e}")
                print("Building new segmentation model...")
                self.segmentor.build_model()
        else:
            print("No segmentation model provided. Building new model...")
            self.segmentor.build_model()
    
    def predict_single_image(self, image_path, enhance_preprocessing=True, 
                           visualize=True, save_results=None):
        """
        Predict tumor type and location for a single image
        
        Args:
            image_path: Path to the input image
            enhance_preprocessing: Whether to apply advanced preprocessing
            visualize: Whether to display visualization
            save_results: Directory to save results (optional)
            
        Returns:
            dict: Complete prediction results
        """
        try:
            # Load and preprocess image
            print(f"Processing image: {image_path}")
            image = self.preprocessor.preprocess_image(image_path)
            
            if enhance_preprocessing:
                # Apply advanced preprocessing
                image = self.preprocessor.enhance_contrast(image.astype(np.uint8))
                image = self.preprocessor.denoise_image(image)
                image = image.astype(np.float32)
            
            # Classification prediction
            classification_result = self.classifier.predict_with_confidence(image)
            
            # Segmentation prediction
            segmentation_result = self.segmentor.predict_segmentation(image)
            
            # Combine results
            combined_result = {
                'classification': classification_result,
                'segmentation': segmentation_result,
                'input_image_path': image_path,
                'processed_image': image
            }
            
            # Add clinical interpretation
            combined_result['clinical_assessment'] = self._generate_clinical_assessment(
                classification_result, segmentation_result
            )
            
            # Visualization
            if visualize:
                self.visualizer.visualize_combined_results(
                    image, classification_result, segmentation_result,
                    save_path=os.path.join(save_results, 'combined_results.png') if save_results else None
                )
            
            # Save results if requested
            if save_results:
                self._save_prediction_results(combined_result, save_results)
            
            return combined_result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
    def predict_batch(self, image_paths, batch_size=8, save_results=None):
        """
        Predict tumor type and location for multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            save_results: Directory to save results (optional)
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images in batches of {batch_size}")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for image_path in batch_paths:
                result = self.predict_single_image(
                    image_path, 
                    visualize=False, 
                    save_results=None
                )
                if result:
                    batch_results.append(result)
            
            results.extend(batch_results)
            
            # Visualize batch
            if batch_results:
                images = [r['processed_image'] for r in batch_results]
                predictions = [r['classification'] for r in batch_results]
                
                self.visualizer.plot_batch_predictions(
                    images, predictions, batch_size=len(batch_results),
                    save_path=os.path.join(save_results, f'batch_{i//batch_size}.png') if save_results else None
                )
        
        # Save combined results
        if save_results:
            self._save_batch_results(results, save_results)
        
        return results
    
    def _generate_clinical_assessment(self, classification_result, segmentation_result):
        """Generate clinical assessment based on predictions"""
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        tumor_percentage = segmentation_result['tumor_percentage']
        has_tumor = segmentation_result['has_tumor']
        
        assessment = {
            'diagnosis': Config.CLASS_NAMES[predicted_class],
            'confidence_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
            'tumor_detected': has_tumor,
            'tumor_burden': 'High' if tumor_percentage > 5 else 'Medium' if tumor_percentage > 1 else 'Low',
            'urgency': 'Immediate' if predicted_class > 0 and confidence > 0.8 else 'Routine'
        }
        
        # Generate recommendation
        if predicted_class == 0:  # No tumor
            if confidence > 0.8:
                assessment['recommendation'] = 'No immediate intervention required. Regular follow-up recommended.'
            else:
                assessment['recommendation'] = 'Further evaluation recommended due to low confidence.'
        else:
            if confidence > 0.8:
                assessment['recommendation'] = 'Immediate medical consultation and further imaging recommended.'
            else:
                assessment['recommendation'] = 'Additional imaging and specialist consultation recommended.'
        
        # Risk stratification
        if predicted_class == 0 and confidence > 0.8:
            assessment['risk_level'] = 'Low'
        elif predicted_class > 0 and confidence > 0.8:
            assessment['risk_level'] = 'High'
        else:
            assessment['risk_level'] = 'Medium'
        
        return assessment
    
    def _save_prediction_results(self, result, save_dir):
        """Save prediction results to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed results as text
        report_path = os.path.join(save_dir, 'prediction_report.txt')
        with open(report_path, 'w') as f:
            f.write("Brain Tumor Detection - Prediction Report\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Classification results
            f.write("CLASSIFICATION RESULTS:\\n")
            f.write("-" * 25 + "\\n")
            class_result = result['classification']
            f.write(f"Predicted Class: {Config.CLASS_NAMES[class_result['predicted_class']]}\\n")
            f.write(f"Confidence: {class_result['confidence']:.4f}\\n\\n")
            
            f.write("Class Probabilities:\\n")
            for i, (name, prob) in enumerate(zip(Config.CLASS_NAMES, class_result['class_probabilities'])):
                f.write(f"  {name}: {prob:.4f}\\n")
            
            # Segmentation results
            f.write("\\nSEGMENTATION RESULTS:\\n")
            f.write("-" * 25 + "\\n")
            seg_result = result['segmentation']
            f.write(f"Tumor Detected: {seg_result['has_tumor']}\\n")
            f.write(f"Tumor Area: {seg_result['tumor_area_pixels']} pixels\\n")
            f.write(f"Tumor Percentage: {seg_result['tumor_percentage']:.2f}%\\n")
            
            if seg_result['bounding_box']:
                bbox = seg_result['bounding_box']
                f.write(f"Bounding Box: ({bbox['x']}, {bbox['y']}, {bbox['width']}, {bbox['height']})\\n")
            
            # Clinical assessment
            f.write("\\nCLINICAL ASSESSMENT:\\n")
            f.write("-" * 25 + "\\n")
            clinical = result['clinical_assessment']
            for key, value in clinical.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\\n")
        
        print(f"Prediction report saved to {report_path}")
    
    def _save_batch_results(self, results, save_dir):
        """Save batch results summary"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Create summary statistics
        summary_path = os.path.join(save_dir, 'batch_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Batch Processing Summary\\n")
            f.write("=" * 30 + "\\n\\n")
            
            total_images = len(results)
            f.write(f"Total Images Processed: {total_images}\\n\\n")
            
            # Count predictions by class
            class_counts = {name: 0 for name in Config.CLASS_NAMES}
            confidence_scores = []
            tumor_detections = 0
            
            for result in results:
                pred_class = result['classification']['predicted_class']
                class_counts[Config.CLASS_NAMES[pred_class]] += 1
                confidence_scores.append(result['classification']['confidence'])
                
                if result['segmentation']['has_tumor']:
                    tumor_detections += 1
            
            f.write("Classification Distribution:\\n")
            for class_name, count in class_counts.items():
                percentage = (count / total_images) * 100
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\\n")
            
            f.write(f"\\nTumor Localizations Detected: {tumor_detections} ({(tumor_detections/total_images)*100:.1f}%)\\n")
            f.write(f"Average Confidence: {np.mean(confidence_scores):.4f}\\n")
            f.write(f"Confidence Std Dev: {np.std(confidence_scores):.4f}\\n")
        
        print(f"Batch summary saved to {summary_path}")
    
    def create_clinical_report(self, image_path, patient_info=None, save_path=None):
        """
        Create a comprehensive clinical report for a single case
        
        Args:
            image_path: Path to the MRI image
            patient_info: Dictionary with patient information
            save_path: Path to save the clinical report
            
        Returns:
            dict: Complete clinical report
        """
        # Get prediction results
        result = self.predict_single_image(image_path, visualize=False)
        
        if result is None:
            return None
        
        # Create clinical visualization
        self.visualizer.create_clinical_report_visualization(
            result['processed_image'],
            result['classification'],
            result['segmentation'],
            patient_info=patient_info,
            save_path=save_path
        )
        
        # Add patient info to result
        if patient_info:
            result['patient_info'] = patient_info
        
        return result

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor Detection and Classification')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--batch', type=str, help='Directory containing multiple images')
    parser.add_argument('--classification_model', type=str, help='Path to classification model')
    parser.add_argument('--segmentation_model', type=str, help='Path to segmentation model')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--clinical_report', action='store_true', help='Generate clinical report')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BrainTumorPredictor(
        classification_model_path=args.classification_model,
        segmentation_model_path=args.segmentation_model
    )
    
    if args.image:
        # Single image prediction
        if args.clinical_report:
            predictor.create_clinical_report(args.image, save_path=os.path.join(args.output, 'clinical_report.png'))
        else:
            predictor.predict_single_image(args.image, save_results=args.output)
            
    elif args.batch:
        # Batch prediction
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.nii', '*.nii.gz', '*.dcm']:
            image_paths.extend(glob.glob(os.path.join(args.batch, ext)))
        
        if image_paths:
            predictor.predict_batch(image_paths, save_results=args.output)
        else:
            print(f"No images found in {args.batch}")
    else:
        print("Please provide either --image or --batch argument")

if __name__ == "__main__":
    main()