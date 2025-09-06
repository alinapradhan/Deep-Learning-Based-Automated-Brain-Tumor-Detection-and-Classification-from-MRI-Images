"""
Main entry point for Brain Tumor Detection and Classification System
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.train_classifier import ClassificationTrainer
from src.inference.predict import BrainTumorPredictor
from src.utils.config import Config
import tensorflow as tf

def setup_environment():
    """Setup the environment for optimal performance"""
    # Set GPU memory growth to avoid memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    import numpy as np
    np.random.seed(42)

def train_models(args):
    """Train classification and segmentation models"""
    print("Starting model training...")
    
    # Train classification model
    print("\\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*60)
    
    trainer = ClassificationTrainer(
        data_dir=args.data_dir,
        model_save_dir=args.output_dir
    )
    
    history = trainer.train(epochs=args.epochs, learning_rate=args.learning_rate)
    trainer.plot_training_history(history)
    metrics = trainer.evaluate()
    
    print(f"\\nClassification model training completed!")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    
    # Note: Segmentation training would be similar but requires mask data
    print("\\nNote: Segmentation model training requires mask annotations.")
    print("The system includes a pre-built U-Net architecture ready for training.")

def run_inference(args):
    """Run inference on new images"""
    print("Starting inference...")
    
    predictor = BrainTumorPredictor(
        classification_model_path=args.classification_model,
        segmentation_model_path=args.segmentation_model
    )
    
    if args.image:
        # Single image prediction
        print(f"Processing single image: {args.image}")
        if args.clinical_report:
            result = predictor.create_clinical_report(
                args.image, 
                save_path=os.path.join(args.output_dir, 'clinical_report.png')
            )
        else:
            result = predictor.predict_single_image(
                args.image, 
                save_results=args.output_dir
            )
        
        if result:
            print("\\nPrediction Results:")
            print("-" * 30)
            class_result = result['classification']
            print(f"Predicted Class: {Config.CLASS_NAMES[class_result['predicted_class']]}")
            print(f"Confidence: {class_result['confidence']:.4f}")
            
            seg_result = result['segmentation']
            print(f"Tumor Detected: {seg_result['has_tumor']}")
            print(f"Tumor Area: {seg_result['tumor_percentage']:.2f}%")
    
    elif args.batch_dir:
        # Batch prediction
        print(f"Processing batch from directory: {args.batch_dir}")
        import glob
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.batch_dir, ext)))
        
        if image_paths:
            results = predictor.predict_batch(image_paths, save_results=args.output_dir)
            print(f"\\nProcessed {len(results)} images successfully")
        else:
            print(f"No images found in {args.batch_dir}")

def create_demo_data(data_dir):
    """Create demo dataset structure for testing"""
    print(f"Creating demo dataset structure in {data_dir}")
    
    class_dirs = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
    
    for class_name in class_dirs:
        class_path = os.path.join(data_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
    
    print("Demo dataset structure created.")
    print("Please add your MRI images to the respective class folders:")
    for class_name in class_dirs:
        print(f"  - {os.path.join(data_dir, class_name)}/")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection and Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models with default settings
  python main.py train --data_dir ./data --output_dir ./models
  
  # Run inference on a single image
  python main.py predict --image path/to/mri_scan.jpg --output_dir ./results
  
  # Generate clinical report
  python main.py predict --image path/to/scan.jpg --clinical_report --output_dir ./results
  
  # Process batch of images
  python main.py predict --batch_dir ./test_images --output_dir ./results
  
  # Create demo dataset structure
  python main.py create_demo --data_dir ./demo_data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the models')
    train_parser.add_argument('--data_dir', type=str, default='./data', 
                             help='Directory containing training data')
    train_parser.add_argument('--output_dir', type=str, default='./models', 
                             help='Directory to save trained models')
    train_parser.add_argument('--epochs', type=int, default=50, 
                             help='Number of training epochs')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, 
                             help='Learning rate')
    train_parser.add_argument('--batch_size', type=int, default=32, 
                             help='Batch size')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Run inference')
    predict_parser.add_argument('--image', type=str, help='Path to single image')
    predict_parser.add_argument('--batch_dir', type=str, help='Directory with multiple images')
    predict_parser.add_argument('--classification_model', type=str, 
                               help='Path to classification model')
    predict_parser.add_argument('--segmentation_model', type=str, 
                               help='Path to segmentation model')
    predict_parser.add_argument('--output_dir', type=str, default='./results', 
                               help='Output directory for results')
    predict_parser.add_argument('--clinical_report', action='store_true', 
                               help='Generate clinical report')
    
    # Demo data creation command
    demo_parser = subparsers.add_parser('create_demo', help='Create demo dataset structure')
    demo_parser.add_argument('--data_dir', type=str, default='./demo_data', 
                            help='Directory to create demo structure')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    if args.command == 'train':
        train_models(args)
    elif args.command == 'predict':
        if not args.image and not args.batch_dir:
            print("Error: Please provide either --image or --batch_dir for prediction")
            return
        run_inference(args)
    elif args.command == 'create_demo':
        create_demo_data(args.data_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()