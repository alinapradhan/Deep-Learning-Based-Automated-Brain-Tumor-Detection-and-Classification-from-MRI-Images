"""
Training script for brain tumor classification model
"""
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_model import TumorClassifier
from preprocessing.data_loader import BrainTumorDataLoader
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import ResultVisualizer
from utils.config import Config

class ClassificationTrainer:
    """
    Trainer class for brain tumor classification model
    """
    
    def __init__(self, data_dir, model_save_dir=None):
        """
        Initialize the trainer
        
        Args:
            data_dir: Directory containing training data
            model_save_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.model_save_dir = model_save_dir or Config.MODEL_SAVE_PATH
        
        # Create directories
        Config.create_directories()
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = BrainTumorDataLoader(
            data_dir=data_dir,
            image_size=Config.IMAGE_SIZE,
            batch_size=Config.BATCH_SIZE
        )
        
        self.model = TumorClassifier(
            input_shape=(*Config.IMAGE_SIZE, 3),
            num_classes=Config.NUM_CLASSES
        )
        
        self.evaluator = ModelEvaluator(class_names=Config.CLASS_NAMES)
        self.visualizer = ResultVisualizer(class_names=Config.CLASS_NAMES)
        
    def prepare_data(self):
        """Prepare training, validation, and test datasets"""
        print("Preparing datasets...")
        
        # Load and split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.load_classification_dataset(
            test_size=Config.TEST_SPLIT,
            val_size=Config.VALIDATION_SPLIT
        )
        
        # Create TensorFlow datasets
        self.train_dataset = self.data_loader.create_tf_dataset(
            X_train, y_train, shuffle=True, augment=True
        )
        
        self.val_dataset = self.data_loader.create_tf_dataset(
            X_val, y_val, shuffle=False, augment=False
        )
        
        self.test_dataset = self.data_loader.create_tf_dataset(
            X_test, y_test, shuffle=False, augment=False
        )
        
        # Store for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(Config.CHECKPOINT_PATH, 'classification_best.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        return callbacks
    
    def train(self, epochs=None, learning_rate=None):
        """
        Train the classification model
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training history
        """
        epochs = epochs or Config.EPOCHS
        learning_rate = learning_rate or Config.LEARNING_RATE
        
        print("Starting training...")
        
        # Prepare data
        train_ds, val_ds, test_ds = self.prepare_data()
        
        # Build and compile model
        self.model.compile_model(learning_rate=learning_rate)
        
        # Print model summary
        print("Model Architecture:")
        self.model.get_model_summary()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        history = self.model.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        print("\\nStarting fine-tuning phase...")
        self.model.fine_tune(learning_rate=learning_rate/10)
        
        # Continue training with fine-tuning
        fine_tune_epochs = epochs // 2
        history_fine = self.model.model.fit(
            train_ds,
            epochs=fine_tune_epochs,
            initial_epoch=len(history.history['loss']),
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
        
        # Save final model
        final_model_path = os.path.join(self.model_save_dir, 'classification_final.h5')
        self.model.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        return history
    
    def evaluate(self, save_results=True):
        """
        Evaluate the trained model
        
        Args:
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation metrics dictionary
        """
        print("Evaluating model...")
        
        # Make predictions on test set
        y_pred_proba = self.model.model.predict(self.test_dataset, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Calculate comprehensive metrics
        metrics = self.evaluator.calculate_classification_metrics(
            y_true, y_pred, y_pred_proba
        )
        
        # Print results
        print("\\nEvaluation Results:")
        print("=" * 50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"Sensitivity (Macro): {metrics['sensitivity_macro']:.4f}")
        print(f"Specificity (Macro): {metrics['specificity_macro']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            print(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
            print(f"PR AUC (Macro): {metrics['pr_auc_macro']:.4f}")
        
        # Per-class results
        print("\\nPer-Class Results:")
        print("-" * 30)
        for i, class_name in enumerate(Config.CLASS_NAMES):
            if i < len(metrics['precision_per_class']):
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
                print(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
                print(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}")
                print(f"  Sensitivity: {metrics['sensitivity_per_class'][i]:.4f}")
                print(f"  Specificity: {metrics['specificity_per_class'][i]:.4f}")
        
        if save_results:
            # Save evaluation results
            results_dir = os.path.join(self.model_save_dir, 'evaluation_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate and save comprehensive report
            self.evaluator.create_evaluation_report(metrics, save_dir=results_dir)
            
            # Generate and save classification report
            self.evaluator.generate_classification_report(
                self.y_test, y_pred, 
                save_path=os.path.join(results_dir, 'classification_report.csv')
            )
            
            # Plot and save confusion matrix
            self.evaluator.plot_confusion_matrix(
                self.y_test, y_pred,
                save_path=os.path.join(results_dir, 'confusion_matrix.png')
            )
            
            # Plot ROC curves
            self.evaluator.plot_roc_curves(
                self.y_test, y_pred_proba,
                save_path=os.path.join(results_dir, 'roc_curves.png')
            )
            
            # Plot PR curves
            self.evaluator.plot_precision_recall_curves(
                self.y_test, y_pred_proba,
                save_path=os.path.join(results_dir, 'pr_curves.png')
            )
            
            # Visualize sample predictions
            sample_indices = np.random.choice(len(self.X_test), min(8, len(self.X_test)), replace=False)
            sample_images = self.X_test[sample_indices]
            sample_predictions = y_pred_proba[sample_indices]
            sample_true = self.y_test[sample_indices]
            
            self.visualizer.plot_batch_predictions(
                sample_images, sample_predictions, sample_true,
                save_path=os.path.join(results_dir, 'sample_predictions.png')
            )
        
        return metrics
    
    def plot_training_history(self, history, save_path=None):
        """Plot and save training history"""
        self.visualizer.plot_training_history(
            history, 
            save_path=save_path or os.path.join(self.model_save_dir, 'training_history.png')
        )

def main():
    """Main function for command-line training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./models', 
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=Config.LEARNING_RATE, 
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, 
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.BATCH_SIZE = args.batch_size
    
    # Initialize trainer
    trainer = ClassificationTrainer(
        data_dir=args.data_dir,
        model_save_dir=args.output_dir
    )
    
    # Train model
    print("Starting classification model training...")
    history = trainer.train(epochs=args.epochs, learning_rate=args.learning_rate)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Evaluate model
    metrics = trainer.evaluate()
    
    print("\\nTraining completed successfully!")
    print(f"Models saved to: {args.output_dir}")

if __name__ == "__main__":
    main()