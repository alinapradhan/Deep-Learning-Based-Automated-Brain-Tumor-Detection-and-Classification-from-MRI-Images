"""
Visualization tools for brain tumor detection results
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

class ResultVisualizer:
    """
    Visualization tools for brain tumor detection and segmentation results
    """
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        
    def visualize_classification_result(self, image, prediction_result, save_path=None, figsize=(12, 8)):
        """
        Visualize classification result with confidence scores
        
        Args:
            image: Input image array
            prediction_result: Prediction result dictionary from model
            save_path: Path to save the visualization
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Input MRI Image')
        axes[0].axis('off')
        
        # Prediction results
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        class_probabilities = prediction_result['class_probabilities']
        
        # Bar plot of class probabilities
        y_pos = np.arange(len(self.class_names))
        bars = axes[1].barh(y_pos, class_probabilities, color='skyblue', alpha=0.7)
        
        # Highlight predicted class
        bars[predicted_class].set_color('red')
        bars[predicted_class].set_alpha(1.0)
        
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(self.class_names)
        axes[1].set_xlabel('Probability')
        axes[1].set_title(f'Prediction: {self.class_names[predicted_class]}\nConfidence: {confidence:.3f}')
        axes[1].set_xlim(0, 1)
        
        # Add probability labels on bars
        for i, (bar, prob) in enumerate(zip(bars, class_probabilities)):
            axes[1].text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification result saved to {save_path}")
        
        plt.show()
    
    def visualize_segmentation_result(self, image, segmentation_result, save_path=None, figsize=(15, 5)):
        """
        Visualize segmentation result with tumor localization
        
        Args:
            image: Input image array
            segmentation_result: Segmentation result dictionary from model
            save_path: Path to save the visualization
            figsize: Figure size tuple
        """
        mask = segmentation_result['segmentation_mask']
        bbox = segmentation_result.get('bounding_box')
        tumor_percentage = segmentation_result['tumor_percentage']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Original MRI Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='hot', alpha=0.8)
        axes[1].set_title('Tumor Segmentation Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image.astype(np.uint8))
        
        # Create colored mask overlay
        mask_colored = np.zeros((*mask.shape, 3))
        mask_colored[mask > 0] = [1, 0, 0]  # Red for tumor
        axes[2].imshow(mask_colored, alpha=0.3)
        
        # Add bounding box if available
        if bbox is not None:
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), 
                bbox['width'], 
                bbox['height'],
                linewidth=2, 
                edgecolor='yellow', 
                facecolor='none'
            )
            axes[2].add_patch(rect)
        
        axes[2].set_title(f'Tumor Localization\nTumor Area: {tumor_percentage:.2f}%')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Segmentation result saved to {save_path}")
        
        plt.show()
    
    def visualize_combined_results(self, image, classification_result, segmentation_result, 
                                 save_path=None, figsize=(20, 6)):
        """
        Visualize both classification and segmentation results together
        
        Args:
            image: Input image array
            classification_result: Classification result dictionary
            segmentation_result: Segmentation result dictionary
            save_path: Path to save the visualization
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Input MRI Image')
        axes[0].axis('off')
        
        # Classification probabilities
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        class_probabilities = classification_result['class_probabilities']
        
        y_pos = np.arange(len(self.class_names))
        bars = axes[1].barh(y_pos, class_probabilities, color='skyblue', alpha=0.7)
        bars[predicted_class].set_color('red')
        bars[predicted_class].set_alpha(1.0)
        
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(self.class_names)
        axes[1].set_xlabel('Probability')
        axes[1].set_title(f'Classification\n{self.class_names[predicted_class]} ({confidence:.3f})')
        axes[1].set_xlim(0, 1)
        
        # Segmentation mask
        mask = segmentation_result['segmentation_mask']
        axes[2].imshow(mask, cmap='hot')
        axes[2].set_title('Segmentation Mask')
        axes[2].axis('off')
        
        # Combined overlay
        axes[3].imshow(image.astype(np.uint8))
        
        # Overlay segmentation
        mask_colored = np.zeros((*mask.shape, 3))
        mask_colored[mask > 0] = [1, 0, 0]
        axes[3].imshow(mask_colored, alpha=0.4)
        
        # Add bounding box
        bbox = segmentation_result.get('bounding_box')
        if bbox is not None:
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), 
                bbox['width'], 
                bbox['height'],
                linewidth=2, 
                edgecolor='yellow', 
                facecolor='none'
            )
            axes[3].add_patch(rect)
        
        tumor_percentage = segmentation_result['tumor_percentage']
        axes[3].set_title(f'Combined Results\nTumor: {tumor_percentage:.2f}%')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined results saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None, figsize=(15, 5)):
        """
        Plot training history curves
        
        Args:
            history: Training history from Keras model.fit()
            save_path: Path to save the plot
            figsize: Figure size tuple
        """
        metrics = list(history.history.keys())
        train_metrics = [m for m in metrics if not m.startswith('val_')]
        
        n_metrics = len(train_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(train_metrics):
            train_values = history.history[metric]
            val_metric = f'val_{metric}'
            
            axes[i].plot(train_values, label=f'Training {metric}', linewidth=2)
            
            if val_metric in history.history:
                val_values = history.history[val_metric]
                axes[i].plot(val_values, label=f'Validation {metric}', linewidth=2)
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
    
    def create_clinical_report_visualization(self, image, classification_result, 
                                           segmentation_result, patient_info=None, 
                                           save_path=None, figsize=(16, 10)):
        """
        Create a clinical report-style visualization
        
        Args:
            image: Input MRI image
            classification_result: Classification results
            segmentation_result: Segmentation results
            patient_info: Optional patient information dictionary
            save_path: Path to save the report
            figsize: Figure size tuple
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[0.1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Title
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.text(0.5, 0.5, 'Brain Tumor Detection - Clinical Report', 
                     fontsize=20, fontweight='bold', ha='center', va='center')
        title_ax.axis('off')
        
        # Patient info (if provided)
        if patient_info:
            info_text = f"Patient ID: {patient_info.get('id', 'N/A')}\n"
            info_text += f"Age: {patient_info.get('age', 'N/A')}\n"
            info_text += f"Gender: {patient_info.get('gender', 'N/A')}\n"
            info_text += f"Scan Date: {patient_info.get('scan_date', 'N/A')}"
            
            title_ax.text(0.1, 0.5, info_text, fontsize=12, va='center')
        
        # Original image
        img_ax = fig.add_subplot(gs[1, 0])
        img_ax.imshow(image.astype(np.uint8))
        img_ax.set_title('Original MRI Scan', fontsize=14, fontweight='bold')
        img_ax.axis('off')
        
        # Segmentation overlay
        overlay_ax = fig.add_subplot(gs[1, 1])
        overlay_ax.imshow(image.astype(np.uint8))
        
        mask = segmentation_result['segmentation_mask']
        mask_colored = np.zeros((*mask.shape, 3))
        mask_colored[mask > 0] = [1, 0, 0]
        overlay_ax.imshow(mask_colored, alpha=0.4)
        
        bbox = segmentation_result.get('bounding_box')
        if bbox:
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=3, edgecolor='yellow', facecolor='none'
            )
            overlay_ax.add_patch(rect)
        
        overlay_ax.set_title('Tumor Localization', fontsize=14, fontweight='bold')
        overlay_ax.axis('off')
        
        # Classification results
        class_ax = fig.add_subplot(gs[1, 2:])
        
        predicted_class = classification_result['predicted_class']
        confidence = classification_result['confidence']
        class_probabilities = classification_result['class_probabilities']
        
        y_pos = np.arange(len(self.class_names))
        bars = class_ax.barh(y_pos, class_probabilities, color='lightblue', alpha=0.7)
        bars[predicted_class].set_color('red')
        bars[predicted_class].set_alpha(1.0)
        
        class_ax.set_yticks(y_pos)
        class_ax.set_yticklabels(self.class_names)
        class_ax.set_xlabel('Probability')
        class_ax.set_title('Classification Results', fontsize=14, fontweight='bold')
        class_ax.set_xlim(0, 1)
        
        # Add probability labels
        for i, (bar, prob) in enumerate(zip(bars, class_probabilities)):
            class_ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                         f'{prob:.3f}', va='center', fontsize=12, fontweight='bold')
        
        # Clinical summary
        summary_ax = fig.add_subplot(gs[2, :])
        summary_ax.axis('off')
        
        # Create summary text
        diagnosis = self.class_names[predicted_class]
        tumor_area = segmentation_result['tumor_percentage']
        
        summary_text = f"""
DIAGNOSIS: {diagnosis}
Confidence Level: {confidence:.1%}
        
TUMOR ANALYSIS:
• Tumor Area: {tumor_area:.2f}% of brain region
• Location: {'Detected' if bbox else 'Not localized'}
"""
        
        if bbox:
            summary_text += f"• Bounding Box: {bbox['width']}×{bbox['height']} pixels\n"
        
        # Risk assessment
        if predicted_class == 0:  # No tumor
            risk_level = "Low"
            recommendation = "Regular follow-up recommended"
        elif confidence > 0.8:
            risk_level = "High"
            recommendation = "Immediate medical consultation recommended"
        else:
            risk_level = "Medium"
            recommendation = "Further imaging and medical evaluation recommended"
        
        summary_text += f"""
RISK ASSESSMENT: {risk_level}
RECOMMENDATION: {recommendation}

NOTE: This is an AI-assisted diagnosis. Final diagnosis should always be confirmed by a qualified radiologist.
"""
        
        summary_ax.text(0.05, 0.95, summary_text, fontsize=12, va='top', ha='left', 
                       transform=summary_ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Clinical report saved to {save_path}")
        
        plt.show()
    
    def plot_batch_predictions(self, images, predictions, true_labels=None, 
                             batch_size=8, save_path=None, figsize=(20, 15)):
        """
        Visualize predictions for a batch of images
        
        Args:
            images: Batch of input images
            predictions: Batch of prediction results
            true_labels: True labels (optional)
            batch_size: Number of images to display
            save_path: Path to save the visualization
            figsize: Figure size tuple
        """
        n_images = min(batch_size, len(images))
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i in range(n_images):
            axes[i].imshow(images[i].astype(np.uint8))
            
            # Get prediction info
            if isinstance(predictions[i], dict):
                pred_class = predictions[i]['predicted_class']
                confidence = predictions[i]['confidence']
            else:
                pred_class = np.argmax(predictions[i])
                confidence = np.max(predictions[i])
            
            pred_label = self.class_names[pred_class]
            
            # Create title with prediction and truth (if available)
            title = f"Pred: {pred_label}\nConf: {confidence:.3f}"
            
            if true_labels is not None:
                true_class = np.argmax(true_labels[i]) if len(true_labels[i].shape) > 0 else true_labels[i]
                true_label = self.class_names[true_class]
                title += f"\nTrue: {true_label}"
                
                # Color code: green for correct, red for incorrect
                color = 'green' if pred_class == true_class else 'red'
                axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
            else:
                axes[i].set_title(title, fontsize=10, fontweight='bold')
            
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Batch predictions saved to {save_path}")
        
        plt.show()