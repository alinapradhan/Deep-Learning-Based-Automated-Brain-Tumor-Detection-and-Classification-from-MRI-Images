"""
Comprehensive evaluation metrics for brain tumor detection and classification
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import os

class ModelEvaluator:
    """
    Comprehensive evaluation metrics for medical AI models
    """
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
        
    def calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels (categorical or one-hot)
            y_pred: Predicted labels (categorical or one-hot)
            y_pred_proba: Prediction probabilities
            
        Returns:
            dict: Comprehensive metrics dictionary
        """
        # Convert one-hot to categorical if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_cat = np.argmax(y_true, axis=1)
        else:
            y_true_cat = y_true
            
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_cat = np.argmax(y_pred, axis=1)
        else:
            y_pred_cat = y_pred
        
        # Basic metrics
        accuracy = accuracy_score(y_true_cat, y_pred_cat)
        
        # Per-class metrics
        precision_macro = precision_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_cat, y_pred_cat, average='macro', zero_division=0)
        
        precision_micro = precision_score(y_true_cat, y_pred_cat, average='micro', zero_division=0)
        recall_micro = recall_score(y_true_cat, y_pred_cat, average='micro', zero_division=0)
        f1_micro = f1_score(y_true_cat, y_pred_cat, average='micro', zero_division=0)
        
        # Per-class detailed metrics
        precision_per_class = precision_score(y_true_cat, y_pred_cat, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_cat, y_pred_cat, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_cat, y_pred_cat, average=None, zero_division=0)
        
        # Clinical metrics (sensitivity and specificity)
        cm = confusion_matrix(y_true_cat, y_pred_cat)
        
        # Calculate specificity and sensitivity for each class
        specificity_per_class = []
        sensitivity_per_class = []
        
        for i in range(len(self.class_names)):
            if i < cm.shape[0]:
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity = 0
                specificity = 0
                
            sensitivity_per_class.append(sensitivity)
            specificity_per_class.append(specificity)
        
        # Overall sensitivity and specificity
        sensitivity_macro = np.mean(sensitivity_per_class)
        specificity_macro = np.mean(specificity_per_class)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'sensitivity_macro': sensitivity_macro,
            'specificity_macro': specificity_macro,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'sensitivity_per_class': sensitivity_per_class,
            'specificity_per_class': specificity_per_class,
            'confusion_matrix': cm.tolist()
        }
        
        # Add AUC metrics if probabilities are provided
        if y_pred_proba is not None:
            auc_metrics = self._calculate_auc_metrics(y_true, y_pred_proba)
            metrics.update(auc_metrics)
        
        return metrics
    
    def _calculate_auc_metrics(self, y_true, y_pred_proba):
        """Calculate AUC-related metrics"""
        # Convert to categorical if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_cat = np.argmax(y_true, axis=1)
        else:
            y_true_cat = y_true
        
        # Binarize labels for multi-class AUC
        y_true_bin = label_binarize(y_true_cat, classes=range(len(self.class_names)))
        
        # Handle binary classification case
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Calculate ROC AUC for each class
        roc_auc_per_class = []
        for i in range(min(y_true_bin.shape[1], y_pred_proba.shape[1])):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                roc_auc_per_class.append(roc_auc)
            except:
                roc_auc_per_class.append(0.5)
        
        # Calculate PR AUC for each class
        pr_auc_per_class = []
        for i in range(min(y_true_bin.shape[1], y_pred_proba.shape[1])):
            try:
                ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
                pr_auc_per_class.append(ap)
            except:
                pr_auc_per_class.append(0.0)
        
        return {
            'roc_auc_per_class': roc_auc_per_class,
            'roc_auc_macro': np.mean(roc_auc_per_class),
            'pr_auc_per_class': pr_auc_per_class,
            'pr_auc_macro': np.mean(pr_auc_per_class)
        }
    
    def calculate_segmentation_metrics(self, y_true, y_pred, threshold=0.5):
        """
        Calculate segmentation metrics
        
        Args:
            y_true: True masks
            y_pred: Predicted masks
            threshold: Threshold for binary classification
            
        Returns:
            dict: Segmentation metrics
        """
        # Binarize predictions
        y_pred_bin = (y_pred > threshold).astype(np.uint8)
        y_true_bin = (y_true > 0.5).astype(np.uint8)
        
        # Flatten arrays for calculation
        y_true_flat = y_true_bin.flatten()
        y_pred_flat = y_pred_bin.flatten()
        
        # Calculate metrics
        intersection = np.sum(y_true_flat * y_pred_flat)
        union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
        
        # Dice coefficient
        dice = (2.0 * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat)) if (np.sum(y_true_flat) + np.sum(y_pred_flat)) > 0 else 0
        
        # IoU (Jaccard Index)
        iou = intersection / union if union > 0 else 0
        
        # Pixel accuracy
        accuracy = np.sum(y_true_flat == y_pred_flat) / len(y_true_flat)
        
        # Sensitivity and Specificity
        tp = intersection
        fn = np.sum(y_true_flat) - tp
        fp = np.sum(y_pred_flat) - tp
        tn = len(y_true_flat) - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'dice_coefficient': dice,
            'iou_score': iou,
            'pixel_accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        }
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """Generate and save detailed classification report"""
        # Convert one-hot to categorical if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_cat = np.argmax(y_true, axis=1)
        else:
            y_true_cat = y_true
            
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_cat = np.argmax(y_pred, axis=1)
        else:
            y_pred_cat = y_pred
        
        # Generate classification report
        report = classification_report(
            y_true_cat, 
            y_pred_cat, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Convert to DataFrame for better formatting
        df_report = pd.DataFrame(report).transpose()
        
        if save_path:
            df_report.to_csv(save_path)
            print(f"Classification report saved to {save_path}")
        
        return df_report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, figsize=(10, 8)):
        """Plot and save confusion matrix"""
        # Convert one-hot to categorical if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_cat = np.argmax(y_true, axis=1)
        else:
            y_true_cat = y_true
            
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_cat = np.argmax(y_pred, axis=1)
        else:
            y_pred_cat = y_pred
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_cat, y_pred_cat)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_path=None, figsize=(12, 8)):
        """Plot ROC curves for each class"""
        # Convert to categorical if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_cat = np.argmax(y_true, axis=1)
        else:
            y_true_cat = y_true
        
        # Binarize labels
        y_true_bin = label_binarize(y_true_cat, classes=range(len(self.class_names)))
        
        # Handle binary classification case
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for i in range(min(y_true_bin.shape[1], y_pred_proba.shape[1])):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})',
                linewidth=2
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Brain Tumor Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true, y_pred_proba, save_path=None, figsize=(12, 8)):
        """Plot Precision-Recall curves for each class"""
        # Convert to categorical if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_cat = np.argmax(y_true, axis=1)
        else:
            y_true_cat = y_true
        
        # Binarize labels
        y_true_bin = label_binarize(y_true_cat, classes=range(len(self.class_names)))
        
        # Handle binary classification case
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        plt.figure(figsize=figsize)
        
        # Plot PR curve for each class
        for i in range(min(y_true_bin.shape[1], y_pred_proba.shape[1])):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(
                recall, precision,
                label=f'{self.class_names[i]} (AP = {ap:.2f})',
                linewidth=2
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Brain Tumor Classification')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curves saved to {save_path}")
        
        plt.show()
    
    def create_evaluation_report(self, metrics, save_dir=None):
        """Create comprehensive evaluation report"""
        report = f"""
Brain Tumor Detection and Classification - Evaluation Report
===========================================================

Overall Performance Metrics:
----------------------------
Accuracy: {metrics['accuracy']:.4f}
Macro-averaged Precision: {metrics['precision_macro']:.4f}
Macro-averaged Recall (Sensitivity): {metrics['recall_macro']:.4f}
Macro-averaged F1-Score: {metrics['f1_macro']:.4f}
Macro-averaged Specificity: {metrics['specificity_macro']:.4f}

Clinical Significance:
---------------------
Sensitivity (True Positive Rate): {metrics['sensitivity_macro']:.4f}
Specificity (True Negative Rate): {metrics['specificity_macro']:.4f}

Per-Class Performance:
---------------------
"""
        
        for i, class_name in enumerate(self.class_names):
            if i < len(metrics['precision_per_class']):
                report += f"""
{class_name}:
  Precision: {metrics['precision_per_class'][i]:.4f}
  Recall: {metrics['recall_per_class'][i]:.4f}
  F1-Score: {metrics['f1_per_class'][i]:.4f}
  Sensitivity: {metrics['sensitivity_per_class'][i]:.4f}
  Specificity: {metrics['specificity_per_class'][i]:.4f}"""
        
        if 'roc_auc_macro' in metrics:
            report += f"""

AUC Metrics:
-----------
ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}
PR AUC (Macro): {metrics['pr_auc_macro']:.4f}
"""
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            report_path = os.path.join(save_dir, 'evaluation_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {report_path}")
        
        print(report)
        return report