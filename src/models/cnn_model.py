"""
CNN model for brain tumor classification
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import numpy as np

class TumorClassifier:
    """
    CNN model for brain tumor classification using ResNet50 backbone
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build the CNN model"""
        # Base model - try with pre-trained weights, fallback to None
        try:
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights ({str(e)})")
            print("Building model without pre-trained weights...")
            base_model = applications.ResNet50(
                weights=None,
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Preprocessing
        x = layers.Rescaling(1./255)(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer with softmax for classification
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=[
                CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
        
        return self.model
    
    def fine_tune(self, learning_rate=0.0001):
        """Enable fine-tuning of the base model"""
        if self.model is None:
            raise ValueError("Model must be built first")
            
        # Unfreeze the base model
        base_model = self.model.layers[2]  # ResNet50 base model
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=[
                CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
        
    def predict_with_confidence(self, image):
        """
        Make prediction with confidence scores
        
        Args:
            image: Input image array
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if self.model is None:
            raise ValueError("Model must be built and loaded first")
            
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Get prediction probabilities
        predictions = self.model.predict(image, verbose=0)
        
        # Get class probabilities and predicted class
        class_probabilities = predictions[0]
        predicted_class = np.argmax(class_probabilities)
        confidence = float(class_probabilities[predicted_class])
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': confidence,
            'class_probabilities': class_probabilities.tolist(),
            'all_predictions': predictions[0].tolist()
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model must be built first")
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        return self.model
        
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()