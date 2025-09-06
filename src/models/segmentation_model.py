"""
U-Net model for brain tumor segmentation and localization
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np

class TumorSegmentation:
    """
    U-Net model for brain tumor segmentation
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes  # Background and tumor
        self.model = None
        
    def conv_block(self, inputs, num_filters):
        """Convolutional block with batch normalization"""
        x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        return x
    
    def encoder_block(self, inputs, num_filters):
        """Encoder block with downsampling"""
        x = self.conv_block(inputs, num_filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p
    
    def decoder_block(self, inputs, skip_features, num_filters):
        """Decoder block with upsampling"""
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x
    
    def build_model(self):
        """Build the U-Net model"""
        inputs = layers.Input(self.input_shape)
        
        # Preprocessing
        x = layers.Rescaling(1./255)(inputs)
        
        # Encoder
        s1, p1 = self.encoder_block(x, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)
        
        # Bridge
        b1 = self.conv_block(p4, 1024)
        
        # Decoder
        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        # Output layer
        if self.num_classes == 1:
            activation = "sigmoid"
        else:
            activation = "softmax"
            
        outputs = layers.Conv2D(self.num_classes, 1, padding="same", activation=activation)(d4)
        
        self.model = models.Model(inputs, outputs, name="unet")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        if self.model is None:
            self.build_model()
            
        if self.num_classes == 1:
            loss = "binary_crossentropy"
            metrics = ["accuracy", self.dice_coefficient, self.iou_score]
        else:
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy", self.dice_coefficient, self.iou_score]
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    @staticmethod
    def iou_score(y_true, y_pred, smooth=1e-6):
        """Intersection over Union (IoU) score"""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def predict_segmentation(self, image):
        """
        Predict segmentation mask for tumor localization
        
        Args:
            image: Input image array
            
        Returns:
            dict: Segmentation results with mask and bounding box
        """
        if self.model is None:
            raise ValueError("Model must be built and loaded first")
            
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Get segmentation prediction
        prediction = self.model.predict(image, verbose=0)
        
        # Process prediction
        if self.num_classes == 1:
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
        else:
            mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
            
        # Calculate bounding box if tumor is detected
        bbox = self.get_bounding_box(mask)
        
        # Calculate tumor area
        tumor_area = np.sum(mask > 0) if self.num_classes == 1 else np.sum(mask == 1)
        total_area = mask.shape[0] * mask.shape[1]
        tumor_percentage = (tumor_area / total_area) * 100
        
        return {
            'segmentation_mask': mask,
            'raw_prediction': prediction[0],
            'bounding_box': bbox,
            'tumor_area_pixels': int(tumor_area),
            'tumor_percentage': float(tumor_percentage),
            'has_tumor': tumor_area > 0
        }
    
    def get_bounding_box(self, mask):
        """
        Extract bounding box from segmentation mask
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            dict: Bounding box coordinates (x, y, width, height)
        """
        tumor_pixels = np.where(mask > 0) if len(mask.shape) == 2 else np.where(mask == 1)
        
        if len(tumor_pixels[0]) == 0:
            return None
            
        y_min, y_max = tumor_pixels[0].min(), tumor_pixels[0].max()
        x_min, x_max = tumor_pixels[1].min(), tumor_pixels[1].max()
        
        return {
            'x': int(x_min),
            'y': int(y_min),
            'width': int(x_max - x_min + 1),
            'height': int(y_max - y_min + 1)
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model must be built first")
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                'dice_coefficient': self.dice_coefficient,
                'iou_score': self.iou_score
            }
        )
        return self.model
        
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()