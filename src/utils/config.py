"""
Configuration settings for brain tumor detection system
"""
import os

class Config:
    # Data settings
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 4  # No tumor, Glioma, Meningioma, Pituitary
    
    # Training settings
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Model settings
    MODEL_SAVE_PATH = "models/saved_models/"
    CHECKPOINT_PATH = "models/checkpoints/"
    
    # Preprocessing settings
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Augmentation settings
    ROTATION_RANGE = 20
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    
    # Evaluation settings
    CONFIDENCE_THRESHOLD = 0.5
    
    # Class names
    CLASS_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_PATH, exist_ok=True)