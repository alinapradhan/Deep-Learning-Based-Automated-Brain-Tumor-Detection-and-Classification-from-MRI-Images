"""
Data loading and preprocessing pipeline for brain tumor dataset
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from .image_utils import ImagePreprocessor
import glob
from tqdm import tqdm

class BrainTumorDataLoader:
    """
    Data loader for brain tumor classification and segmentation datasets
    """
    
    def __init__(self, data_dir, image_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocessor = ImagePreprocessor(target_size=image_size)
        self.label_encoder = LabelEncoder()
        
    def load_classification_dataset(self, test_size=0.2, val_size=0.1):
        """
        Load and prepare classification dataset
        
        Expected directory structure:
        data_dir/
        ├── no_tumor/
        ├── glioma/
        ├── meningioma/
        └── pituitary/
        
        Args:
            test_size: Proportion of test set
            val_size: Proportion of validation set
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Loading classification dataset...")
        
        # Collect all images and labels
        images = []
        labels = []
        
        # Define class directories
        class_dirs = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
        
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found. Creating sample structure...")
                self._create_sample_dataset()
                break
                
            # Get all image files in the class directory
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.nii', '*.nii.gz', '*.dcm']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_path, ext)))
            
            print(f"Found {len(image_files)} images in {class_name}")
            
            # Load and preprocess images
            for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                try:
                    image = self.preprocessor.preprocess_image(img_path)
                    images.append(image)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
                    continue
        
        if len(images) == 0:
            print("No images found. Creating synthetic dataset for demonstration...")
            return self._create_synthetic_dataset(test_size, val_size)
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split dataset
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, stratify=y_encoded, random_state=42
        )
        
        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, stratify=np.argmax(y_temp, axis=1), random_state=42
        )
        
        print(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_synthetic_dataset(self, test_size=0.2, val_size=0.1):
        """Create synthetic dataset for demonstration"""
        print("Creating synthetic dataset for demonstration...")
        
        # Create synthetic images and labels
        num_samples = 1000
        X = np.random.rand(num_samples, *self.image_size, 3) * 255
        X = X.astype(np.float32)
        
        # Create balanced labels
        samples_per_class = num_samples // 4
        y = []
        for i in range(4):
            y.extend([i] * samples_per_class)
        
        y = np.array(y[:num_samples])
        y_categorical = to_categorical(y, num_classes=4)
        
        # Fit label encoder with class names
        class_names = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
        self.label_encoder.fit(class_names)
        
        # Split dataset
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, stratify=y, random_state=42
        )
        
        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, stratify=np.argmax(y_temp, axis=1), random_state=42
        )
        
        print(f"Synthetic dataset created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_sample_dataset(self):
        """Create sample dataset structure"""
        class_dirs = ['no_tumor', 'glioma', 'meningioma', 'pituitary']
        
        for class_name in class_dirs:
            class_path = os.path.join(self.data_dir, class_name)
            os.makedirs(class_path, exist_ok=True)
            
        print(f"Created sample dataset structure in {self.data_dir}")
        print("Please add your MRI images to the respective class folders.")
    
    def create_tf_dataset(self, X, y, shuffle=True, augment=False):
        """
        Create TensorFlow dataset from arrays
        
        Args:
            X: Image arrays
            y: Label arrays
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        if augment:
            dataset = dataset.map(self._augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_image(self, image, label):
        """Apply random augmentations to image"""
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0.0, 255.0)
        
        return image, label
    
    def load_segmentation_dataset(self, images_dir, masks_dir, test_size=0.2):
        """
        Load dataset for segmentation task
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing segmentation masks
            test_size: Proportion of test set
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Loading segmentation dataset...")
        
        # Check if directories exist
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print("Segmentation directories not found. Creating synthetic segmentation dataset...")
            return self._create_synthetic_segmentation_dataset(test_size)
        
        # Get image and mask files
        image_files = glob.glob(os.path.join(images_dir, "*"))
        mask_files = glob.glob(os.path.join(masks_dir, "*"))
        
        # Match images with masks
        images = []
        masks = []
        
        for img_file in tqdm(image_files, desc="Loading segmentation data"):
            # Find corresponding mask file
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            mask_file = None
            
            for mask_path in mask_files:
                mask_name = os.path.splitext(os.path.basename(mask_path))[0]
                if img_name in mask_name or mask_name in img_name:
                    mask_file = mask_path
                    break
            
            if mask_file is None:
                continue
            
            try:
                # Load and preprocess image
                image = self.preprocessor.preprocess_image(img_file)
                images.append(image)
                
                # Load and preprocess mask
                mask = self.preprocessor.load_image(mask_file)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]  # Take first channel if RGB
                mask = (mask > 127).astype(np.uint8)  # Binarize
                masks.append(mask)
                
            except Exception as e:
                print(f"Error loading {img_file}: {str(e)}")
                continue
        
        if len(images) == 0:
            print("No segmentation data found. Creating synthetic dataset...")
            return self._create_synthetic_segmentation_dataset(test_size)
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(masks)
        y = np.expand_dims(y, axis=-1)  # Add channel dimension
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Segmentation dataset split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _create_synthetic_segmentation_dataset(self, test_size=0.2):
        """Create synthetic segmentation dataset"""
        print("Creating synthetic segmentation dataset...")
        
        num_samples = 200
        X = np.random.rand(num_samples, *self.image_size, 3) * 255
        X = X.astype(np.float32)
        
        # Create synthetic masks with circular "tumors"
        y = np.zeros((num_samples, *self.image_size, 1))
        
        for i in range(num_samples):
            if np.random.rand() > 0.3:  # 70% chance of having a tumor
                # Random circle parameters
                center_x = np.random.randint(50, self.image_size[0] - 50)
                center_y = np.random.randint(50, self.image_size[1] - 50)
                radius = np.random.randint(10, 30)
                
                # Create circular mask
                Y, X_coord = np.ogrid[:self.image_size[0], :self.image_size[1]]
                mask = (X_coord - center_x)**2 + (Y - center_y)**2 <= radius**2
                y[i, :, :, 0] = mask.astype(np.uint8)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Synthetic segmentation dataset created: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_class_names(self):
        """Get class names"""
        if hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.classes_.tolist()
        else:
            return ['no_tumor', 'glioma', 'meningioma', 'pituitary']