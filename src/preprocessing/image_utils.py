"""
Image preprocessing utilities for MRI brain scans
"""
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Optional medical image libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. NIfTI support disabled.")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    print("Warning: SimpleITK not available. DICOM support disabled.")

class ImagePreprocessor:
    """
    Image preprocessing utilities for MRI brain images
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def load_image(self, image_path):
        """
        Load image from various formats (DICOM, NIfTI, standard images)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image array
        """
        try:
            # Handle NIfTI files
            if image_path.endswith(('.nii', '.nii.gz')):
                return self.load_nifti(image_path)
            
            # Handle DICOM files
            elif image_path.endswith('.dcm'):
                return self.load_dicom(image_path)
            
            # Handle standard image formats
            else:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.array(image)
                
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def load_nifti(self, nifti_path):
        """Load NIfTI format medical images"""
        if not NIBABEL_AVAILABLE:
            raise ValueError("nibabel not available. Please install: pip install nibabel")
            
        try:
            # Load using nibabel
            nii_img = nib.load(nifti_path)
            image_data = nii_img.get_fdata()
            
            # If 3D, take middle slice
            if len(image_data.shape) == 3:
                middle_slice = image_data.shape[2] // 2
                image_data = image_data[:, :, middle_slice]
            
            # Normalize to 0-255 range
            image_data = self.normalize_intensity(image_data)
            
            # Convert to RGB
            image_rgb = np.stack([image_data] * 3, axis=-1)
            
            return image_rgb.astype(np.uint8)
            
        except Exception as e:
            raise ValueError(f"Error loading NIfTI image: {str(e)}")
    
    def load_dicom(self, dicom_path):
        """Load DICOM format medical images"""
        if not SITK_AVAILABLE:
            raise ValueError("SimpleITK not available. Please install: pip install SimpleITK")
            
        try:
            # Load using SimpleITK
            reader = sitk.ImageFileReader()
            reader.SetFileName(dicom_path)
            image = reader.Execute()
            
            # Convert to numpy array
            image_array = sitk.GetArrayFromImage(image)
            
            # If 3D, take middle slice
            if len(image_array.shape) == 3:
                middle_slice = image_array.shape[0] // 2
                image_array = image_array[middle_slice, :, :]
            
            # Normalize to 0-255 range
            image_array = self.normalize_intensity(image_array)
            
            # Convert to RGB
            image_rgb = np.stack([image_array] * 3, axis=-1)
            
            return image_rgb.astype(np.uint8)
            
        except Exception as e:
            raise ValueError(f"Error loading DICOM image: {str(e)}")
    
    def normalize_intensity(self, image):
        """
        Normalize image intensity to 0-255 range
        
        Args:
            image: Input image array
            
        Returns:
            numpy.ndarray: Normalized image
        """
        # Remove outliers (clip to 1st and 99th percentiles)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Normalize to 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255)
        
        return image
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: Input image array or path
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize image
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        # Normalize pixel values
        image = image.astype(np.float32)
        
        return image
    
    def enhance_contrast(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image array
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def denoise_image(self, image):
        """
        Apply denoising to the image
        
        Args:
            image: Input image array
            
        Returns:
            numpy.ndarray: Denoised image
        """
        if len(image.shape) == 3:
            # Color image denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Grayscale image denoising
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def skull_strip(self, image):
        """
        Simple skull stripping using thresholding and morphology
        
        Args:
            image: Input brain image
            
        Returns:
            numpy.ndarray: Skull-stripped image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to create brain mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component (brain)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        # Apply mask to original image
        if len(image.shape) == 3:
            mask_3d = np.stack([mask] * 3, axis=-1)
            result = np.where(mask_3d > 0, image, 0)
        else:
            result = np.where(mask > 0, image, 0)
        
        return result
    
    def create_augmentation_generator(self, rotation_range=20, zoom_range=0.1, 
                                   horizontal_flip=True, vertical_flip=False):
        """
        Create data augmentation generator
        
        Args:
            rotation_range: Rotation range in degrees
            zoom_range: Zoom range for random zoom
            horizontal_flip: Whether to apply horizontal flip
            vertical_flip: Whether to apply vertical flip
            
        Returns:
            ImageDataGenerator: Configured augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )