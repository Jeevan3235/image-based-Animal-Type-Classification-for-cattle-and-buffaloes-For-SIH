import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import joblib
import json
import os
from measurement_processor import BodyMeasurementProcessor

class AnimalTypeClassifier:
    def __init__(self, model_path='cattle_buffalo_model.joblib', 
                 metadata_path='cattlebuffalomodelmetadata.json'):
        """
        Initialize the animal type classification system
        """
        self.classifier = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.measurement_processor = BodyMeasurementProcessor()
        self.is_initialized = True
        
    def preprocess_image(self, image_path):
        """
        Preprocess image for classification
        """
        try:
            # Load and resize image
            image = Image.open(image_path)
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            # Ensure 3 channels
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
                
            return img_array.reshape(1, 224, 224, 3)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_features(self, image_path):
        """
        Extract features from image for classification
        """
        try:
            # Load image for feature extraction
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Basic image features
            features = []
            
            # Color features (mean and std of each channel)
            for i in range(3):
                features.append(np.mean(image[:, :, i]))
                features.append(np.std(image[:, :, i]))
            
            # Texture features using GLCM
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            glcm = self.calculate_glcm(gray)
            features.extend(glcm)
            
            # Shape features (basic contour analysis)
            contours, _ = cv2.findContours(
                cv2.Canny(gray, 50, 150), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                features.append(cv2.contourArea(largest_contour))
                features.append(cv2.arcLength(largest_contour, True))
            else:
                features.extend([0, 0])
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_glcm(self, gray_image):
        """
        Calculate Gray Level Co-occurrence Matrix features
        """
        # Simplified GLCM calculation
        glcm = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        glcm = glcm.flatten()
        glcm = glcm / glcm.sum()  # Normalize
        
        # Return basic statistics
        return [
            np.mean(glcm),
            np.std(glcm),
            np.max(glcm),
            np.min(glcm)
        ]
    
    def classify_animal(self, image_path):
        """
        Classify animal as cattle or buffalo and extract measurements
        """
        try:
            # Extract features for classification
            features = self.extract_features(image_path)
            if features is None:
                return {"error": "Could not extract features from image"}
            
            # Predict animal type
            prediction = self.classifier.predict(features)
            probability = self.classifier.predict_proba(features)
            
            animal_type = "cattle" if prediction[0] == 0 else "buffalo"
            confidence = max(probability[0])
            
            # Extract body measurements
            measurements = self.measurement_processor.process_image(image_path)
            
            # Compile results
            result = {
                "animal_type": animal_type,
                "confidence": float(confidence),
                "measurements": measurements,
                "timestamp": np.datetime64('now').astype(str),
                "image_path": image_path
            }
            
            return result
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

    def batch_process(self, image_directory):
        """
        Process multiple images in a directory
        """
        results = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for filename in os.listdir(image_directory):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(image_directory, filename)
                result = self.classify_animal(image_path)
                result['filename'] = filename
                results.append(result)
        
        return results
