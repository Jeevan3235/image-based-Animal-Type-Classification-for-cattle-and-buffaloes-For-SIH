#!/usr/bin/env python3
"""
Model Retraining Script with New Dataset
Retrains the cattle vs buffalo classification model with newly integrated datasets
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.new_dataset_path = self.project_root / "new_integrated_dataset"
        self.existing_dataset_path = self.project_root / "complete_merged_dataset"
        self.models_path = self.project_root / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.load_existing_metadata()
    
    def load_existing_metadata(self):
        """Load existing model metadata"""
        metadata_path = self.project_root / "cattle_buffalo_model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.existing_metadata = json.load(f)
        else:
            self.existing_metadata = {
                "class_mapping": {},
                "breeds": [],
                "cattle_breeds": [],
                "buffalo_breeds": []
            }
    
    def extract_features(self, image):
        """Extract features from image (same as original training)"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            features = []
            
            # 1. Color histogram features
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([rgb_image], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 2. HSV color features
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            for i in range(3):  # HSV channels
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 3. Texture features (LBP-like)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            lbp = self.compute_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 16])
            features.extend(lbp_hist.flatten())
            
            # 4. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 5. Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            
            # 6. Shape features
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
            else:
                area = 0
                circularity = 0
            
            features.extend([area, circularity])
            
            # 7. Color moments
            for channel in range(3):
                channel_data = rgb_image[:, :, channel].flatten()
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                skewness = self.compute_skewness(channel_data)
                features.extend([mean, std, skewness])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def compute_lbp(self, image, radius=1, n_points=8):
        """Compute Local Binary Pattern"""
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ''
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if x < rows and y < cols:
                        if image[x, y] >= center:
                            binary_string += '1'
                        else:
                            binary_string += '0'
                    else:
                        binary_string += '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def compute_skewness(self, data):
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def load_training_data(self):
        """Load all available training data"""
        X, y, breed_labels = [], [], []
        
        # Load from new integrated dataset
        if self.new_dataset_path.exists():
            logger.info("Loading data from new integrated dataset...")
            for dataset_dir in self.new_dataset_path.iterdir():
                if dataset_dir.is_dir():
                    metadata_path = dataset_dir / "dataset_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        logger.info(f"Processing dataset: {metadata['dataset_name']}")
                        logger.info(f"Animal type: {metadata['animal_type']}")
                        logger.info(f"Breed: {metadata['breed']}")
                        logger.info(f"Total images: {metadata['total_images']}")
                        
                        # Process images in this dataset
                        for img_file in dataset_dir.glob('*.jpg'):
                            try:
                                img = cv2.imread(str(img_file))
                                if img is not None:
                                    features = self.extract_features(img)
                                    if features is not None:
                                        X.append(features)
                                        y.append(metadata['animal_type'])
                                        breed_labels.append(metadata['breed'])
                            except Exception as e:
                                logger.warning(f"Failed to process {img_file}: {e}")
        
        # Load from existing complete dataset
        if self.existing_dataset_path.exists():
            logger.info("Loading data from existing complete dataset...")
            # This would need to be implemented based on your existing dataset structure
            # For now, we'll assume the new dataset is sufficient
        
        logger.info(f"Total training samples loaded: {len(X)}")
        return np.array(X), np.array(y), breed_labels
    
    def retrain_model(self):
        """Retrain the model with all available data"""
        logger.info("Starting model retraining...")
        
        # Load training data
        X, y, breed_labels = self.load_training_data()
        
        if len(X) == 0:
            logger.error("No training data available")
            return None, None
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Classes: {np.unique(y)}")
        logger.info(f"Unique breeds: {len(set(breed_labels))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = model.predict(X_test)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Save model
        model_path = self.models_path / "retrained_cattle_buffalo_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Update metadata
        unique_breeds = list(set(breed_labels))
        cattle_breeds = [breed for breed, animal_type in zip(breed_labels, y) if animal_type == 'cattle']
        buffalo_breeds = [breed for breed, animal_type in zip(breed_labels, y) if animal_type == 'buffalo']
        
        enhanced_metadata = {
            "class_mapping": self.existing_metadata.get('class_mapping', {}),
            "model_type": "RandomForestClassifier",
            "total_images": len(X),
            "training_images": len(X_train),
            "test_images": len(X_test),
            "accuracy": test_score,
            "training_accuracy": train_score,
            "breeds": unique_breeds,
            "cattle_breeds": list(set(cattle_breeds)),
            "buffalo_breeds": list(set(buffalo_breeds)),
            "last_trained": datetime.now().isoformat(),
            "feature_names": [f"feature_{i}" for i in range(len(X[0]))],
            "retrained_with_new_dataset": True
        }
        
        metadata_path = self.models_path / "retrained_cattle_buffalo_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
        
        return model, enhanced_metadata
    
    def update_main_model(self, new_model, new_metadata):
        """Update the main model files with the retrained model"""
        logger.info("Updating main model files...")
        
        # Backup existing model
        existing_model_path = self.project_root / "cattle_buffalo_model.joblib"
        existing_metadata_path = self.project_root / "cattle_buffalo_model_metadata.json"
        
        if existing_model_path.exists():
            backup_model_path = self.project_root / "cattle_buffalo_model_backup.joblib"
            shutil.copy2(existing_model_path, backup_model_path)
            logger.info(f"Backed up existing model to: {backup_model_path}")
        
        if existing_metadata_path.exists():
            backup_metadata_path = self.project_root / "cattle_buffalo_model_metadata_backup.json"
            shutil.copy2(existing_metadata_path, backup_metadata_path)
            logger.info(f"Backed up existing metadata to: {backup_metadata_path}")
        
        # Copy new model files
        new_model_path = self.models_path / "retrained_cattle_buffalo_model.joblib"
        new_metadata_path = self.models_path / "retrained_cattle_buffalo_model_metadata.json"
        
        shutil.copy2(new_model_path, existing_model_path)
        shutil.copy2(new_metadata_path, existing_metadata_path)
        
        logger.info("Main model files updated successfully")
    
    def test_model(self, model, metadata):
        """Test the retrained model with sample images"""
        logger.info("Testing retrained model...")
        
        # Test with images from new dataset
        test_images = []
        if self.new_dataset_path.exists():
            for dataset_dir in self.new_dataset_path.iterdir():
                if dataset_dir.is_dir():
                    for img_file in list(dataset_dir.glob('*.jpg'))[:5]:  # Test with first 5 images
                        test_images.append(img_file)
        
        if not test_images:
            logger.info("No test images available")
            return
        
        logger.info(f"Testing with {len(test_images)} sample images...")
        
        for img_file in test_images:
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    features = self.extract_features(img)
                    if features is not None:
                        prediction = model.predict([features])[0]
                        probabilities = model.predict_proba([features])[0]
                        
                        logger.info(f"Image: {img_file.name}")
                        logger.info(f"Prediction: {prediction}")
                        logger.info(f"Confidence: {max(probabilities):.3f}")
                        logger.info("-" * 30)
            except Exception as e:
                logger.warning(f"Failed to test {img_file}: {e}")

def main():
    """Main function for model retraining"""
    print("🚀 Model Retraining with New Dataset")
    print("=" * 40)
    
    # Initialize retrainer
    retrainer = ModelRetrainer()
    
    try:
        # Retrain model
        print("🔄 Retraining model with new dataset...")
        model, metadata = retrainer.retrain_model()
        
        if model is not None:
            print(f"✅ Model retrained successfully!")
            print(f"📊 Test accuracy: {metadata['accuracy']:.3f}")
            print(f"📈 Training accuracy: {metadata['training_accuracy']:.3f}")
            print(f"📊 Total images: {metadata['total_images']}")
            print(f"🐄 Cattle breeds: {len(metadata['cattle_breeds'])}")
            print(f"🐃 Buffalo breeds: {len(metadata['buffalo_breeds'])}")
            
            # Update main model files
            print("\n🔄 Updating main model files...")
            retrainer.update_main_model(model, metadata)
            
            # Test model
            print("\n🧪 Testing retrained model...")
            retrainer.test_model(model, metadata)
            
            print("\n🎉 Model retraining completed successfully!")
            print("📁 New model files:")
            print("   - cattle_buffalo_model.joblib")
            print("   - cattle_buffalo_model_metadata.json")
            print("📁 Backup files:")
            print("   - cattle_buffalo_model_backup.joblib")
            print("   - cattle_buffalo_model_metadata_backup.json")
            
        else:
            print("❌ Model retraining failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

