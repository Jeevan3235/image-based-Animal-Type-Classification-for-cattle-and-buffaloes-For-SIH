#!/usr/bin/env python3
"""
Retrain Model with Combined Datasets
This script combines all three datasets and retrains the model to improve breed classification
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
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CombinedDatasetTrainer:
    def __init__(self):
        self.project_root = Path(os.path.dirname(os.path.abspath(__file__)))
        self.models_path = self.project_root / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Dataset paths
        self.buffalo_breed_set = self.project_root / "Buffalo Breed Set"
        self.processed_buffalo_dataset = self.project_root / "processed_buffalo_dataset"
        self.cow_buffalo_model = self.project_root / "cow vs buffalo computer vision  model"
        self.tensorflow_dataset = self.project_root / "cow-and-buffalo.v1i.tensorflow"
        self.cattle_breed_dataset = self.project_root / "cattle breed" / "Cattle Breeds"
        
        # Output paths
        self.combined_dataset_path = self.project_root / "combined_dataset"
        self.combined_dataset_path.mkdir(exist_ok=True)
        
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
        """Extract features from image for model training"""
        try:
            # Resize image
            image = cv2.resize(image, (224, 224))
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract HOG features
            features = self.extract_hog_features(gray)
            
            # Extract color features
            color_features = self.extract_color_features(image)
            
            # Extract texture features
            texture_features = self.extract_texture_features(gray)
            
            # Combine all features
            all_features = np.concatenate([features, color_features, texture_features])
            
            return all_features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def extract_hog_features(self, gray_image):
        """Extract HOG features"""
        # Simple gradient magnitude as HOG approximation
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Bin the angles into 9 bins
        bins = np.linspace(0, 2*np.pi, 9+1)
        bin_idx = np.digitize(ang, bins) - 1
        bin_idx[bin_idx == 9] = 0  # Wrap around for last bin
        
        # Create histogram for each cell
        cell_size = 8
        n_cells_x = gray_image.shape[1] // cell_size
        n_cells_y = gray_image.shape[0] // cell_size
        hog_features = np.zeros((n_cells_y, n_cells_x, 9))
        
        for y in range(n_cells_y):
            for x in range(n_cells_x):
                for bin_id in range(9):
                    mask = (bin_idx[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size] == bin_id)
                    hog_features[y, x, bin_id] = np.sum(mag[y*cell_size:(y+1)*cell_size, x*cell_size:(x+1)*cell_size][mask])
        
        # Flatten HOG features
        hog_features = hog_features.flatten()
        
        return hog_features
    
    def extract_color_features(self, image):
        """Extract color features"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        # Combine histograms
        color_features = np.concatenate([h_hist, s_hist, v_hist])
        
        return color_features
    
    def extract_texture_features(self, gray_image):
        """Extract texture features using LBP"""
        # Simple LBP implementation
        lbp = np.zeros_like(gray_image)
        neighbors = 8
        radius = 1
        
        for i in range(radius, gray_image.shape[0] - radius):
            for j in range(radius, gray_image.shape[1] - radius):
                center = gray_image[i, j]
                binary_pattern = 0
                
                # Define neighbors in a circle
                for k in range(neighbors):
                    # Get coordinates of neighbor pixel
                    x = i + int(radius * np.cos(2 * np.pi * k / neighbors))
                    y = j + int(radius * np.sin(2 * np.pi * k / neighbors))
                    
                    # Compare with center pixel
                    if gray_image[x, y] >= center:
                        binary_pattern += 1 << k
                
                lbp[i, j] = binary_pattern
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, neighbors**2 + 1), density=True)
        
        return hist
    
    def process_buffalo_breed_set(self):
        """Process Buffalo Breed Set dataset"""
        logger.info("Processing Buffalo Breed Set dataset...")
        
        if not self.buffalo_breed_set.exists():
            logger.error(f"Buffalo Breed Set directory not found: {self.buffalo_breed_set}")
            return []
        
        features_list = []
        labels = []
        breed_labels = []
        
        # Process each breed directory
        for breed_dir in self.buffalo_breed_set.iterdir():
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                logger.info(f"Processing breed: {breed_name}")
                
                # Process each image in the breed directory
                for img_path in breed_dir.glob("*.jpg"):
                    try:
                        # Read image
                        image = cv2.imread(str(img_path))
                        if image is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue
                        
                        # Extract features
                        features = self.extract_features(image)
                        if features is not None:
                            features_list.append(features)
                            labels.append("buffalo")
                            breed_labels.append(breed_name)
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {e}")
        
        logger.info(f"Processed {len(features_list)} images from Buffalo Breed Set")
        return features_list, labels, breed_labels
    
    def process_cow_buffalo_model_dataset(self):
        """Process cow vs buffalo computer vision model dataset"""
        logger.info("Processing cow vs buffalo computer vision model dataset...")
        
        train_dir = self.cow_buffalo_model / "train"
        if not train_dir.exists():
            logger.error(f"Train directory not found: {train_dir}")
            return [], [], []
        
        features_list = []
        labels = []
        breed_labels = []
        
        # Process each class directory
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                logger.info(f"Processing class: {class_name}")
                
                # Process each image in the class directory
                for img_path in class_dir.glob("*.jpg"):
                    try:
                        # Read image
                        image = cv2.imread(str(img_path))
                        if image is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue
                        
                        # Extract features
                        features = self.extract_features(image)
                        if features is not None:
                            features_list.append(features)
                            labels.append(class_name)
                            # For this dataset, we don't have specific breed information
                            breed_labels.append("Unknown")
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {e}")
        
        logger.info(f"Processed {len(features_list)} images from cow vs buffalo model dataset")
        return features_list, labels, breed_labels
    
    def process_tensorflow_dataset(self):
        """Process cow-and-buffalo.v1i.tensorflow dataset"""
        logger.info("Processing cow-and-buffalo.v1i.tensorflow dataset...")
        
        features_list = []
        labels = []
        breed_labels = []
        
        # Process train directory
        train_dir = self.tensorflow_dataset / "train"
        if train_dir.exists():
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name.lower()
                    logger.info(f"Processing class: {class_name}")
                    
                    # Process each image in the class directory
                    for img_path in class_dir.glob("*.jpg"):
                        try:
                            # Read image
                            image = cv2.imread(str(img_path))
                            if image is None:
                                logger.warning(f"Could not read image: {img_path}")
                                continue
                            
                            # Extract features
                            features = self.extract_features(image)
                            if features is not None:
                                features_list.append(features)
                                labels.append(class_name)
                                # For this dataset, we don't have specific breed information
                                breed_labels.append("Unknown")
                        except Exception as e:
                            logger.error(f"Error processing image {img_path}: {e}")
        
        # Process valid directory
        valid_dir = self.tensorflow_dataset / "valid"
        if valid_dir.exists():
            for class_dir in valid_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name.lower()
                    logger.info(f"Processing class: {class_name}")
                    
                    # Process each image in the class directory
                    for img_path in class_dir.glob("*.jpg"):
                        try:
                            # Read image
                            image = cv2.imread(str(img_path))
                            if image is None:
                                logger.warning(f"Could not read image: {img_path}")
                                continue
                            
                            # Extract features
                            features = self.extract_features(image)
                            if features is not None:
                                features_list.append(features)
                                labels.append(class_name)
                                # For this dataset, we don't have specific breed information
                                breed_labels.append("Unknown")
                        except Exception as e:
                            logger.error(f"Error processing image {img_path}: {e}")
        
        logger.info(f"Processed {len(features_list)} images from tensorflow dataset")
        return features_list, labels, breed_labels
    
    def process_processed_buffalo_dataset(self):
        """Process the processed_buffalo_dataset"""
        logger.info("Processing processed_buffalo_dataset...")
        
        if not self.processed_buffalo_dataset.exists():
            logger.error(f"Processed buffalo dataset directory not found: {self.processed_buffalo_dataset}")
            return [], [], []
        
        features_list = []
        labels = []
        breed_labels = []
        
        # Process each image in the dataset
        for img_path in self.processed_buffalo_dataset.glob("*.jpg"):
            try:
                # Extract breed from filename (format: buffalo_BreedName_XXXX.jpg)
                parts = img_path.stem.split('_')
                if len(parts) >= 2:
                    breed_name = parts[1]
                    
                    # Read image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue
                    
                    # Extract features
                    features = self.extract_features(image)
                    if features is not None:
                        features_list.append(features)
                        labels.append("buffalo")
                        breed_labels.append(breed_name)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
        
        logger.info(f"Processed {len(features_list)} images from processed_buffalo_dataset")
        return features_list, labels, breed_labels
    
    def process_cattle_breed_dataset(self):
        """Process the cattle breed dataset"""
        logger.info("Processing cattle breed dataset...")
        
        if not self.cattle_breed_dataset.exists():
            logger.error(f"Cattle breed dataset directory not found: {self.cattle_breed_dataset}")
            return [], [], []
        
        features_list = []
        labels = []
        breed_labels = []
        
        # Process each breed directory
        for breed_dir in self.cattle_breed_dataset.iterdir():
            if breed_dir.is_dir():
                breed_name = breed_dir.name
                logger.info(f"Processing cattle breed: {breed_name}")
                
                # Process each image in the breed directory
                for img_path in breed_dir.glob("*.jpg"):
                    try:
                        # Read image
                        image = cv2.imread(str(img_path))
                        if image is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue
                        
                        # Extract features
                        features = self.extract_features(image)
                        if features is not None:
                            features_list.append(features)
                            labels.append("cattle")
                            breed_labels.append(breed_name)
                    except Exception as e:
                        logger.error(f"Error processing image {img_path}: {e}")
        
        logger.info(f"Processed {len(features_list)} images from cattle breed dataset")
        return features_list, labels, breed_labels
        
    def combine_datasets(self):
        """Combine all datasets"""
        logger.info("Combining all datasets...")
        
        # Process each dataset
        buffalo_features, buffalo_labels, buffalo_breeds = self.process_buffalo_breed_set()
        processed_buffalo_features, processed_buffalo_labels, processed_buffalo_breeds = self.process_processed_buffalo_dataset()
        cow_buffalo_features, cow_buffalo_labels, cow_buffalo_breeds = self.process_cow_buffalo_model_dataset()
        tensorflow_features, tensorflow_labels, tensorflow_breeds = self.process_tensorflow_dataset()
        cattle_features, cattle_labels, cattle_breeds = self.process_cattle_breed_dataset()
        
        # Combine all features and labels
        all_features = buffalo_features + processed_buffalo_features + cow_buffalo_features + tensorflow_features + cattle_features
        all_labels = buffalo_labels + processed_buffalo_labels + cow_buffalo_labels + tensorflow_labels + cattle_labels
        all_breeds = buffalo_breeds + processed_buffalo_breeds + cow_buffalo_breeds + tensorflow_breeds + cattle_breeds
        
        if len(all_features) == 0:
            logger.error("No features extracted from any dataset")
            return None, None, None
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"Combined dataset: {X.shape[0]} samples")
        logger.info(f"Classes: {np.unique(y)}")
        logger.info(f"Unique breeds: {len(set(all_breeds))}")
        
        return X, y, all_breeds
    
    def train_model(self):
        """Train model with combined dataset"""
        logger.info("Training model with combined dataset...")
        
        # Combine datasets
        X, y, breed_labels = self.combine_datasets()
        
        if X is None or len(X) == 0:
            logger.error("No training data available")
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test, breed_train, breed_test = train_test_split(
            X, y, breed_labels, test_size=0.2, random_state=42, stratify=y
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
        model_path = self.project_root / "cattle_buffalo_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Update metadata
        unique_breeds = list(set(breed_labels))
        cattle_breeds = [breed for breed, animal_type in zip(breed_labels, y) if animal_type == 'cattle']
        buffalo_breeds = [breed for breed, animal_type in zip(breed_labels, y) if animal_type == 'buffalo']
        
        # Filter out 'Unknown' breeds
        cattle_breeds = [breed for breed in cattle_breeds if breed != 'Unknown']
        buffalo_breeds = [breed for breed in buffalo_breeds if breed != 'Unknown']
        
        # Create class mapping
        class_mapping = {}
        for i, label in enumerate(np.unique(y)):
            class_mapping[str(i)] = label
        
        metadata = {
            "class_mapping": class_mapping,
            "model_type": "RandomForestClassifier",
            "total_images": len(X),
            "training_images": len(X_train),
            "test_images": len(X_test),
            "accuracy": test_score,
            "training_accuracy": train_score,
            "breeds": [breed for breed in unique_breeds if breed != 'Unknown'],
            "cattle_breeds": list(set(cattle_breeds)),
            "buffalo_breeds": list(set(buffalo_breeds)),
            "last_trained": datetime.now().isoformat(),
            "feature_names": [f"feature_{i}" for i in range(len(X[0]))],
            "combined_datasets_trained": True
        }
        
        metadata_path = self.project_root / "cattle_buffalo_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
        
        # Test the model
        logger.info("\nTesting model with sample images...")
        self.test_sample_images(model, X_test[:5], y_test[:5], breed_test[:5])
        
        return model, metadata
    
    def test_sample_images(self, model, X_test, y_test, breed_labels):
        """Test model with sample images"""
        for i, (features, true_label, breed) in enumerate(zip(X_test, y_test, breed_labels)):
            prediction = model.predict([features])[0]
            confidence = model.predict_proba([features])[0].max()
            
            logger.info(f"Sample {i+1}: {breed} ({true_label}) -> {prediction} (confidence: {confidence:.3f})")

def main():
    """Main function"""
    print("🔄 Combined Dataset Trainer")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = CombinedDatasetTrainer()
        
        # Train model
        print("🔄 Training model with combined datasets...")
        model, metadata = trainer.train_model()
        
        if model is not None:
            print(f"\n🎉 Model training completed successfully!")
            print(f"📊 Test accuracy: {metadata['accuracy']:.3f}")
            print(f"📈 Training accuracy: {metadata['training_accuracy']:.3f}")
            print(f"📊 Total images: {metadata['total_images']}")
            print(f"🐄 Cattle breeds: {len(metadata['cattle_breeds'])}")
            print(f"🐃 Buffalo breeds: {len(metadata['buffalo_breeds'])}")
            print(f"💾 Model saved: cattle_buffalo_model.joblib")
            print(f"📋 Metadata saved: cattle_buffalo_model_metadata.json")
        else:
            print("❌ Model training failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
