#!/usr/bin/env python3
"""
New Dataset Creation and Integration Script
Processes a new image folder and integrates it with the existing cattle vs buffalo classification system
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCreator:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.new_dataset_path = self.project_root / "new_dataset"
        self.processed_dataset_path = self.project_root / "processed_new_dataset"
        self.merged_dataset_path = self.project_root / "final_enhanced_dataset"
        
        # Create directories
        self.new_dataset_path.mkdir(exist_ok=True)
        self.processed_dataset_path.mkdir(exist_ok=True)
        self.merged_dataset_path.mkdir(exist_ok=True)
        
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
    
    def process_new_dataset(self, source_folder, dataset_name, animal_type, breed_name=None):
        """
        Process a new dataset folder and prepare it for integration
        
        Args:
            source_folder (str): Path to the folder containing images
            dataset_name (str): Name for the dataset
            animal_type (str): "cattle" or "buffalo"
            breed_name (str): Specific breed name (optional)
        """
        logger.info(f"Processing new dataset: {dataset_name}")
        logger.info(f"Source folder: {source_folder}")
        logger.info(f"Animal type: {animal_type}")
        logger.info(f"Breed: {breed_name or 'Mixed'}")
        
        source_path = Path(source_folder)
        if not source_path.exists():
            raise ValueError(f"Source folder does not exist: {source_folder}")
        
        # Create dataset structure
        dataset_dir = self.new_dataset_path / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Process images
        processed_images = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for img_file in source_path.rglob('*'):
            if img_file.suffix.lower() in supported_formats:
                try:
                    # Load and validate image
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        # Resize to standard size
                        img_resized = cv2.resize(img, (224, 224))
                        
                        # Create new filename
                        new_filename = f"{animal_type}_{breed_name or 'mixed'}_{len(processed_images):04d}.jpg"
                        new_path = dataset_dir / new_filename
                        
                        # Save processed image
                        cv2.imwrite(str(new_path), img_resized)
                        
                        # Extract features
                        features = self.extract_features(img_resized)
                        
                        processed_images.append({
                            'filename': new_filename,
                            'path': str(new_path),
                            'animal_type': animal_type,
                            'breed': breed_name or 'Unknown',
                            'features': features,
                            'original_path': str(img_file)
                        })
                        
                        logger.info(f"Processed: {img_file.name} -> {new_filename}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process {img_file}: {e}")
        
        logger.info(f"Successfully processed {len(processed_images)} images")
        
        # Save dataset metadata
        metadata = {
            'dataset_name': dataset_name,
            'animal_type': animal_type,
            'breed': breed_name,
            'total_images': len(processed_images),
            'processed_date': datetime.now().isoformat(),
            'source_folder': str(source_folder),
            'images': processed_images
        }
        
        metadata_path = dataset_dir / f"{dataset_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def extract_features(self, image):
        """Extract features from image (same as training script)"""
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
    
    def integrate_with_existing_dataset(self, new_dataset_metadata):
        """Integrate new dataset with existing complete dataset"""
        logger.info("Integrating new dataset with existing dataset...")
        
        # Load existing complete dataset
        existing_dataset_path = self.project_root / "complete_merged_dataset"
        if not existing_dataset_path.exists():
            logger.warning("Existing complete dataset not found, creating new one")
            existing_dataset_path.mkdir(exist_ok=True)
        
        # Create enhanced dataset directory
        enhanced_dataset_path = self.merged_dataset_path
        enhanced_dataset_path.mkdir(exist_ok=True)
        
        # Copy existing dataset
        if existing_dataset_path.exists():
            for item in existing_dataset_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, enhanced_dataset_path)
                elif item.is_dir():
                    shutil.copytree(item, enhanced_dataset_path / item.name, dirs_exist_ok=True)
        
        # Add new dataset
        new_dataset_dir = self.new_dataset_path / new_dataset_metadata['dataset_name']
        if new_dataset_dir.exists():
            # Copy new images
            for img_file in new_dataset_dir.glob('*.jpg'):
                shutil.copy2(img_file, enhanced_dataset_path)
        
        # Update metadata
        self.update_enhanced_metadata(new_dataset_metadata)
        
        logger.info(f"Enhanced dataset created at: {enhanced_dataset_path}")
        return enhanced_dataset_path
    
    def update_enhanced_metadata(self, new_dataset_metadata):
        """Update metadata with new dataset information"""
        enhanced_metadata = self.existing_metadata.copy()
        
        # Add new breed if specified
        if new_dataset_metadata['breed'] and new_dataset_metadata['breed'] != 'Unknown':
            breed_name = new_dataset_metadata['breed']
            animal_type = new_dataset_metadata['animal_type']
            
            if animal_type == 'cattle' and breed_name not in enhanced_metadata['cattle_breeds']:
                enhanced_metadata['cattle_breeds'].append(breed_name)
            elif animal_type == 'buffalo' and breed_name not in enhanced_metadata['buffalo_breeds']:
                enhanced_metadata['buffalo_breeds'].append(breed_name)
        
        # Update total counts
        enhanced_metadata['total_images'] = enhanced_metadata.get('total_images', 0) + new_dataset_metadata['total_images']
        enhanced_metadata['last_updated'] = datetime.now().isoformat()
        enhanced_metadata['new_dataset'] = new_dataset_metadata['dataset_name']
        
        # Save enhanced metadata
        metadata_path = self.merged_dataset_path / "enhanced_dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        logger.info("Enhanced metadata updated")
    
    def retrain_model_with_new_dataset(self):
        """Retrain the model with the enhanced dataset"""
        logger.info("Retraining model with enhanced dataset...")
        
        # Load existing model
        model_path = self.project_root / "cattle_buffalo_model.joblib"
        if model_path.exists():
            existing_model = joblib.load(model_path)
            logger.info("Loaded existing model")
        else:
            existing_model = None
            logger.info("No existing model found, creating new one")
        
        # Prepare training data from enhanced dataset
        X, y, breed_labels = self.prepare_training_data()
        
        if len(X) == 0:
            logger.error("No training data available")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")
        
        # Save model
        model_path = self.project_root / "enhanced_cattle_buffalo_model.joblib"
        joblib.dump(model, model_path)
        
        # Update metadata
        enhanced_metadata = {
            "class_mapping": self.existing_metadata.get('class_mapping', {}),
            "model_type": "RandomForestClassifier",
            "total_images": len(X),
            "training_images": len(X_train),
            "test_images": len(X_test),
            "accuracy": test_score,
            "breeds": list(set(breed_labels)),
            "cattle_breeds": self.existing_metadata.get('cattle_breeds', []),
            "buffalo_breeds": self.existing_metadata.get('buffalo_breeds', []),
            "last_trained": datetime.now().isoformat(),
            "feature_names": [f"feature_{i}" for i in range(len(X[0]))]
        }
        
        metadata_path = self.project_root / "enhanced_cattle_buffalo_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        logger.info(f"Enhanced model saved: {model_path}")
        logger.info(f"Enhanced metadata saved: {metadata_path}")
        
        return model, enhanced_metadata
    
    def prepare_training_data(self):
        """Prepare training data from enhanced dataset"""
        X, y, breed_labels = [], [], []
        
        # Load all processed datasets
        for dataset_dir in self.new_dataset_path.iterdir():
            if dataset_dir.is_dir():
                metadata_path = dataset_dir / f"{dataset_dir.name}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    for img_data in metadata['images']:
                        if img_data['features'] is not None:
                            X.append(img_data['features'])
                            y.append(img_data['animal_type'])
                            breed_labels.append(img_data['breed'])
        
        # Also load from existing complete dataset if available
        existing_dataset_path = self.project_root / "complete_merged_dataset"
        if existing_dataset_path.exists():
            # This would need to be implemented based on your existing dataset structure
            pass
        
        return np.array(X), np.array(y), breed_labels

def main():
    """Main function to create and integrate new dataset"""
    print("🚀 New Dataset Creation and Integration System")
    print("=" * 50)
    
    # Initialize dataset creator
    creator = DatasetCreator()
    
    # Get user input
    print("\nPlease provide the following information:")
    source_folder = input("📁 Path to your image folder: ").strip()
    dataset_name = input("📝 Dataset name (e.g., 'my_cattle_dataset'): ").strip()
    
    print("\nAnimal type:")
    print("1. Cattle")
    print("2. Buffalo")
    animal_choice = input("Choose (1 or 2): ").strip()
    animal_type = "cattle" if animal_choice == "1" else "buffalo"
    
    breed_name = input("🐄 Specific breed name (or press Enter for mixed): ").strip()
    if not breed_name:
        breed_name = None
    
    try:
        # Process new dataset
        print(f"\n🔄 Processing dataset: {dataset_name}")
        metadata = creator.process_new_dataset(source_folder, dataset_name, animal_type, breed_name)
        
        print(f"✅ Successfully processed {metadata['total_images']} images")
        
        # Integrate with existing dataset
        print("\n🔄 Integrating with existing dataset...")
        enhanced_dataset_path = creator.integrate_with_existing_dataset(metadata)
        
        print(f"✅ Enhanced dataset created at: {enhanced_dataset_path}")
        
        # Retrain model
        print("\n🔄 Retraining model with enhanced dataset...")
        model, enhanced_metadata = creator.retrain_model_with_new_dataset()
        
        if model is not None:
            print(f"✅ Model retrained successfully!")
            print(f"📊 Test accuracy: {enhanced_metadata['accuracy']:.3f}")
            print(f"📈 Total images: {enhanced_metadata['total_images']}")
            print(f"🐄 Supported breeds: {len(enhanced_metadata['breeds'])}")
            
            print("\n🎉 Dataset integration complete!")
            print(f"📁 Enhanced dataset: {enhanced_dataset_path}")
            print(f"🤖 Enhanced model: enhanced_cattle_buffalo_model.joblib")
            print(f"📋 Enhanced metadata: enhanced_cattle_buffalo_model_metadata.json")
        else:
            print("❌ Model retraining failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
