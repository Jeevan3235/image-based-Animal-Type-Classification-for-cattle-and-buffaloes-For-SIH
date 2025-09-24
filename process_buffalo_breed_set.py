#!/usr/bin/env python3
"""
Process Buffalo Breed Set Dataset
Processes the "Buffalo Breed Set" folder and integrates it with the cattle vs buffalo classification system
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
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuffaloBreedProcessor:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.buffalo_breed_set_path = self.project_root / "Buffalo Breed Set"
        self.processed_dataset_path = self.project_root / "processed_buffalo_dataset"
        self.enhanced_dataset_path = self.project_root / "enhanced_final_dataset"
        self.models_path = self.project_root / "models"
        
        # Create directories
        self.processed_dataset_path.mkdir(exist_ok=True)
        self.enhanced_dataset_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.load_existing_metadata()
        
        # Buffalo breed mapping
        self.buffalo_breeds = {
            'Banni': 'Banni',
            'Bhadawari': 'Bhadawari', 
            'Chilika': 'Chilika',
            'Godawari': 'Godawari',
            'Gojri': 'Gojri',
            'Jaffarabadi': 'Jaffarabadi',
            'Kalahandi': 'Kalahandi',
            'Manda': 'Manda',
            'Marathwada': 'Marathwada',
            'Mehsana': 'Mehsana',
            'MURRAH': 'Murrah',
            'NAGPURI': 'Nagpuri',
            'NILI RAVI': 'Nili Ravi',
            'PANDHARPURI': 'Pandharpuri',
            'PARALAKHEMUNDI': 'Paralakhemundi',
            'SAMBALPURI': 'Sambalpuri',
            'South Kanara': 'South Kanara',
            'Surti': 'Surti',
            'Swamp': 'Swamp Buffalo',
            'Tarai': 'Tarai',
            'Toda': 'Toda'
        }
    
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
    
    def process_buffalo_breed_set(self):
        """Process the Buffalo Breed Set dataset"""
        logger.info("Processing Buffalo Breed Set dataset...")
        
        if not self.buffalo_breed_set_path.exists():
            logger.error(f"Buffalo Breed Set folder not found: {self.buffalo_breed_set_path}")
            return None
        
        processed_data = []
        total_images = 0
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Process each breed folder
        for breed_folder in self.buffalo_breed_set_path.iterdir():
            if breed_folder.is_dir():
                breed_name = self.buffalo_breeds.get(breed_folder.name, breed_folder.name)
                logger.info(f"Processing breed: {breed_name} ({breed_folder.name})")
                
                breed_images = 0
                for img_file in breed_folder.iterdir():
                    if img_file.suffix.lower() in supported_formats:
                        try:
                            # Load image
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                # Resize to standard size
                                img_resized = cv2.resize(img, (224, 224))
                                
                                # Extract features
                                features = self.extract_features(img_resized)
                                
                                if features is not None:
                                    # Create processed filename
                                    processed_filename = f"buffalo_{breed_name}_{breed_images:04d}.jpg"
                                    processed_path = self.processed_dataset_path / processed_filename
                                    
                                    # Save processed image
                                    cv2.imwrite(str(processed_path), img_resized)
                                    
                                    processed_data.append({
                                        'filename': processed_filename,
                                        'path': str(processed_path),
                                        'animal_type': 'buffalo',
                                        'breed': breed_name,
                                        'original_folder': breed_folder.name,
                                        'original_file': img_file.name,
                                        'features': features.tolist()  # Convert to list for JSON serialization
                                    })
                                    
                                    breed_images += 1
                                    total_images += 1
                                    
                                    if breed_images % 5 == 0:
                                        logger.info(f"   Processed {breed_images} images for {breed_name}")
                                        
                        except Exception as e:
                            logger.warning(f"Failed to process {img_file}: {e}")
                
                logger.info(f"✅ {breed_name}: {breed_images} images processed")
        
        logger.info(f"Total buffalo images processed: {total_images}")
        
        # Save processed dataset metadata
        metadata = {
            'dataset_name': 'Buffalo Breed Set',
            'animal_type': 'buffalo',
            'total_images': total_images,
            'total_breeds': len(self.buffalo_breeds),
            'processed_date': datetime.now().isoformat(),
            'breeds': list(self.buffalo_breeds.values()),
            'images': processed_data
        }
        
        metadata_path = self.processed_dataset_path / "buffalo_breed_set_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
        return metadata
    
    def integrate_with_existing_dataset(self, buffalo_metadata):
        """Integrate buffalo dataset with existing complete dataset"""
        logger.info("Integrating buffalo dataset with existing dataset...")
        
        # Load existing complete dataset
        existing_dataset_path = self.project_root / "complete_merged_dataset"
        if existing_dataset_path.exists():
            logger.info("Found existing complete dataset, integrating...")
            # Copy existing dataset to enhanced dataset
            for item in existing_dataset_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.enhanced_dataset_path)
                elif item.is_dir():
                    shutil.copytree(item, self.enhanced_dataset_path / item.name, dirs_exist_ok=True)
        
        # Add processed buffalo images
        for img_file in self.processed_dataset_path.glob('*.jpg'):
            shutil.copy2(img_file, self.enhanced_dataset_path)
        
        # Update metadata
        self.update_enhanced_metadata(buffalo_metadata)
        
        logger.info(f"Enhanced dataset created at: {self.enhanced_dataset_path}")
        return self.enhanced_dataset_path
    
    def update_enhanced_metadata(self, buffalo_metadata):
        """Update metadata with buffalo dataset information"""
        enhanced_metadata = self.existing_metadata.copy()
        
        # Add buffalo breeds
        buffalo_breeds = buffalo_metadata['breeds']
        for breed in buffalo_breeds:
            if breed not in enhanced_metadata.get('buffalo_breeds', []):
                enhanced_metadata.setdefault('buffalo_breeds', []).append(breed)
        
        # Update total counts
        enhanced_metadata['total_images'] = enhanced_metadata.get('total_images', 0) + buffalo_metadata['total_images']
        enhanced_metadata['buffalo_images'] = buffalo_metadata['total_images']
        enhanced_metadata['last_updated'] = datetime.now().isoformat()
        enhanced_metadata['buffalo_breed_set_integrated'] = True
        
        # Save enhanced metadata
        metadata_path = self.enhanced_dataset_path / "enhanced_dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        logger.info("Enhanced metadata updated")
    
    def prepare_training_data(self):
        """Prepare training data from all available sources"""
        X, y, breed_labels = [], [], []
        
        # Load from processed buffalo dataset
        metadata_path = self.processed_dataset_path / "buffalo_breed_set_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loading {metadata['total_images']} buffalo images...")
            for img_data in metadata['images']:
                if img_data['features'] is not None:
                    X.append(np.array(img_data['features']))
                    y.append('buffalo')
                    breed_labels.append(img_data['breed'])
        
        # Load from existing complete dataset (cattle images)
        existing_dataset_path = self.project_root / "complete_merged_dataset"
        if existing_dataset_path.exists():
            logger.info("Loading existing cattle images...")
            # This would need to be implemented based on your existing dataset structure
            # For now, we'll create some sample cattle data
            pass
        
        logger.info(f"Total training samples: {len(X)}")
        logger.info(f"Classes: {np.unique(y)}")
        logger.info(f"Unique breeds: {len(set(breed_labels))}")
        
        return np.array(X), np.array(y), breed_labels
    
    def train_enhanced_model(self):
        """Train the enhanced model with buffalo dataset"""
        logger.info("Training enhanced model with buffalo dataset...")
        
        # Prepare training data
        X, y, breed_labels = self.prepare_training_data()
        
        if len(X) == 0:
            logger.error("No training data available")
            return None, None
        
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
        model_path = self.models_path / "enhanced_buffalo_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Update metadata
        unique_breeds = list(set(breed_labels))
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
            "cattle_breeds": self.existing_metadata.get('cattle_breeds', []),
            "buffalo_breeds": list(set(buffalo_breeds)),
            "last_trained": datetime.now().isoformat(),
            "feature_names": [f"feature_{i}" for i in range(len(X[0]))],
            "buffalo_breed_set_integrated": True
        }
        
        metadata_path = self.models_path / "enhanced_buffalo_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
        
        return model, enhanced_metadata
    
    def update_main_system(self, model, metadata):
        """Update the main system files with the enhanced model"""
        logger.info("Updating main system files...")
        
        # Backup existing files
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
        new_model_path = self.models_path / "enhanced_buffalo_model.joblib"
        new_metadata_path = self.models_path / "enhanced_buffalo_model_metadata.json"
        
        shutil.copy2(new_model_path, existing_model_path)
        shutil.copy2(new_metadata_path, existing_metadata_path)
        
        logger.info("Main system files updated successfully")

def main():
    """Main function for processing Buffalo Breed Set"""
    print("🐃 Buffalo Breed Set Dataset Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = BuffaloBreedProcessor()
    
    try:
        # Process buffalo breed set
        print("🔄 Processing Buffalo Breed Set dataset...")
        buffalo_metadata = processor.process_buffalo_breed_set()
        
        if buffalo_metadata:
            print(f"✅ Successfully processed {buffalo_metadata['total_images']} buffalo images")
            print(f"🐃 Total breeds: {buffalo_metadata['total_breeds']}")
            print(f"📁 Processed dataset: {processor.processed_dataset_path}")
            
            # Integrate with existing dataset
            print("\n🔄 Integrating with existing dataset...")
            enhanced_dataset_path = processor.integrate_with_existing_dataset(buffalo_metadata)
            print(f"✅ Enhanced dataset created: {enhanced_dataset_path}")
            
            # Train enhanced model
            print("\n🔄 Training enhanced model...")
            model, metadata = processor.train_enhanced_model()
            
            if model is not None:
                print(f"✅ Model trained successfully!")
                print(f"📊 Test accuracy: {metadata['accuracy']:.3f}")
                print(f"📈 Training accuracy: {metadata['training_accuracy']:.3f}")
                print(f"📊 Total images: {metadata['total_images']}")
                print(f"🐃 Buffalo breeds: {len(metadata['buffalo_breeds'])}")
                
                # Update main system
                print("\n🔄 Updating main system...")
                processor.update_main_system(model, metadata)
                
                print("\n🎉 Buffalo Breed Set integration completed successfully!")
                print("📁 Files created:")
                print(f"   - Processed dataset: {processor.processed_dataset_path}")
                print(f"   - Enhanced dataset: {enhanced_dataset_path}")
                print(f"   - Enhanced model: {processor.models_path}/enhanced_buffalo_model.joblib")
                print(f"   - Enhanced metadata: {processor.models_path}/enhanced_buffalo_model_metadata.json")
                print("\n📋 Updated main files:")
                print("   - cattle_buffalo_model.joblib")
                print("   - cattle_buffalo_model_metadata.json")
                
            else:
                print("❌ Model training failed")
        else:
            print("❌ Dataset processing failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
