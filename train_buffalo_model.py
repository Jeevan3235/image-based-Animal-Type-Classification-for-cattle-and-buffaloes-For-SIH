#!/usr/bin/env python3
"""
Train Model with Buffalo Breed Set
Trains a new model specifically with the processed Buffalo Breed Set dataset
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

def extract_features(image):
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
        lbp = compute_lbp(gray)
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
            skewness = compute_skewness(channel_data)
            features.extend([mean, std, skewness])
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def compute_lbp(image, radius=1, n_points=8):
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

def compute_skewness(data):
    """Compute skewness of data"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def load_training_data():
    """Load training data from processed buffalo dataset"""
    project_root = Path(".")
    processed_dataset_path = project_root / "processed_buffalo_dataset"
    
    X, y, breed_labels = [], [], []
    
    # Load from processed buffalo dataset
    metadata_path = processed_dataset_path / "buffalo_breed_set_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loading {metadata['total_images']} buffalo images...")
        
        for img_data in metadata['images']:
            try:
                # Load image
                img = cv2.imread(img_data['path'])
                if img is not None:
                    # Extract features
                    features = extract_features(img)
                    if features is not None:
                        X.append(features)
                        # Use actual breed label for multi-class training
                        y.append(img_data['breed'])
                        breed_labels.append(img_data['breed'])
            except Exception as e:
                logger.warning(f"Failed to process {img_data['filename']}: {e}")
    
    # Add some cattle data for binary classification
    # For now, we'll create a simple binary classifier
    # In a real scenario, you'd load existing cattle data
    
    logger.info(f"Total training samples: {len(X)}")
    logger.info(f"Classes: {np.unique(y)}")
    logger.info(f"Unique breeds: {len(set(breed_labels))}")
    
    return np.array(X), np.array(y), breed_labels

def train_buffalo_model():
    """Train model with buffalo dataset"""
    print("🐃 Training Model with Buffalo Breed Set")
    print("=" * 50)
    
    # Load training data
    print("🔄 Loading training data...")
    X, y, breed_labels = load_training_data()
    
    if len(X) == 0:
        print("❌ No training data available")
        return None, None
    
    print(f"📊 Training data shape: {X.shape}")
    print(f"🐃 Classes: {np.unique(y)}")
    print(f"🏷️  Unique breeds: {len(set(breed_labels))}")
    
    # For this demo, we'll create a breed classification model
    # In practice, you'd want both cattle and buffalo data for binary classification
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("🔄 Training Random Forest model...")
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
    
    print(f"📈 Training accuracy: {train_score:.3f}")
    print(f"📊 Test accuracy: {test_score:.3f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    
    model_path = models_path / "buffalo_breed_model.joblib"
    joblib.dump(model, model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Create metadata
    unique_breeds = list(set(breed_labels))
    metadata = {
        "model_type": "RandomForestClassifier",
        "total_images": len(X),
        "training_images": len(X_train),
        "test_images": len(X_test),
        "accuracy": test_score,
        "training_accuracy": train_score,
        "breeds": unique_breeds,
        "animal_type": "buffalo",
        "last_trained": datetime.now().isoformat(),
        "feature_names": [f"feature_{i}" for i in range(len(X[0]))],
        "buffalo_breed_set_trained": True
    }
    
    metadata_path = models_path / "buffalo_breed_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved: {metadata_path}")
    
    # Test the model
    print("\n🧪 Testing model with sample images...")
    test_sample_images(model, X_test[:5], y_test[:5], breed_labels[:5])
    
    return model, metadata

def test_sample_images(model, X_test, y_test, breed_labels):
    """Test model with sample images"""
    for i, (features, true_label, breed) in enumerate(zip(X_test, y_test, breed_labels)):
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0].max()
        
        print(f"   Sample {i+1}: {breed} -> {prediction} (confidence: {confidence:.3f})")

def main():
    """Main function"""
    try:
        model, metadata = train_buffalo_model()
        
        if model is not None:
            print(f"\n🎉 Model training completed successfully!")
            print(f"📊 Test accuracy: {metadata['accuracy']:.3f}")
            print(f"📈 Training accuracy: {metadata['training_accuracy']:.3f}")
            print(f"📊 Total images: {metadata['total_images']}")
            print(f"🐃 Buffalo breeds: {len(metadata['breeds'])}")
            print(f"💾 Model saved: models/buffalo_breed_model.joblib")
            print(f"📋 Metadata saved: models/buffalo_breed_model_metadata.json")
        else:
            print("❌ Model training failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

