#!/usr/bin/env python3
"""
Flask Backend for Cattle vs Buffalo Classification
Serves the trained ML model via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import cv2
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from breed_classifier import breed_classifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables for model
model = None
metadata = None

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    
    try:
        # Load model
        model_path = "cattle_buffalo_model.joblib"
        if Path(model_path).exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Load metadata
        metadata_path = "cattle_buffalo_model_metadata.json"
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info("Metadata loaded successfully")
        else:
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def extract_features_from_image(image_array, bbox=None):
    """Extract features from image array (same as training)"""
    try:
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image = image_array
        else:
            image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # Crop to bounding box if provided
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            image = image[ymin:ymax, xmin:xmax]
        
        # Resize image to standard size
        image = cv2.resize(image, (224, 224))
        
        # Extract features (same as training script)
        features = []
        
        # 1. Color histogram features
        for i in range(3):  # RGB channels
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. HSV color features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i in range(3):  # HSV channels
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 3. Texture features (LBP-like)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern approximation
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
            channel_data = image[:, :, channel].flatten()
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

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classify an image as cattle or buffalo"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Extract features
        bbox = data.get('bbox')
        features = extract_features_from_image(image, bbox)
        
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        # Make prediction
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class names
        class_names = model.classes_
        probabilities_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        # Advanced breed classification
        breed_result = breed_classifier.classify_breed(image, prediction)
        breed = breed_result['breed']
        breed_confidence = breed_result['confidence']
        
        result = {
            'prediction': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': probabilities_dict,
            'breed': breed,
            'breedConfidence': breed_confidence,
            'model_type': metadata.get('model_type', 'Unknown'),
            'class_mapping': metadata.get('class_mapping', {})
        }
        
        logger.info(f"Classification result: {prediction} (confidence: {result['confidence']:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'metadata_loaded': metadata is not None
    })

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if metadata is None:
        return jsonify({'error': 'Metadata not loaded'}), 500
    
    return jsonify(metadata)

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model. Exiting.")
