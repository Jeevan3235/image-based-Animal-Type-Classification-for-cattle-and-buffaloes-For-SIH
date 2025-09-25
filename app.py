#!/usr/bin/env python3
"""
Unified Flask Backend for Cattle vs Buffalo Classification with Body Measurements
Combines classification, body measurements, and BPA integration
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
import io
import cv2
import numpy as np
import joblib
import json
from pathlib import Path
import logging
import yaml
import sqlite3
import pandas as pd
from datetime import datetime
import os
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
metadata = None

class BodyMeasurementProcessor:
    def __init__(self):
        self.measurement_units = "pixels"
    
    def process_image(self, image_array):
        """Extract body measurements from animal image"""
        try:
            if image_array is None:
                return {"error": "Invalid image"}
            
            # Preprocess image
            processed_image = self.preprocess_image(image_array)
            
            # Detect animal body
            body_contour = self.detect_animal_body(processed_image)
            if body_contour is None:
                return {"error": "Could not detect animal body"}
            
            # Extract measurements
            measurements = self.extract_measurements(body_contour, image_array.shape)
            return measurements
            
        except Exception as e:
            return {"error": f"Measurement extraction failed: {str(e)}"}
    
    def preprocess_image(self, image):
        """Preprocess image for better contour detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned
    
    def detect_animal_body(self, processed_image):
        """Detect the main animal body contour"""
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if area > 1000 and 0.3 < aspect_ratio < 3.0:
                valid_contours.append(contour)
        
        if not valid_contours:
            return None
        
        return max(valid_contours, key=cv2.contourArea)
    
    def extract_measurements(self, contour, image_shape):
        """Extract various body measurements from contour"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # Basic measurements
        body_length = w
        height_at_withers = h
        
        # Calculate convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Estimate chest width
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = x + w//2, y + h//2
        
        # Calculate rump angle
        rump_angle = self.calculate_rump_angle(contour)
        
        measurements = {
            "body_length_pixels": int(body_length),
            "height_withers_pixels": int(height_at_withers),
            "chest_width_pixels": int(w * 0.6),
            "rump_angle_degrees": float(rump_angle),
            "body_area_pixels": int(hull_area),
            "body_condition_score": self.calculate_body_condition(contour, hull_area),
            "contour_centroid": (int(cx), int(cy))
        }
        
        return measurements
    
    def calculate_rump_angle(self, contour):
        """Calculate rump angle from contour"""
        try:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                return angle % 180
        except:
            pass
        return 45.0
    
    def calculate_body_condition(self, contour, area):
        """Simplified body condition score"""
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.5
        bcs = 1 + (solidity - 0.5) * 8
        return max(1, min(5, bcs))

class BPAIntegration:
    def __init__(self, database_path="animal_classification.db"):
        self.database_path = database_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing classification records"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS animal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                animal_type TEXT,
                confidence REAL,
                body_length REAL,
                height_withers REAL,
                chest_width REAL,
                rump_angle REAL,
                body_condition_score REAL,
                breed TEXT,
                breed_confidence REAL,
                image_path TEXT,
                filename TEXT,
                processed_by TEXT,
                notes TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_record(self, classification_result, filename=None, processed_by="auto", notes=""):
        """Save classification record to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            measurements = classification_result.get('measurements', {})
            
            cursor.execute('''
                INSERT INTO animal_records (
                    timestamp, animal_type, confidence, body_length, height_withers,
                    chest_width, rump_angle, body_condition_score, breed, breed_confidence,
                    image_path, filename, processed_by, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                classification_result.get('timestamp', datetime.now().isoformat()),
                classification_result.get('animal_type', 'unknown'),
                classification_result.get('confidence', 0.0),
                measurements.get('body_length_pixels', 0),
                measurements.get('height_withers_pixels', 0),
                measurements.get('chest_width_pixels', 0),
                measurements.get('rump_angle_degrees', 0),
                measurements.get('body_condition_score', 0),
                classification_result.get('breed', 'unknown'),
                classification_result.get('breed_confidence', 0.0),
                classification_result.get('image_path', ''),
                filename or os.path.basename(classification_result.get('image_path', '')),
                processed_by,
                notes
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
            return record_id
            
        except Exception as e:
            logger.error(f"Error saving record: {e}")
            return None
    
    def export_to_csv(self, output_path="animal_records_export.csv"):
        """Export records to CSV file"""
        try:
            conn = sqlite3.connect(self.database_path)
            df = pd.read_sql_query("SELECT * FROM animal_records", conn)
            df.to_csv(output_path, index=False)
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def get_records(self, limit=100, offset=0):
        """Retrieve records from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM animal_records 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            records = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
            
            result = []
            for record in records:
                result.append(dict(zip(columns, record)))
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving records: {e}")
            return []

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
        metadata_path = "cattlebuffalomodelmetadata.json"
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
        
        # 3. Texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

# Initialize components
measurement_processor = BodyMeasurementProcessor()
bpa_integration = BPAIntegration()

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Classify an image as cattle or buffalo with body measurements"""
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
        
        # Extract features for classification
        bbox = data.get('bbox')
        features = extract_features_from_image(image, bbox)
        
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        # Make prediction
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class names and probabilities
        class_names = model.classes_
        probabilities_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        # Extract body measurements
        measurements = measurement_processor.process_image(image)
        
        # Simple breed classification based on type and confidence
        breed = "Unknown"
        breed_confidence = 0.0
        if prediction == "cattle":
            breed = "Dairy Cattle" if probabilities_dict.get("cattle", 0) > 0.8 else "Beef Cattle"
            breed_confidence = probabilities_dict.get("cattle", 0)
        else:
            breed = "Water Buffalo" if probabilities_dict.get("buffalo", 0) > 0.8 else "Swamp Buffalo"
            breed_confidence = probabilities_dict.get("buffalo", 0)
        
        # Compile complete result
        result = {
            'animal_type': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': probabilities_dict,
            'breed': breed,
            'breed_confidence': breed_confidence,
            'measurements': measurements,
            'timestamp': datetime.now().isoformat(),
            'model_type': metadata.get('model_type', 'Unknown') if metadata else 'Unknown'
        }
        
        # Auto-save to BPA system
        record_id = bpa_integration.save_record(
            result, 
            filename=data.get('filename', 'uploaded_image'),
            processed_by=data.get('operator', 'web_user'),
            notes=data.get('notes', '')
        )
        
        result['record_id'] = record_id
        
        logger.info(f"Classification result: {prediction} (confidence: {result['confidence']:.3f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and classification"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read image for processing
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Convert to base64 for consistency with other endpoint
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create request data
            request_data = {
                'image': f"data:image/jpeg;base64,{image_base64}",
                'filename': filename,
                'operator': request.form.get('operator', 'web_user'),
                'notes': request.form.get('notes', '')
            }
            
            # Use the classification endpoint
            with app.test_client() as client:
                response = client.post('/api/classify', json=request_data)
                return response.get_json(), response.status_code
    
    except Exception as e:
        logger.error(f"Error in file upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-classify', methods=['POST'])
def batch_classify():
    """Handle batch processing of multiple images"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process each file
                image = cv2.imread(filepath)
                if image is not None:
                    _, buffer = cv2.imencode('.jpg', image)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    request_data = {
                        'image': f"data:image/jpeg;base64,{image_base64}",
                        'filename': filename
                    }
                    
                    with app.test_client() as client:
                        response = client.post('/api/classify', json=request_data)
                        if response.status_code == 200:
                            result = response.get_json()
                            results.append(result)
                        else:
                            results.append({'filename': filename, 'error': 'Processing failed'})
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/records')
def get_records():
    """Retrieve classification records"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        records = bpa_integration.get_records(limit=limit, offset=offset)
        return jsonify({'records': records})
    
    except Exception as e:
        logger.error(f"Error retrieving records: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def export_records():
    """Export records to CSV"""
    try:
        output_path = "temp_export.csv"
        if bpa_integration.export_to_csv(output_path):
            return send_file(output_path, as_attachment=True)
        else:
            return jsonify({'error': 'Export failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'metadata_loaded': metadata is not None,
        'components': {
            'measurement_processor': True,
            'bpa_integration': True
        }
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
        logger.info("Starting unified Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to load model. Exiting.")
