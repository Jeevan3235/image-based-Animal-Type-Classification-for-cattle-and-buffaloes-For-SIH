#!/usr/bin/env python3
"""
Unified Flask Backend for Cattle vs Buffalo Classification with Body Measurements
Focuses on body length, height at withers, chest width, and rump angle
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
        """Extract specific body measurements from animal image"""
        try:
            if image_array is None:
                return {"error": "Invalid image"}
            
            # Preprocess image
            processed_image = self.preprocess_image(image_array)
            
            # Detect animal body
            body_contour, bounding_rect = self.detect_animal_body(processed_image)
            if body_contour is None:
                return {"error": "Could not detect animal body"}
            
            # Extract specific measurements
            measurements = self.extract_specific_measurements(body_contour, bounding_rect, image_array)
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
        """Detect the main animal body contour with bounding box"""
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, None
        
        # Filter contours by area and aspect ratio
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Animal-specific size constraints
            if area > 5000 and 0.5 < aspect_ratio < 2.5:
                valid_contours.append(contour)
        
        if not valid_contours:
            return None, None
        
        # Get largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        bounding_rect = cv2.boundingRect(largest_contour)
        
        return largest_contour, bounding_rect
    
    def extract_specific_measurements(self, contour, bounding_rect, original_image):
        """Extract specific body measurements: body length, height at withers, chest width, rump angle"""
        x, y, w, h = bounding_rect
        
        # 1. Body Length (horizontal extent)
        body_length = w
        
        # 2. Height at Withers (vertical extent at front portion)
        height_at_withers = h
        
        # 3. Chest Width (width at the chest region - middle section)
        chest_width = self.calculate_chest_width(contour, x, y, w, h)
        
        # 4. Rump Angle (angle of the rear portion)
        rump_angle = self.calculate_rump_angle(contour)
        
        # 5. Additional useful measurements
        body_area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate body condition score based on contour shape
        body_condition_score = self.calculate_body_condition(contour, hull_area)
        
        measurements = {
            "body_length": int(body_length),
            "height_at_withers": int(height_at_withers),
            "chest_width": int(chest_width),
            "rump_angle": float(rump_angle),
            "body_area": int(body_area),
            "body_condition_score": float(body_condition_score),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h)
            }
        }
        
        return measurements
    
    def calculate_chest_width(self, contour, x, y, w, h):
        """Calculate chest width at the middle section of the animal"""
        try:
            # Create a mask for the contour
            mask = np.zeros((h, w), dtype=np.uint8)
            contour_offset = contour - [x, y]
            cv2.fillPoly(mask, [contour_offset], 255)
            
            # Find the horizontal slice at 40% height from top (chest region)
            slice_height = int(h * 0.4)
            if slice_height < h:
                slice_row = mask[slice_height, :]
                white_pixels = np.where(slice_row == 255)[0]
                if len(white_pixels) > 0:
                    chest_width = white_pixels[-1] - white_pixels[0]
                    return max(chest_width, w * 0.3)  # Ensure reasonable minimum
            return w * 0.6  # Fallback estimation
        except:
            return w * 0.6  # Fallback estimation
    
    def calculate_rump_angle(self, contour):
        """Calculate rump angle using linear regression on rear points"""
        try:
            if len(contour) < 5:
                return 45.0
            
            # Get contour points
            points = contour.reshape(-1, 2)
            
            # Find rear section (rightmost points for horizontal animal)
            x_coords = points[:, 0]
            rear_threshold = np.percentile(x_coords, 70)  # Last 30% of points
            rear_points = points[x_coords >= rear_threshold]
            
            if len(rear_points) < 2:
                return 45.0
            
            # Fit line to rear points
            vx, vy, x0, y0 = cv2.fitLine(rear_points, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx)[0] * 180 / np.pi
            angle = angle % 180
            
            # Normalize angle to be between 0-90 degrees for rump
            if angle > 90:
                angle = 180 - angle
            
            return max(10, min(80, angle))  # Reasonable range for animal rump
            
        except Exception as e:
            logger.warning(f"Rump angle calculation failed: {e}")
            return 45.0
    
    def calculate_body_condition(self, contour, hull_area):
        """Calculate body condition score based on contour shape and fullness"""
        try:
            contour_area = cv2.contourArea(contour)
            solidity = contour_area / hull_area if hull_area > 0 else 0.5
            
            # Additional shape metrics
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Combine metrics for body condition score (1-5 scale)
            bcs = 1 + (solidity - 0.4) * 10 + circularity * 2
            return max(1.0, min(5.0, bcs))
            
        except:
            return 3.0  # Average condition
    
    def calibrate_measurements(self, reference_length_cm, reference_length_pixels):
        """Calibrate pixel measurements to real-world units"""
        self.pixels_to_cm = reference_length_cm / reference_length_pixels
        self.measurement_units = "cm"
        logger.info(f"Calibrated: 1 pixel = {self.pixels_to_cm:.4f} cm")

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
                height_at_withers REAL,
                chest_width REAL,
                rump_angle REAL,
                body_condition_score REAL,
                filename TEXT,
                processed_by TEXT,
                notes TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_record(self, classification_result, filename=None, processed_by="auto", notes=""):
        """Save classification record to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            measurements = classification_result.get('measurements', {})
            
            cursor.execute('''
                INSERT INTO animal_records (
                    timestamp, animal_type, confidence, body_length, height_at_withers,
                    chest_width, rump_angle, body_condition_score, filename, 
                    processed_by, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                classification_result.get('timestamp', datetime.now().isoformat()),
                classification_result.get('animal_type', 'unknown'),
                classification_result.get('confidence', 0.0),
                measurements.get('body_length', 0),
                measurements.get('height_at_withers', 0),
                measurements.get('chest_width', 0),
                measurements.get('rump_angle', 0),
                measurements.get('body_condition_score', 0),
                filename or 'unknown',
                processed_by,
                notes
            ))
            
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
            
            logger.info(f"Record saved with ID: {record_id}")
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
            logger.info(f"Records exported to {output_path}")
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
            
            logger.info(f"Retrieved {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving records: {e}")
            return []
    
    def generate_summary_report(self):
        """Generate summary statistics"""
        try:
            records = self.get_records(limit=1000)
            if not records:
                return {"total_records": 0}
            
            df = pd.DataFrame(records)
            
            summary = {
                "total_records": len(df),
                "cattle_count": len(df[df['animal_type'] == 'cattle']),
                "buffalo_count": len(df[df['animal_type'] == 'buffalo']),
                "average_confidence": float(df['confidence'].mean()),
                "average_measurements": {
                    "body_length": float(df['body_length'].mean()),
                    "height_at_withers": float(df['height_at_withers'].mean()),
                    "chest_width": float(df['chest_width'].mean()),
                    "rump_angle": float(df['rump_angle'].mean())
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    
    try:
        # Load model
        model_path = "cattle_buffalo_model.joblib"
        if Path(model_path).exists():
            model = joblib.load(model_path)
            logger.info("Classification model loaded successfully")
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
            logger.warning(f"Metadata file not found: {metadata_path}")
            metadata = {"model_type": "Cattle/Buffalo Classifier", "version": "1.0"}
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def extract_simple_features(image_array):
    """Extract simplified features for classification"""
    try:
        # Resize image
        image = cv2.resize(image_array, (224, 224))
        
        features = []
        
        # Basic color features (mean of each channel)
        for i in range(3):
            features.append(np.mean(image[:, :, i]))
            features.append(np.std(image[:, :, i]))
        
        # Basic texture (edge density)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        return np.array(features)
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

# Initialize components
measurement_processor = BodyMeasurementProcessor()
bpa_integration = BPAIntegration()

@app.route('/')
def index():
    """Serve the main interface"""
    return jsonify({
        "message": "Animal Classification API",
        "version": "2.0",
        "endpoints": {
            "/api/classify": "POST - Classify animal and get measurements",
            "/api/upload": "POST - Upload image for classification",
            "/api/records": "GET - Get classification records",
            "/api/summary": "GET - Get system summary",
            "/api/health": "GET - Health check"
        }
    })

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
        
        # Extract simple features for classification
        features = extract_simple_features(image)
        if features is None:
            return jsonify({'error': 'Failed to extract features'}), 500
        
        # Make prediction
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = float(max(model.predict_proba(features)[0]))
        
        # Extract body measurements
        measurements = measurement_processor.process_image(image)
        
        if 'error' in measurements:
            return jsonify({'error': measurements['error']}), 500
        
        # Compile complete result
        result = {
            'animal_type': prediction,
            'confidence': confidence,
            'measurements': measurements,
            'timestamp': datetime.now().isoformat()
        }
        
        # Auto-save to BPA system
        record_id = bpa_integration.save_record(
            result, 
            filename=data.get('filename', 'uploaded_image'),
            processed_by=data.get('operator', 'web_user'),
            notes=data.get('notes', '')
        )
        
        result['record_id'] = record_id
        
        logger.info(f"Classification: {prediction} (confidence: {confidence:.3f})")
        logger.info(f"Measurements - Length: {measurements['body_length']}, "
                   f"Height: {measurements['height_at_withers']}, "
                   f"Chest: {measurements['chest_width']}, "
                   f"Rump Angle: {measurements['rump_angle']:.1f}°")
        
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
            
            # Convert to base64 for consistency
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

@app.route('/api/records', methods=['GET'])
def get_records():
    """Retrieve classification records"""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        records = bpa_integration.get_records(limit=limit, offset=offset)
        return jsonify({'records': records, 'count': len(records)})
    
    except Exception as e:
        logger.error(f"Error retrieving records: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get system summary and statistics"""
    try:
        summary = bpa_integration.generate_summary_report()
        return jsonify(summary)
    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['GET'])
def export_records():
    """Export records to CSV"""
    try:
        output_path = "animal_records_export.csv"
        if bpa_integration.export_to_csv(output_path):
            return send_file(output_path, as_attachment=True,
                           download_name=f"animal_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            return jsonify({'error': 'Export failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibrate', methods=['POST'])
def calibrate_measurements():
    """Calibrate pixel measurements to real-world units"""
    try:
        data = request.get_json()
        reference_length_cm = data.get('reference_length_cm')
        reference_length_pixels = data.get('reference_length_pixels')
        
        if not reference_length_cm or not reference_length_pixels:
            return jsonify({'error': 'Reference length in cm and pixels required'}), 400
        
        measurement_processor.calibrate_measurements(reference_length_cm, reference_length_pixels)
        
        return jsonify({
            'message': 'Calibration successful',
            'pixels_to_cm_ratio': measurement_processor.pixels_to_cm,
            'units': measurement_processor.measurement_units
        })
    
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_initialized': True,
        'timestamp': datetime.now().isoformat()
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
        logger.info("Starting Animal Classification API Server...")
        logger.info("Available endpoints:")
        logger.info("  POST /api/classify - Classify animal with measurements")
        logger.info("  POST /api/upload - Upload image for classification")
        logger.info("  GET  /api/records - Get classification records")
        logger.info("  GET  /api/summary - Get system summary")
        logger.info("  GET  /api/health - Health check")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Exiting.")
