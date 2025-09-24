from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from animal_classifier import AnimalTypeClassifier
from bpa_integration import BPAIntegration
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
classifier = AnimalTypeClassifier()
bpa_integration = BPAIntegration()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_animal():
    """
    Handle image upload and classification
    """
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
            
            # Process image
            result = classifier.classify_animal(filepath)
            
            if 'error' in result:
                return jsonify(result), 400
            
            # Auto-save to BPA system
            record_id = bpa_integration.save_record(
                result, 
                filename=filename,
                processed_by=request.form.get('operator', 'web_user'),
                notes=request.form.get('notes', '')
            )
            
            result['record_id'] = record_id
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-classify', methods=['POST'])
def batch_classify():
    """
    Handle batch processing of multiple images
    """
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
                
                result = classifier.classify_animal(filepath)
                result['filename'] = filename
                
                if 'error' not in result:
                    bpa_integration.save_record(result, filename=filename)
                
                results.append(result)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/records')
def get_records():
    """
    Retrieve classification records
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        records = bpa_integration.get_records(limit=limit, offset=offset)
        return jsonify({'records': records})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report')
def generate_report():
    """
    Generate classification report
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        report = bpa_integration.generate_report(start_date, end_date)
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export')
def export_records():
    """
    Export records to CSV
    """
    try:
        output_path = "temp_export.csv"
        if bpa_integration.export_to_csv(output_path):
            return send_file(output_path, as_attachment=True)
        else:
            return jsonify({'error': 'Export failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
