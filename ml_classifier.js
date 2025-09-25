/**
 * Updated ML Classifier for Animal Classification with Body Measurements
 * Simplified version focusing on body measurements only
 */

class AnimalClassifier {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.isProcessing = false;
    }

    /**
     * Classify an image and get body measurements
     */
    async classifyImage(imageFile) {
        if (this.isProcessing) {
            throw new Error('Another image is being processed');
        }

        this.isProcessing = true;
        
        try {
            // Convert image to base64
            const imageBase64 = await this.fileToBase64(imageFile);
            
            // Send to API
            const response = await fetch(`${this.apiBaseUrl}/api/classify`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageBase64,
                    filename: imageFile.name,
                    operator: 'web_user'
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Classification failed');
            }

            const result = await response.json();
            return this.formatResults(result);
            
        } catch (error) {
            console.error('Classification error:', error);
            throw error;
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Upload and classify image via file upload endpoint
     */
    async uploadAndClassify(imageFile) {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('operator', 'web_user');

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const result = await response.json();
            return this.formatResults(result);
            
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    }

    /**
     * Format results for display (simplified - no percentages)
     */
    formatResults(apiResult) {
        if (apiResult.error) {
            throw new Error(apiResult.error);
        }

        // Simplified results focusing on body measurements
        const measurements = apiResult.measurements || {};
        
        return {
            // Basic classification
            animalType: apiResult.animal_type || 'Unknown',
            confidence: Math.round((apiResult.confidence || 0) * 100),
            recordId: apiResult.record_id,
            timestamp: apiResult.timestamp,

            // Body Measurements (main focus)
            bodyMeasurements: {
                bodyLength: measurements.body_length || 0,
                heightAtWithers: measurements.height_at_withers || 0,
                chestWidth: measurements.chest_width || 0,
                rumpAngle: measurements.rump_angle || 0,
                bodyConditionScore: measurements.body_condition_score || 0,
                bodyArea: measurements.body_area || 0
            },

            // Bounding box for visualization
            boundingBox: measurements.bounding_box || {},

            // Quality assessment (simplified)
            qualityAssessment: this.getQualityAssessment(apiResult.confidence),
            breedInfo: {
                breed: 'Automatic Detection', // Simplified breed info
                accuracy: 'Based on body measurements'
            }
        };
    }

    /**
     * Simple quality assessment based on confidence
     */
    getQualityAssessment(confidence) {
        if (confidence >= 0.8) return 'High';
        if (confidence >= 0.6) return 'Medium';
        return 'Low';
    }

    /**
     * Convert file to base64
     */
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    /**
     * Get classification records
     */
    async getRecords(limit = 50) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/records?limit=${limit}`);
            if (!response.ok) throw new Error('Failed to fetch records');
            
            const data = await response.json();
            return data.records || [];
        } catch (error) {
            console.error('Error fetching records:', error);
            return [];
        }
    }

    /**
     * Get system summary
     */
    async getSummary() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/summary`);
            if (!response.ok) throw new Error('Failed to fetch summary');
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching summary:', error);
            return null;
        }
    }

    /**
     * Export records to CSV
     */
    async exportRecords() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/export`);
            if (!response.ok) throw new Error('Export failed');
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `animal_records_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            return true;
        } catch (error) {
            console.error('Export error:', error);
            return false;
        }
    }
}

/**
 * UI Handler for displaying results
 */
class ResultsUI {
    constructor() {
        this.classifier = new AnimalClassifier();
    }

    /**
     * Display classification results
     */
    displayResults(results) {
        this.clearResults();
        
        // Basic classification info
        this.displayBasicInfo(results);
        
        // Body measurements
        this.displayBodyMeasurements(results.bodyMeasurements);
        
        // Quality assessment
        this.displayQualityAssessment(results);
        
        // Visualization
        this.displayVisualization(results);
    }

    displayBasicInfo(results) {
        const basicInfoHtml = `
            <div class="result-card">
                <h3>Classification Result</h3>
                <div class="animal-type-badge ${results.animalType.toLowerCase()}">
                    ${results.animalType.toUpperCase()}
                </div>
                <div class="confidence-level">
                    Confidence: ${results.confidence}%
                </div>
                <div class="timestamp">
                    Analyzed: ${new Date(results.timestamp).toLocaleString()}
                </div>
                ${results.recordId ? `<div class="record-id">Record ID: ${results.recordId}</div>` : ''}
            </div>
        `;
        
        document.getElementById('basicResults').innerHTML = basicInfoHtml;
    }

    displayBodyMeasurements(measurements) {
        const measurementsHtml = `
            <div class="result-card">
                <h3>Body Measurements</h3>
                <div class="measurements-grid">
                    <div class="measurement-item">
                        <label>Body Length:</label>
                        <span class="value">${measurements.bodyLength} pixels</span>
                    </div>
                    <div class="measurement-item">
                        <label>Height at Withers:</label>
                        <span class="value">${measurements.heightAtWithers} pixels</span>
                    </div>
                    <div class="measurement-item">
                        <label>Chest Width:</label>
                        <span class="value">${measurements.chestWidth} pixels</span>
                    </div>
                    <div class="measurement-item">
                        <label>Rump Angle:</label>
                        <span class="value">${measurements.rumpAngle.toFixed(1)}°</span>
                    </div>
                    <div class="measurement-item">
                        <label>Body Area:</label>
                        <span class="value">${measurements.bodyArea.toLocaleString()} pixels</span>
                    </div>
                    <div class="measurement-item">
                        <label>Body Condition Score:</label>
                        <span class="value">${measurements.bodyConditionScore.toFixed(1)}/5.0</span>
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('measurementResults').innerHTML = measurementsHtml;
    }

    displayQualityAssessment(results) {
        const qualityHtml = `
            <div class="result-card">
                <h3>Quality Assessment</h3>
                <div class="quality-info">
                    <div class="quality-badge ${results.qualityAssessment.toLowerCase()}">
                        ${results.qualityAssessment} Quality
                    </div>
                    <div class="breed-info">
                        <strong>Breed Detection:</strong> ${results.breedInfo.breed}
                    </div>
                    <div class="accuracy-note">
                        ${results.breedInfo.accuracy}
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('qualityResults').innerHTML = qualityHtml;
    }

    displayVisualization(results) {
        const vizHtml = `
            <div class="result-card">
                <h3>Measurement Visualization</h3>
                <div class="visualization">
                    <div class="animal-silhouette">
                        <div class="measurement-line length" 
                             style="width: ${Math.min(results.bodyMeasurements.bodyLength / 10, 100)}%">
                            <span>Body Length: ${results.bodyMeasurements.bodyLength}px</span>
                        </div>
                        <div class="measurement-line height" 
                             style="height: ${Math.min(results.bodyMeasurements.heightAtWithers / 10, 100)}%">
                            <span>Height: ${results.bodyMeasurements.heightAtWithers}px</span>
                        </div>
                        <div class="angle-indicator">
                            Rump Angle: ${results.bodyMeasurements.rumpAngle.toFixed(1)}°
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.getElementById('visualizationResults').innerHTML = vizHtml;
    }

    clearResults() {
        const sections = ['basicResults', 'measurementResults', 'qualityResults', 'visualizationResults'];
        sections.forEach(section => {
            const element = document.getElementById(section);
            if (element) element.innerHTML = '';
        });
    }

    displayError(error) {
        const errorHtml = `
            <div class="error-card">
                <h3>Analysis Failed</h3>
                <p>${error.message || 'An error occurred during analysis'}</p>
                <button onclick="location.reload()">Try Again</button>
            </div>
        `;
        
        document.getElementById('basicResults').innerHTML = errorHtml;
    }

    showLoading() {
        const loadingHtml = `
            <div class="loading-card">
                <h3>Analyzing Image</h3>
                <div class="spinner"></div>
                <p>Processing body measurements...</p>
            </div>
        `;
        
        document.getElementById('basicResults').innerHTML = loadingHtml;
    }
}

/**
 * Initialize the application
 */
function initializeAnimalClassifier() {
    const ui = new ResultsUI();
    
    // File upload handler
    const fileInput = document.getElementById('imageUpload');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!fileInput.files.length) {
                alert('Please select an image first');
                return;
            }
            
            const file = fileInput.files[0];
            ui.showLoading();
            
            try {
                const results = await ui.classifier.classifyImage(file);
                ui.displayResults(results);
            } catch (error) {
                ui.displayError(error);
            }
        });
    }
    
    // Drag and drop support
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length && files[0].type.startsWith('image/')) {
                fileInput.files = files;
                // Trigger analysis automatically
                if (analyzeBtn) analyzeBtn.click();
            }
        });
    }
    
    // Export functionality
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', async () => {
            const success = await ui.classifier.exportRecords();
            if (success) {
                alert('Records exported successfully!');
            } else {
                alert('Export failed. Please try again.');
            }
        });
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeAnimalClassifier);
