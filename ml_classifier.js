/**
 * Machine Learning Cattle vs Buffalo Classifier
 * Integrates with the trained Python model
 */

class MLClassifier {
    constructor() {
        this.modelLoaded = false;
        this.model = null;
        this.metadata = null;
        this.apiEndpoint = 'http://localhost:5000/api/classify'; // Backend API endpoint with correct port
    }

    async loadModel() {
        try {
            // Load model metadata
            const response = await fetch('/models/cattle_buffalo_model_metadata.json');
            this.metadata = await response.json();
            this.modelLoaded = true;
            console.log('ML Model metadata loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading model metadata:', error);
            return false;
        }
    }

    async classifyImage(imageElement, bbox = null) {
        if (!this.modelLoaded) {
            const loaded = await this.loadModel();
            if (!loaded) {
                throw new Error('Failed to load model');
            }
        }

        try {
            // Convert image to base64
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = imageElement.naturalWidth || imageElement.width;
            canvas.height = imageElement.naturalHeight || imageElement.height;
            
            // Draw image
            ctx.drawImage(imageElement, 0, 0);
            
            // Crop to bounding box if provided
            if (bbox) {
                const { xmin, ymin, xmax, ymax } = bbox;
                const croppedCanvas = document.createElement('canvas');
                const croppedCtx = croppedCanvas.getContext('2d');
                
                croppedCanvas.width = xmax - xmin;
                croppedCanvas.height = ymax - ymin;
                
                croppedCtx.drawImage(
                    canvas,
                    xmin, ymin, xmax - xmin, ymax - ymin,
                    0, 0, xmax - xmin, ymax - ymin
                );
                
                canvas = croppedCanvas;
            }
            
            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to backend for classification
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData,
                    bbox: bbox
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result;

        } catch (error) {
            console.error('Error classifying image:', error);
            throw error;
        }
    }

    // Fallback to rule-based classification if ML model fails
    fallbackClassification(imageElement) {
        console.log('Using fallback rule-based classification');
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = imageElement.naturalWidth || imageElement.width;
        canvas.height = imageElement.naturalHeight || imageElement.height;
        ctx.drawImage(imageElement, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Use existing rule-based analysis
        const features = extractAdvancedFeatures(data, canvas.width, canvas.height);
        const classification = advancedClassification(features);
        const breedResult = identifyBreed(classification.type, features);
        
        return {
            prediction: classification.type,
            confidence: classification.confidence / 100, // Convert to 0-1 range
            probabilities: {
                [classification.type]: classification.confidence / 100,
                [classification.type === 'Cattle' ? 'Buffalo' : 'Cattle']: (100 - classification.confidence) / 100
            },
            breed: breedResult.breed,
            breedConfidence: breedResult.breedConfidence / 100
        };
    }
}

// Enhanced classification function that uses ML model with fallback
async function enhancedMLClassification(imageElement, bbox = null) {
    const mlClassifier = new MLClassifier();
    
    try {
        // Try ML classification first
        const result = await mlClassifier.classifyImage(imageElement, bbox);
        console.log('ML Classification result:', result);
        return result;
    } catch (error) {
        console.warn('ML classification failed, using fallback:', error);
        // Fallback to rule-based classification
        return mlClassifier.fallbackClassification(imageElement);
    }
}

// Update the existing classification function to use ML model
async function performMLAnalysis(imageData, width, height) {
    // Create a temporary image element for ML classification
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    
    // Put image data into canvas
    const imageDataObj = new ImageData(imageData, width, height);
    ctx.putImageData(imageDataObj, 0, 0);
    
    // Convert canvas to image element
    const imageElement = new Image();
    imageElement.src = canvas.toDataURL();
    
    // Wait for image to load
    return new Promise((resolve) => {
        imageElement.onload = async () => {
            try {
                const result = await enhancedMLClassification(imageElement);
                
                // Convert result to expected format
                const mlResult = {
                    type: result.prediction,
                    confidence: Math.round(result.confidence * 100),
                    breed: result.breed || 'Unknown',
                    breedConfidence: result.breedConfidence ? Math.round(result.breedConfidence * 100) : 0,
                    scores: {
                        cattle: result.probabilities.Cattle || 0,
                        buffalo: result.probabilities.Buffalo || 0
                    },
                    analysis: {
                        method: 'ML Model',
                        modelType: 'Trained on cow-and-buffalo.v1i.tensorflow dataset'
                    }
                };
                
                resolve(mlResult);
            } catch (error) {
                console.error('ML analysis failed:', error);
                // Fallback to original analysis
                const features = extractAdvancedFeatures(imageData, width, height);
                const classification = advancedClassification(features);
                const breedResult = identifyBreed(classification.type, features);
                
                resolve({
                    type: classification.type,
                    confidence: classification.confidence,
                    breed: breedResult.breed,
                    breedConfidence: breedResult.breedConfidence,
                    scores: classification.scores,
                    analysis: features,
                    method: 'Rule-based (Fallback)'
                });
            }
        };
    });
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MLClassifier, enhancedMLClassification, performMLAnalysis };
}
