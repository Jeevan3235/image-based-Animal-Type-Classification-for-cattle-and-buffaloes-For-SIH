// Simple, Working Cattle vs Buffalo Classifier
// Fixed algorithm without NaN issues

let isReady = true;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const classifyBtn = document.getElementById('classifyBtn');
const resultSection = document.getElementById('resultSection');
const predictedType = document.getElementById('predictedType');
const confidenceScore = document.getElementById('confidenceScore');
const predictedBreed = document.getElementById('predictedBreed');
const breedConfidenceEl = document.getElementById('breedConfidence');
const confidenceFill = document.getElementById('confidenceFill');
const btnText = document.querySelector('.btn-text');
const loadingSpinner = document.querySelector('.loading-spinner');
const debugInfo = document.getElementById('debugInfo');
const debugContent = document.getElementById('debugContent');
const breedInfo = document.getElementById('breedInfo');
const breedDetails = document.getElementById('breedDetails');
const buffaloBreeds = document.getElementById('buffaloBreeds');
const cattleBreeds = document.getElementById('cattleBreeds');

document.addEventListener('DOMContentLoaded', function() {
    console.log('Professional Classifier Ready!');
    setupEventListeners();
    populateBreedDatabase();
    runAccuracyTests();
    showStatusMessage('Professional Classifier Ready! Upload an image for analysis.', 'success');
});

function runAccuracyTests() {
    // Test the advanced classification algorithm with realistic patterns
    console.log('Running advanced accuracy tests...');
    
    // Test buffalo patterns (realistic buffalo characteristics)
    const buffaloTestFeatures = {
        colorHistogram: { 
            black: 0.4, darkGray: 0.3, gray: 0.15, lightGray: 0.05, 
            white: 0.02, brown: 0.03, lightBrown: 0.02, tan: 0.01, 
            darkBrown: 0.01, red: 0.01, yellow: 0.0 
        },
        brightnessDistribution: { 
            dark: 0.75, medium: 0.2, light: 0.05, 
            average: 65, variance: 1200 
        },
        contrastAnalysis: { 
            low: 0.4, medium: 0.4, high: 0.2, 
            average: 45 
        },
        textureFeatures: { 
            textureScore: 2.5, smoothness: 12, contrast: 25 
        },
        edgeAnalysis: { 
            edgeRatio: 0.08, strongEdgeRatio: 0.03, edgeDensity: 0.006 
        },
        regionAnalysis: {
            topLeft: { dark: 0.8, light: 0.2 },
            topRight: { dark: 0.7, light: 0.3 },
            bottomLeft: { dark: 0.75, light: 0.25 },
            bottomRight: { dark: 0.8, light: 0.2 }
        }
    };
    
    // Test cattle patterns (realistic cattle characteristics)
    const cattleTestFeatures = {
        colorHistogram: { 
            black: 0.15, darkGray: 0.1, gray: 0.05, lightGray: 0.05, 
            white: 0.35, brown: 0.2, lightBrown: 0.05, tan: 0.03, 
            darkBrown: 0.01, red: 0.01, yellow: 0.0 
        },
        brightnessDistribution: { 
            dark: 0.2, medium: 0.5, light: 0.3, 
            average: 125, variance: 2800 
        },
        contrastAnalysis: { 
            low: 0.2, medium: 0.4, high: 0.4, 
            average: 75 
        },
        textureFeatures: { 
            textureScore: 4.2, smoothness: 28, contrast: 45 
        },
        edgeAnalysis: { 
            edgeRatio: 0.15, strongEdgeRatio: 0.08, edgeDensity: 0.012 
        },
        regionAnalysis: {
            topLeft: { dark: 0.3, light: 0.7 },
            topRight: { dark: 0.4, light: 0.6 },
            bottomLeft: { dark: 0.2, light: 0.8 },
            bottomRight: { dark: 0.3, light: 0.7 }
        }
    };
    
    // Test Holstein pattern (black and white cattle)
    const holsteinTestFeatures = {
        colorHistogram: { 
            black: 0.4, darkGray: 0.05, gray: 0.05, lightGray: 0.05, 
            white: 0.4, brown: 0.02, lightBrown: 0.01, tan: 0.01, 
            darkBrown: 0.01, red: 0.0, yellow: 0.0 
        },
        brightnessDistribution: { 
            dark: 0.4, medium: 0.3, light: 0.3, 
            average: 110, variance: 4500 
        },
        contrastAnalysis: { 
            low: 0.1, medium: 0.3, high: 0.6, 
            average: 95 
        },
        textureFeatures: { 
            textureScore: 5.8, smoothness: 35, contrast: 65 
        },
        edgeAnalysis: { 
            edgeRatio: 0.25, strongEdgeRatio: 0.15, edgeDensity: 0.020 
        },
        regionAnalysis: {
            topLeft: { dark: 0.6, light: 0.4 },
            topRight: { dark: 0.3, light: 0.7 },
            bottomLeft: { dark: 0.4, light: 0.6 },
            bottomRight: { dark: 0.5, light: 0.5 }
        }
    };
    
    const buffaloResult = advancedClassification(buffaloTestFeatures);
    const cattleResult = advancedClassification(cattleTestFeatures);
    const holsteinResult = advancedClassification(holsteinTestFeatures);
    
    console.log('Buffalo test result:', buffaloResult);
    console.log('Cattle test result:', cattleResult);
    console.log('Holstein test result:', holsteinResult);
    
    // Validate results with more realistic expectations
    if (buffaloResult.type === 'Buffalo' && buffaloResult.confidence > 70) {
        console.log('✅ Buffalo classification test PASSED');
    } else {
        console.log('❌ Buffalo classification test FAILED - Expected Buffalo, got', buffaloResult.type, 'with confidence', buffaloResult.confidence);
    }
    
    if (cattleResult.type === 'Cattle' && cattleResult.confidence > 70) {
        console.log('✅ Cattle classification test PASSED');
    } else {
        console.log('❌ Cattle classification test FAILED - Expected Cattle, got', cattleResult.type, 'with confidence', cattleResult.confidence);
    }
    
    if (holsteinResult.type === 'Cattle' && holsteinResult.confidence > 70) {
        console.log('✅ Holstein classification test PASSED');
    } else {
        console.log('❌ Holstein classification test FAILED - Expected Cattle, got', holsteinResult.type, 'with confidence', holsteinResult.confidence);
    }
}

function setupEventListeners() {
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    classifyBtn.addEventListener('click', classifyImage);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function loadSampleImage(imageUrl) {
    previewImage.src = imageUrl;
    previewSection.style.display = 'block';
    resultSection.style.display = 'none';
    previewSection.scrollIntoView({ behavior: 'smooth' });
    showStatusMessage('Sample image loaded! Click "Analyze Image".', 'success');
}

function handleFile(file) {
    if (!isValidImageFile(file)) {
        showStatusMessage('Please select a valid image file (JPG, JPEG, or PNG).', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        resultSection.style.display = 'none';
        previewSection.scrollIntoView({ behavior: 'smooth' });
        showStatusMessage('Image uploaded! Click "Analyze Image".', 'success');
    };
    reader.readAsDataURL(file);
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    return validTypes.includes(file.type);
}

async function classifyImage() {
    if (!previewImage.src) {
        showStatusMessage('Please upload an image first.', 'error');
        return;
    }
    
    setLoadingState(true);
    showStatusMessage('Performing professional analysis...', 'info');
    
    try {
        // Validate image before analysis
        if (!validateImageForAnalysis(previewImage)) {
            showStatusMessage('Image quality too low for accurate analysis. Please upload a clearer image.', 'error');
            setLoadingState(false);
            return;
        }
        
        const result = await simpleWorkingAnalysis(previewImage);
        
        // Validate results
        const validatedResult = validateClassificationResult(result);
        
        displayResults(validatedResult);
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
        
        // Show appropriate success message
        if (validatedResult.confidence > 80) {
            showStatusMessage(`High confidence analysis complete! ${validatedResult.type} detected.`, 'success');
        } else if (validatedResult.confidence > 60) {
            showStatusMessage(`Analysis complete! ${validatedResult.type} detected with moderate confidence.`, 'info');
        } else {
            showStatusMessage(`Analysis complete! ${validatedResult.type} detected with low confidence. Please try a clearer image.`, 'warning');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        showStatusMessage('Error during analysis. Please try again with a different image.', 'error');
    } finally {
        setLoadingState(false);
    }
}

function validateImageForAnalysis(imageElement) {
    // Check image dimensions
    if (imageElement.naturalWidth < 100 || imageElement.naturalHeight < 100) {
        return false;
    }
    
    // Check if image is loaded
    if (!imageElement.complete) {
        return false;
    }
    
    return true;
}

function validateClassificationResult(result) {
    // Ensure minimum confidence thresholds
    if (result.confidence < 30) {
        result.type = 'Unknown';
        result.confidence = 30;
        result.breed = 'Unknown';
        result.breedConfidence = 0;
    }
    
    // Validate breed confidence
    if (result.breedConfidence < 30) {
        result.breed = 'Unknown';
        result.breedConfidence = 0;
    }
    
    // Add quality indicators
    result.quality = {
        imageQuality: result.confidence > 70 ? 'High' : result.confidence > 50 ? 'Medium' : 'Low',
        breedAccuracy: result.breedConfidence > 70 ? 'High' : result.breedConfidence > 50 ? 'Medium' : 'Low'
    };
    
    return result;
}

async function simpleWorkingAnalysis(imageElement) {
    return new Promise(async (resolve) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        if (imageElement.complete) {
            await analyze();
        } else {
            imageElement.onload = async () => await analyze();
        }
        
        async function analyze() {
            canvas.width = imageElement.naturalWidth;
            canvas.height = imageElement.naturalHeight;
            ctx.drawImage(imageElement, 0, 0);
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Try ML classification first, fallback to rule-based
            try {
                const result = await performMLAnalysis(data, canvas.width, canvas.height);
                resolve(result);
            } catch (error) {
                console.warn('ML analysis failed, using rule-based fallback:', error);
                const result = performSimpleAnalysis(data, canvas.width, canvas.height);
                resolve(result);
            }
        }
    });
}

function performSimpleAnalysis(imageData, width, height) {
    // Advanced computer vision analysis with multiple techniques
    const features = extractAdvancedFeatures(imageData, width, height);
    const classification = advancedClassification(features);
    const breedResult = identifyBreed(classification.type, features);
    
    return {
        type: classification.type,
        confidence: classification.confidence,
        breed: breedResult.breed,
        breedConfidence: breedResult.breedConfidence,
        scores: classification.scores,
        analysis: features,
        features: features // Additional feature data for debugging
    };
}

function extractAdvancedFeatures(imageData, width, height) {
    // Advanced computer vision feature extraction
    const features = {
        colorHistogram: {},
        textureFeatures: {},
        shapeFeatures: {},
        brightnessDistribution: {},
        contrastAnalysis: {},
        edgeAnalysis: {},
        regionAnalysis: {}
    };
    
    // Create image matrix for analysis
    const imageMatrix = createImageMatrix(imageData, width, height);
    
    // 1. Advanced Color Analysis
    features.colorHistogram = analyzeColorHistogram(imageMatrix);
    
    // 2. Texture Analysis
    features.textureFeatures = analyzeTexture(imageMatrix);
    
    // 3. Brightness Distribution Analysis
    features.brightnessDistribution = analyzeBrightnessDistribution(imageMatrix);
    
    // 4. Contrast Analysis
    features.contrastAnalysis = analyzeContrast(imageMatrix);
    
    // 5. Edge Detection and Analysis
    features.edgeAnalysis = analyzeEdges(imageMatrix);
    
    // 6. Regional Analysis
    features.regionAnalysis = analyzeRegions(imageMatrix);
    
    return features;
}

function createImageMatrix(imageData, width, height) {
    const matrix = [];
    for (let y = 0; y < height; y++) {
        matrix[y] = [];
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            matrix[y][x] = {
                r: imageData[index],
                g: imageData[index + 1],
                b: imageData[index + 2],
                brightness: (imageData[index] + imageData[index + 1] + imageData[index + 2]) / 3
            };
        }
    }
    return matrix;
}

function analyzeColorHistogram(matrix) {
    const histogram = {
        black: 0, darkGray: 0, gray: 0, lightGray: 0, white: 0,
        darkBrown: 0, brown: 0, lightBrown: 0, tan: 0,
        red: 0, yellow: 0, total: 0
    };
    
    const height = matrix.length;
    const width = matrix[0].length;
    
    // Sample every 2nd pixel for better accuracy
    for (let y = 0; y < height; y += 2) {
        for (let x = 0; x < width; x += 2) {
            const pixel = matrix[y][x];
            classifyPixelColorAdvanced(pixel.r, pixel.g, pixel.b, histogram);
            histogram.total++;
        }
    }
    
    // Convert to ratios
    const ratios = {};
    for (const [color, count] of Object.entries(histogram)) {
        if (color !== 'total') {
            ratios[color] = count / histogram.total;
        }
    }
    
    return ratios;
}

function analyzeTexture(matrix) {
    const height = matrix.length;
    const width = matrix[0].length;
    let textureScore = 0;
    let smoothness = 0;
    let contrast = 0;
    
    // Calculate local binary patterns (simplified)
    for (let y = 1; y < height - 1; y += 3) {
        for (let x = 1; x < width - 1; x += 3) {
            const center = matrix[y][x].brightness;
            const neighbors = [
                matrix[y-1][x-1].brightness, matrix[y-1][x].brightness, matrix[y-1][x+1].brightness,
                matrix[y][x-1].brightness, matrix[y][x+1].brightness,
                matrix[y+1][x-1].brightness, matrix[y+1][x].brightness, matrix[y+1][x+1].brightness
            ];
            
            // Calculate local contrast
            const localContrast = Math.max(...neighbors) - Math.min(...neighbors);
            contrast += localContrast;
            
            // Calculate smoothness
            const avgNeighbor = neighbors.reduce((a, b) => a + b, 0) / neighbors.length;
            smoothness += Math.abs(center - avgNeighbor);
            
            // Texture pattern detection
            let pattern = 0;
            neighbors.forEach(neighbor => {
                if (neighbor > center) pattern++;
            });
            textureScore += pattern;
        }
    }
    
    const totalSamples = Math.floor((height - 2) / 3) * Math.floor((width - 2) / 3);
    
    return {
        textureScore: textureScore / totalSamples,
        smoothness: smoothness / totalSamples,
        contrast: contrast / totalSamples
    };
}

function analyzeBrightnessDistribution(matrix) {
    const height = matrix.length;
    const width = matrix[0].length;
    let darkPixels = 0, mediumPixels = 0, lightPixels = 0;
    let totalBrightness = 0;
    let brightnessVariance = 0;
    
    // Sample pixels
    for (let y = 0; y < height; y += 2) {
        for (let x = 0; x < width; x += 2) {
            const brightness = matrix[y][x].brightness;
            totalBrightness += brightness;
            
            if (brightness < 60) darkPixels++;
            else if (brightness < 180) mediumPixels++;
            else lightPixels++;
        }
    }
    
    const totalPixels = darkPixels + mediumPixels + lightPixels;
    const avgBrightness = totalBrightness / totalPixels;
    
    // Calculate variance
    for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
            const brightness = matrix[y][x].brightness;
            brightnessVariance += Math.pow(brightness - avgBrightness, 2);
        }
    }
    
    return {
        dark: darkPixels / totalPixels,
        medium: mediumPixels / totalPixels,
        light: lightPixels / totalPixels,
        average: avgBrightness,
        variance: brightnessVariance / (totalPixels / 4)
    };
}

function analyzeContrast(matrix) {
    const height = matrix.length;
    const width = matrix[0].length;
    let lowContrast = 0, mediumContrast = 0, highContrast = 0;
    let totalContrast = 0;
    
    for (let y = 0; y < height - 1; y += 2) {
        for (let x = 0; x < width - 1; x += 2) {
            const current = matrix[y][x].brightness;
            const right = matrix[y][x + 1].brightness;
            const down = matrix[y + 1][x].brightness;
            
            const contrast1 = Math.abs(current - right);
            const contrast2 = Math.abs(current - down);
            const avgContrast = (contrast1 + contrast2) / 2;
            
            totalContrast += avgContrast;
            
            if (avgContrast < 30) lowContrast++;
            else if (avgContrast < 100) mediumContrast++;
            else highContrast++;
        }
    }
    
    const totalSamples = lowContrast + mediumContrast + highContrast;
    
    return {
        low: lowContrast / totalSamples,
        medium: mediumContrast / totalSamples,
        high: highContrast / totalSamples,
        average: totalContrast / totalSamples
    };
}

function analyzeEdges(matrix) {
    const height = matrix.length;
    const width = matrix[0].length;
    let edgePixels = 0;
    let strongEdges = 0;
    let edgeDensity = 0;
    
    // Sobel edge detection (simplified)
    for (let y = 1; y < height - 1; y += 2) {
        for (let x = 1; x < width - 1; x += 2) {
            const gx = Math.abs(
                matrix[y-1][x+1].brightness + 2*matrix[y][x+1].brightness + matrix[y+1][x+1].brightness -
                matrix[y-1][x-1].brightness - 2*matrix[y][x-1].brightness - matrix[y+1][x-1].brightness
            );
            
            const gy = Math.abs(
                matrix[y+1][x-1].brightness + 2*matrix[y+1][x].brightness + matrix[y+1][x+1].brightness -
                matrix[y-1][x-1].brightness - 2*matrix[y-1][x].brightness - matrix[y-1][x+1].brightness
            );
            
            const edgeMagnitude = Math.sqrt(gx*gx + gy*gy);
            
            if (edgeMagnitude > 50) {
                edgePixels++;
                if (edgeMagnitude > 100) {
                    strongEdges++;
                }
            }
        }
    }
    
    const totalSamples = Math.floor((height - 2) / 2) * Math.floor((width - 2) / 2);
    
    return {
        edgeRatio: edgePixels / totalSamples,
        strongEdgeRatio: strongEdges / totalSamples,
        edgeDensity: edgePixels / (height * width)
    };
}

function analyzeRegions(matrix) {
    const height = matrix.length;
    const width = matrix[0].length;
    
    // Divide image into regions and analyze each
    const regions = {
        topLeft: { dark: 0, light: 0, total: 0 },
        topRight: { dark: 0, light: 0, total: 0 },
        bottomLeft: { dark: 0, light: 0, total: 0 },
        bottomRight: { dark: 0, light: 0, total: 0 }
    };
    
    const midY = Math.floor(height / 2);
    const midX = Math.floor(width / 2);
    
    for (let y = 0; y < height; y += 3) {
        for (let x = 0; x < width; x += 3) {
            const brightness = matrix[y][x].brightness;
            let region;
            
            if (y < midY && x < midX) region = regions.topLeft;
            else if (y < midY && x >= midX) region = regions.topRight;
            else if (y >= midY && x < midX) region = regions.bottomLeft;
            else region = regions.bottomRight;
            
            if (brightness < 100) region.dark++;
            else region.light++;
            region.total++;
        }
    }
    
    // Calculate ratios for each region
    const regionRatios = {};
    for (const [regionName, region] of Object.entries(regions)) {
        regionRatios[regionName] = {
            dark: region.dark / region.total,
            light: region.light / region.total
        };
    }
    
    return regionRatios;
}

function classifyPixelColorAdvanced(r, g, b, stats) {
    const brightness = (r + g + b) / 3;
    const maxRGB = Math.max(r, g, b);
    const minRGB = Math.min(r, g, b);
    const saturation = maxRGB - minRGB;
    
    // Enhanced color classification with better buffalo detection
    if (brightness < 20) {
        stats.black++;
    } else if (brightness < 45) {
        stats.darkGray++;
    } else if (brightness < 70 && saturation < 35) {
        stats.gray++;
    } else if (brightness < 100 && saturation < 30) {
        stats.lightGray++;
    } else if (brightness > 170 && saturation < 20) {
        stats.white++;
    } else if (r > g && g > b && r - b > 20) {
        if (brightness < 60) stats.darkBrown++;
        else if (brightness < 110) stats.brown++;
        else if (brightness < 150) stats.lightBrown++;
        else stats.tan++;
    } else if (r > g * 1.1 && r > b * 1.1 && r > 90) {
        stats.red++;
    } else if (r > 130 && g > 130 && b < r * 0.9 && brightness > 110) {
        stats.yellow++;
    } else if (brightness < 90 && saturation < 45) {
        stats.darkGray++; // Additional dark color detection
    }
}

function classifyPixelColor(r, g, b, stats) {
    const brightness = (r + g + b) / 3;
    const maxRGB = Math.max(r, g, b);
    const minRGB = Math.min(r, g, b);
    const saturation = maxRGB - minRGB;
    
    // Enhanced color classification for better buffalo detection
    
    // Black and very dark colors (strong buffalo indicator)
    if (brightness < 25) {
        stats.black++;
    } else if (brightness < 50) {
        stats.darkGray++;
    }
    // Dark gray colors (buffalo characteristic)
    else if (brightness < 80 && saturation < 40) {
        stats.gray++;
    } else if (brightness < 120 && saturation < 35) {
        stats.lightGray++;
    }
    // White colors (cattle characteristic)
    else if (brightness > 180 && saturation < 25) {
        stats.white++;
    }
    // Brown colors (cattle characteristic - more sophisticated detection)
    else if (r > g && g > b && r - b > 25) {
        if (brightness < 70) stats.darkBrown++;
        else if (brightness < 120) stats.brown++;
        else if (brightness < 160) stats.lightBrown++;
        else stats.tan++;
    }
    // Red colors (some cattle breeds)
    else if (r > g * 1.15 && r > b * 1.15 && r > 100) {
        stats.red++;
    }
    // Yellow/cream colors (cattle characteristic)
    else if (r > 140 && g > 140 && b < r * 0.85 && brightness > 120) {
        stats.yellow++;
    }
    // Additional dark color detection for buffalo
    else if (brightness < 100 && saturation < 50) {
        // This could be dark gray or dark brown - classify as dark gray for buffalo
        stats.darkGray++;
    }
}

function advancedClassification(features) {
    // Advanced multi-feature classification using computer vision techniques
    const { colorHistogram, textureFeatures, brightnessDistribution, contrastAnalysis, edgeAnalysis, regionAnalysis } = features;
    
    let cattleScore = 0;
    let buffaloScore = 0;
    
    // 1. COLOR-BASED CLASSIFICATION
    const darkColors = colorHistogram.black + colorHistogram.darkGray;
    const grayColors = colorHistogram.gray + colorHistogram.lightGray;
    const brownColors = colorHistogram.brown + colorHistogram.lightBrown + colorHistogram.tan + colorHistogram.darkBrown;
    const lightColors = colorHistogram.white + colorHistogram.yellow;
    
    // Buffalo color characteristics
    if (darkColors > 0.25) {
        buffaloScore += darkColors * 5.0; // Very strong indicator
    }
    if (grayColors > 0.15) {
        buffaloScore += grayColors * 3.0;
    }
    if (darkColors > 0.4) {
        buffaloScore += 4.0; // Strong buffalo indicator
    }
    
    // Cattle color characteristics
    if (brownColors > 0.15) {
        cattleScore += brownColors * 4.0;
    }
    if (lightColors > 0.25) {
        cattleScore += lightColors * 3.5;
    }
    if (colorHistogram.white > 0.3) {
        cattleScore += 3.0; // White is strong cattle indicator
    }
    
    // 2. BRIGHTNESS DISTRIBUTION ANALYSIS
    if (brightnessDistribution.dark > 0.5) {
        buffaloScore += brightnessDistribution.dark * 3.0;
    }
    if (brightnessDistribution.light > 0.3) {
        cattleScore += brightnessDistribution.light * 2.5;
    }
    if (brightnessDistribution.average < 80) {
        buffaloScore += 2.0; // Low average brightness = buffalo
    }
    if (brightnessDistribution.average > 120) {
        cattleScore += 2.0; // High average brightness = cattle
    }
    
    // 3. TEXTURE ANALYSIS
    if (textureFeatures.smoothness < 20) {
        buffaloScore += 2.0; // Buffalo have smoother texture
    }
    if (textureFeatures.contrast > 30) {
        cattleScore += 1.5; // Cattle have more textured appearance
    }
    
    // 4. CONTRAST ANALYSIS
    if (contrastAnalysis.low > 0.4) {
        buffaloScore += contrastAnalysis.low * 2.5; // Uniform coloring
    }
    if (contrastAnalysis.high > 0.3) {
        cattleScore += contrastAnalysis.high * 2.0; // Spotted patterns
    }
    
    // 5. EDGE ANALYSIS
    if (edgeAnalysis.edgeRatio < 0.1) {
        buffaloScore += 1.5; // Fewer edges = smoother buffalo
    }
    if (edgeAnalysis.strongEdgeRatio > 0.05) {
        cattleScore += 1.0; // More defined edges = cattle patterns
    }
    
    // 6. REGIONAL ANALYSIS
    let uniformRegions = 0;
    let variedRegions = 0;
    
    for (const [regionName, region] of Object.entries(regionAnalysis)) {
        const regionVariation = Math.abs(region.dark - region.light);
        if (regionVariation < 0.3) {
            uniformRegions++;
        } else {
            variedRegions++;
        }
    }
    
    if (uniformRegions > 2) {
        buffaloScore += 2.0; // Uniform across regions
    }
    if (variedRegions > 2) {
        cattleScore += 1.5; // Varied across regions
    }
    
    // 7. COMBINED FEATURE ANALYSIS
    // Spotted pattern detection (strong cattle indicator)
    if (lightColors > 0.2 && darkColors > 0.15 && contrastAnalysis.high > 0.25) {
        cattleScore += 3.0;
    }
    
    // Very dark and uniform (strong buffalo indicator)
    if (darkColors > 0.5 && contrastAnalysis.low > 0.4 && brightnessDistribution.average < 70) {
        buffaloScore += 4.0;
    }
    
    // Pure white animals (cattle)
    if (colorHistogram.white > 0.6) {
        cattleScore += 4.0;
    }
    
    // Very dark animals with low texture variation (buffalo)
    if (darkColors > 0.4 && textureFeatures.smoothness < 15 && brightnessDistribution.average < 80) {
        buffaloScore += 3.5;
    }
    
    // Determine final classification
    const totalScore = cattleScore + buffaloScore;
    let type, confidence;
    
    if (totalScore === 0) {
        type = 'Unknown';
        confidence = 30;
    } else {
        const cattlePercentage = (cattleScore / totalScore) * 100;
        const buffaloPercentage = (buffaloScore / totalScore) * 100;
        
        if (cattlePercentage > buffaloPercentage) {
            type = 'Cattle';
            confidence = Math.min(95, Math.round(cattlePercentage));
        } else {
            type = 'Buffalo';
            confidence = Math.min(95, Math.round(buffaloPercentage));
        }
    }
    
    // Ensure minimum confidence
    confidence = Math.max(confidence, 40);
    
    // Enhanced debug information
    console.log('Advanced Classification Debug:', {
        colorAnalysis: {
            darkColors: darkColors.toFixed(3),
            grayColors: grayColors.toFixed(3),
            brownColors: brownColors.toFixed(3),
            lightColors: lightColors.toFixed(3)
        },
        brightnessAnalysis: {
            average: brightnessDistribution.average.toFixed(1),
            dark: brightnessDistribution.dark.toFixed(3),
            light: brightnessDistribution.light.toFixed(3)
        },
        textureAnalysis: {
            smoothness: textureFeatures.smoothness.toFixed(1),
            contrast: textureFeatures.contrast.toFixed(1)
        },
        contrastAnalysis: {
            low: contrastAnalysis.low.toFixed(3),
            high: contrastAnalysis.high.toFixed(3)
        },
        edgeAnalysis: {
            edgeRatio: edgeAnalysis.edgeRatio.toFixed(3),
            strongEdgeRatio: edgeAnalysis.strongEdgeRatio.toFixed(3)
        },
        scores: {
            cattle: cattleScore.toFixed(2),
            buffalo: buffaloScore.toFixed(2)
        },
        result: type,
        confidence: confidence
    });
    
    return {
        type,
        confidence,
        scores: {
            cattle: cattleScore,
            buffalo: buffaloScore
        }
    };
}

function identifyBreed(animalType, features) {
    if (animalType === 'Unknown') {
        return { breed: 'Unknown', breedConfidence: 0 };
    }
    
    // Enhanced breed identification with advanced features
    const breedResult = findBestMatchingBreed(animalType, {
        black: features.colorHistogram.black + features.colorHistogram.darkGray,
        gray: features.colorHistogram.gray + features.colorHistogram.lightGray,
        brown: features.colorHistogram.brown + features.colorHistogram.lightBrown + features.colorHistogram.tan + features.colorHistogram.darkBrown,
        white: features.colorHistogram.white
    });
    
    // Adjust breed confidence based on advanced features
    let adjustedConfidence = breedResult.confidence;
    
    // Boost confidence for distinctive patterns using advanced features
    if (animalType === 'Cattle') {
        // Holstein pattern (black and white with high contrast)
        if (features.colorHistogram.white > 0.3 && features.colorHistogram.black > 0.2 && features.contrastAnalysis.high > 0.3) {
            adjustedConfidence = Math.min(95, adjustedConfidence + 15);
        }
        // Jersey pattern (brown/tan with smooth texture)
        else if ((features.colorHistogram.brown + features.colorHistogram.tan) > 0.5 && features.textureFeatures.smoothness < 25) {
            adjustedConfidence = Math.min(95, adjustedConfidence + 12);
        }
        // Sahiwal pattern (red/brown with medium contrast)
        else if (features.colorHistogram.red > 0.1 && features.colorHistogram.brown > 0.2) {
            adjustedConfidence = Math.min(95, adjustedConfidence + 10);
        }
    } else if (animalType === 'Buffalo') {
        // Murrah pattern (very dark with low contrast)
        if (features.colorHistogram.black > 0.4 && features.contrastAnalysis.low > 0.4 && features.brightnessDistribution.average < 70) {
            adjustedConfidence = Math.min(95, adjustedConfidence + 15);
        }
        // Surti pattern (gray with uniform texture)
        else if (features.colorHistogram.gray > 0.3 && features.textureFeatures.smoothness < 20) {
            adjustedConfidence = Math.min(95, adjustedConfidence + 12);
        }
        // Jaffarabadi pattern (dark gray with medium contrast)
        else if (features.colorHistogram.darkGray > 0.3 && features.contrastAnalysis.medium > 0.3) {
            adjustedConfidence = Math.min(95, adjustedConfidence + 10);
        }
    }
    
    return {
        breed: breedResult.breed,
        breedConfidence: Math.max(adjustedConfidence, 35)
    };
}

function populateBreedDatabase() {
    if (!buffaloBreeds || !cattleBreeds) return;
    
    // Populate buffalo breeds
    const buffaloBreedsList = Object.values(BREED_DATABASE.buffalo);
    buffaloBreeds.innerHTML = buffaloBreedsList.map(breed => 
        `<div class="breed-item" onclick="showBreedDetails('${breed.name}', 'buffalo')">
            <strong>${breed.name}</strong> - ${breed.origin}
            <br><small>${breed.characteristics.color} • ${breed.characteristics.size} • ${breed.milkProduction}</small>
        </div>`
    ).join('');
    
    // Populate cattle breeds
    const cattleBreedsList = Object.values(BREED_DATABASE.cattle);
    cattleBreeds.innerHTML = cattleBreedsList.map(breed => 
        `<div class="breed-item" onclick="showBreedDetails('${breed.name}', 'cattle')">
            <strong>${breed.name}</strong> - ${breed.origin}
            <br><small>${breed.characteristics.color} • ${breed.characteristics.size} • ${breed.milkProduction}</small>
        </div>`
    ).join('');
}

function showBreedDetails(breedName, type) {
    const breedData = getBreedInformation(breedName, type);
    if (!breedData) return;
    
    const detailsHTML = `
        <div class="breed-detail-card">
            <h5>${breedData.name}</h5>
            <div class="breed-detail-grid">
                <div class="detail-item">
                    <strong>Origin:</strong> ${breedData.origin}
                </div>
                <div class="detail-item">
                    <strong>Color:</strong> ${breedData.characteristics.color}
                </div>
                <div class="detail-item">
                    <strong>Body Type:</strong> ${breedData.characteristics.bodyType}
                </div>
                <div class="detail-item">
                    <strong>Horns:</strong> ${breedData.characteristics.horns}
                </div>
                <div class="detail-item">
                    <strong>Size:</strong> ${breedData.characteristics.size}
                </div>
                <div class="detail-item">
                    <strong>Weight:</strong> ${breedData.characteristics.weight}
                </div>
                <div class="detail-item">
                    <strong>Milk Production:</strong> ${breedData.milkProduction}
                </div>
                <div class="detail-item">
                    <strong>Temperament:</strong> ${breedData.temperament}
                </div>
                <div class="detail-item">
                    <strong>Distribution:</strong> ${breedData.distribution}
                </div>
                ${breedData.temperatureTolerance ? `
                <div class="detail-item">
                    <strong>Temperature Tolerance:</strong> ${breedData.temperatureTolerance}
                </div>` : ''}
                ${breedData.climateAdaptation ? `
                <div class="detail-item">
                    <strong>Climate Adaptation:</strong> ${breedData.climateAdaptation}
                </div>` : ''}
                ${breedData.feedEfficiency ? `
                <div class="detail-item">
                    <strong>Feed Efficiency:</strong> ${breedData.feedEfficiency}
                </div>` : ''}
                ${breedData.diseaseResistance ? `
                <div class="detail-item">
                    <strong>Disease Resistance:</strong> ${breedData.diseaseResistance}
                </div>` : ''}
            </div>
        </div>
    `;
    
    breedDetails.innerHTML = detailsHTML;
    breedInfo.style.display = 'block';
    breedInfo.scrollIntoView({ behavior: 'smooth' });
}

function displayResults(result) {
    predictedType.textContent = result.type;
    confidenceScore.textContent = `${result.confidence}%`;
    if (predictedBreed) {
        predictedBreed.textContent = result.breed || '-';
    }
    if (breedConfidenceEl) {
        breedConfidenceEl.textContent = result.breedConfidence ? `${result.breedConfidence}%` : '-';
    }
    
    setTimeout(() => {
        confidenceFill.style.width = `${result.confidence}%`;
    }, 100);
    
    // Professional color coding based on confidence levels
    const valueElements = document.querySelectorAll('.value');
    valueElements.forEach(element => {
        if (element === predictedType) {
            if (result.confidence > 85) {
                element.style.color = '#28a745'; // High confidence - Green
            } else if (result.confidence > 70) {
                element.style.color = '#20c997'; // Good confidence - Teal
            } else if (result.confidence > 55) {
                element.style.color = '#ffc107'; // Moderate confidence - Yellow
            } else {
                element.style.color = '#dc3545'; // Low confidence - Red
            }
        }
        if (element === predictedBreed) {
            const bc = result.breedConfidence || 0;
            if (bc > 80) {
                element.style.color = '#28a745';
            } else if (bc > 60) {
                element.style.color = '#20c997';
            } else if (bc > 40) {
                element.style.color = '#ffc107';
            } else {
                element.style.color = '#6c757d';
            }
        }
    });
    
    // Show breed information if available and confident
    if (result.breed && result.breed !== 'Unknown' && result.breedConfidence > 40) {
        showBreedDetailsFromResult(result.breed, result.type);
    } else {
        if (breedInfo) breedInfo.style.display = 'none';
    }
    
    // Add quality indicators to debug info
    showDebugInfo(result);
}

function showBreedDetailsFromResult(breedName, type) {
    const breedData = getBreedInformation(breedName, type);
    if (!breedData) return;
    
    const detailsHTML = `
        <div class="breed-detail-card">
            <h5>${breedData.name} - Detected Breed</h5>
            <div class="breed-detail-grid">
                <div class="detail-item">
                    <strong>Origin:</strong> ${breedData.origin}
                </div>
                <div class="detail-item">
                    <strong>Color:</strong> ${breedData.characteristics.color}
                </div>
                <div class="detail-item">
                    <strong>Body Type:</strong> ${breedData.characteristics.bodyType}
                </div>
                <div class="detail-item">
                    <strong>Horns:</strong> ${breedData.characteristics.horns}
                </div>
                <div class="detail-item">
                    <strong>Size:</strong> ${breedData.characteristics.size}
                </div>
                <div class="detail-item">
                    <strong>Weight:</strong> ${breedData.characteristics.weight}
                </div>
                <div class="detail-item">
                    <strong>Milk Production:</strong> ${breedData.milkProduction}
                </div>
                <div class="detail-item">
                    <strong>Temperament:</strong> ${breedData.temperament}
                </div>
                <div class="detail-item">
                    <strong>Distribution:</strong> ${breedData.distribution}
                </div>
            </div>
        </div>
    `;
    
    if (breedDetails) {
        breedDetails.innerHTML = detailsHTML;
        breedInfo.style.display = 'block';
    }
}

function showDebugInfo(result) {
    debugInfo.style.display = 'block';
    
    // Professional analysis details
    const debugData = {
        'Analysis Method': result.method || 'Rule-based Classification',
        'Analysis Result': result.type,
        'Confidence Level': `${result.confidence}%`,
        'Quality Assessment': result.quality ? result.quality.imageQuality : 'Unknown',
        'Predicted Breed': result.breed || '-',
        'Breed Confidence': result.breedConfidence ? `${result.breedConfidence}%` : '-',
        'Breed Accuracy': result.quality ? result.quality.breedAccuracy : 'Unknown'
    };
    
    // Advanced color analysis details
    if (result.analysis && result.analysis.colorHistogram) {
        const colorData = {
            'Black/Dark Gray': `${((result.analysis.colorHistogram.black + result.analysis.colorHistogram.darkGray) * 100).toFixed(1)}%`,
            'Brown/Tan': `${((result.analysis.colorHistogram.brown + result.analysis.colorHistogram.lightBrown + result.analysis.colorHistogram.tan + result.analysis.colorHistogram.darkBrown) * 100).toFixed(1)}%`,
            'White': `${(result.analysis.colorHistogram.white * 100).toFixed(1)}%`,
            'Gray': `${((result.analysis.colorHistogram.gray + result.analysis.colorHistogram.lightGray) * 100).toFixed(1)}%`,
            'Red/Yellow': `${((result.analysis.colorHistogram.red + result.analysis.colorHistogram.yellow) * 100).toFixed(1)}%`
        };
        
        Object.assign(debugData, colorData);
    }
    
    // Advanced brightness and contrast analysis
    if (result.analysis && result.analysis.brightnessDistribution) {
        const brightnessData = {
            'Average Brightness': `${result.analysis.brightnessDistribution.average.toFixed(1)}`,
            'Dark Areas': `${(result.analysis.brightnessDistribution.dark * 100).toFixed(1)}%`,
            'Medium Brightness': `${(result.analysis.brightnessDistribution.medium * 100).toFixed(1)}%`,
            'Light Areas': `${(result.analysis.brightnessDistribution.light * 100).toFixed(1)}%`,
            'Brightness Variance': `${result.analysis.brightnessDistribution.variance.toFixed(1)}`
        };
        
        Object.assign(debugData, brightnessData);
    }
    
    // Texture analysis
    if (result.analysis && result.analysis.textureFeatures) {
        const textureData = {
            'Texture Score': `${result.analysis.textureFeatures.textureScore.toFixed(2)}`,
            'Smoothness': `${result.analysis.textureFeatures.smoothness.toFixed(1)}`,
            'Local Contrast': `${result.analysis.textureFeatures.contrast.toFixed(1)}`
        };
        
        Object.assign(debugData, textureData);
    }
    
    // Edge analysis
    if (result.analysis && result.analysis.edgeAnalysis) {
        const edgeData = {
            'Edge Ratio': `${(result.analysis.edgeAnalysis.edgeRatio * 100).toFixed(1)}%`,
            'Strong Edges': `${(result.analysis.edgeAnalysis.strongEdgeRatio * 100).toFixed(1)}%`,
            'Edge Density': `${(result.analysis.edgeAnalysis.edgeDensity * 100).toFixed(2)}%`
        };
        
        Object.assign(debugData, edgeData);
    }
    
    // Contrast analysis
    if (result.analysis && result.analysis.contrastAnalysis) {
        const contrastData = {
            'Low Contrast': `${(result.analysis.contrastAnalysis.low * 100).toFixed(1)}%`,
            'Medium Contrast': `${(result.analysis.contrastAnalysis.medium * 100).toFixed(1)}%`,
            'High Contrast': `${(result.analysis.contrastAnalysis.high * 100).toFixed(1)}%`,
            'Average Contrast': `${result.analysis.contrastAnalysis.average.toFixed(1)}`
        };
        
        Object.assign(debugData, contrastData);
    }
    
    // Classification scores
    if (result.scores) {
        const scoreData = {
            'Cattle Score': `${result.scores.cattle.toFixed(2)}`,
            'Buffalo Score': `${result.scores.buffalo.toFixed(2)}`
        };
        
        Object.assign(debugData, scoreData);
    }
    
    let debugHTML = '<div class="debug-section">';
    debugHTML += '<h5>Professional Analysis Details</h5>';
    
    for (const [key, value] of Object.entries(debugData)) {
        debugHTML += `<div class="debug-item"><strong>${key}:</strong> ${value}</div>`;
    }
    
    debugHTML += '</div>';
    
    // Add recommendations
    if (result.confidence < 70) {
        debugHTML += '<div class="debug-section">';
        debugHTML += '<h5>Recommendations</h5>';
        debugHTML += '<div class="debug-item">• Try uploading a clearer, higher resolution image</div>';
        debugHTML += '<div class="debug-item">• Ensure good lighting and minimal background interference</div>';
        debugHTML += '<div class="debug-item">• Make sure the animal is clearly visible in the image</div>';
        debugHTML += '</div>';
    }
    
    debugContent.innerHTML = debugHTML;
}

function setLoadingState(isLoading) {
    if (isLoading) {
        classifyBtn.disabled = true;
        btnText.style.display = 'none';
        loadingSpinner.style.display = 'block';
    } else {
        classifyBtn.disabled = false;
        btnText.style.display = 'block';
        loadingSpinner.style.display = 'none';
    }
}

function showStatusMessage(message, type = 'info') {
    const existingStatus = document.querySelector('.status-message');
    if (existingStatus) {
        existingStatus.remove();
    }
    
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-message status-${type}`;
    statusDiv.textContent = message;
    
    statusDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
        max-width: 300px;
        word-wrap: break-word;
    `;
    
    switch(type) {
        case 'success':
            statusDiv.style.backgroundColor = '#28a745';
            break;
        case 'error':
            statusDiv.style.backgroundColor = '#dc3545';
            break;
        case 'warning':
            statusDiv.style.backgroundColor = '#ffc107';
            statusDiv.style.color = '#212529';
            break;
        case 'info':
            statusDiv.style.backgroundColor = '#17a2b8';
            break;
        default:
            statusDiv.style.backgroundColor = '#6c757d';
    }
    
    document.body.appendChild(statusDiv);
    
    setTimeout(() => {
        if (statusDiv.parentNode) {
            statusDiv.remove();
        }
    }, 5000);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);
