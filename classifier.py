#!/usr/bin/env python3
"""
Advanced Breed Classification System
Uses trained model to identify specific breeds from features
"""

import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
from pathlib import Path

class BreedClassifier:
    def __init__(self, model_path="cattle_buffalo_model.joblib", metadata_path="cattle_buffalo_model_metadata.json"):
        self.model = None
        self.metadata = None
        self.breed_classifier = None
        # Optional specialized breed models
        self.buffalo_breed_model = None
        self.buffalo_breed_classes = None
        
        self.load_model_and_metadata(model_path, metadata_path)
        self.setup_breed_classifier()
        # Try to load specialized breed models (optional)
        self.load_buffalo_breed_model()
    
    def load_model_and_metadata(self, model_path, metadata_path):
        """Load the main model and metadata"""
        try:
            self.model = joblib.load(model_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print("Model and metadata loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_buffalo_breed_model(self):
        """Load optional specialized buffalo breed model if available"""
        try:
            model_path = Path("models") / "buffalo_breed_model.joblib"
            meta_path = Path("models") / "buffalo_breed_model_metadata.json"
            if model_path.exists() and meta_path.exists():
                self.buffalo_breed_model = joblib.load(str(model_path))
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                # Use model.classes_ to align probabilities with class order
                try:
                    self.buffalo_breed_classes = list(self.buffalo_breed_model.classes_)
                except Exception:
                    # Fallback to metadata if classes_ missing
                    self.buffalo_breed_classes = list(meta.get("breeds", []))
                print(f"Buffalo breed model loaded with {len(self.buffalo_breed_classes)} breeds")
        except Exception as e:
            print(f"Could not load buffalo breed model: {e}")
    
    def setup_breed_classifier(self):
        """Setup breed-specific classification logic"""
        # Define breed characteristics based on color and texture patterns
        self.breed_characteristics = {
            # Cattle breeds
            'Holstein': {
                'color_pattern': 'black_white_spotted',
                'primary_colors': ['black', 'white'],
                'texture': 'smooth',
                'contrast': 'high',
                'pattern_weight': 0.8
            },
            'Ayrshire': {
                'color_pattern': 'red_white_spotted',
                'primary_colors': ['red', 'white'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.7
            },
            'Brown Swiss': {
                'color_pattern': 'solid_brown',
                'primary_colors': ['brown'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Jersey': {
                'color_pattern': 'light_brown',
                'primary_colors': ['brown', 'tan'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Red Dane': {
                'color_pattern': 'solid_red',
                'primary_colors': ['red'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.7
            },
            'Holstein Friesian': {
                'color_pattern': 'black_white_spotted',
                'primary_colors': ['black', 'white'],
                'texture': 'smooth',
                'contrast': 'high',
                'pattern_weight': 0.8
            },
            'Brahman cow': {
                'color_pattern': 'light_gray',
                'primary_colors': ['gray', 'white'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.5
            },
            'Charolais': {
                'color_pattern': 'solid_white',
                'primary_colors': ['white'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Limousin': {
                'color_pattern': 'golden_red',
                'primary_colors': ['red', 'gold'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.7
            },
            'Zebu': {
                'color_pattern': 'varied',
                'primary_colors': ['brown', 'gray', 'white'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.4
            },
            
            # Buffalo breeds
            'Water Buffalo': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['black', 'dark_gray'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.7
            },
            'African Buffalo': {
                'color_pattern': 'dark_brown_black',
                'primary_colors': ['black', 'dark_brown'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.7
            },
            'White bufflo': {
                'color_pattern': 'white_light_gray',
                'primary_colors': ['white', 'light_gray'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Asian buffalo': {
                'color_pattern': 'dark_gray',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Wild bufflo': {
                'color_pattern': 'dark_brown',
                'primary_colors': ['dark_brown', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            # New Indian Buffalo Breeds
            'Bhadawari': {
                'color_pattern': 'copper_brown',
                'primary_colors': ['brown', 'copper'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.7
            },
            'Banni': {
                'color_pattern': 'black_dark_gray',
                'primary_colors': ['black', 'dark_gray'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.7
            },
            'Chilika': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Godawari': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Jaffarabadi': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.7
            },
            'Jerangi': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Kalahandi': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Kujang': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Manda': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Marathwada': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Mehsana': {
                'color_pattern': 'black_white_marked',
                'primary_colors': ['black', 'white'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.7
            },
            'Murrah': {
                'color_pattern': 'jet_black',
                'primary_colors': ['black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.8
            },
            'Nagpuri': {
                'color_pattern': 'black_dark_gray',
                'primary_colors': ['black', 'dark_gray'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.7
            },
            'Nili Ravi': {
                'color_pattern': 'black_white_marked',
                'primary_colors': ['black', 'white'],
                'texture': 'smooth',
                'contrast': 'medium',
                'pattern_weight': 0.7
            },
            'Pandharpuri': {
                'color_pattern': 'black_dark_gray',
                'primary_colors': ['black', 'dark_gray'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Paralakhemundi': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Sambalpuri': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'South Kanara': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Surti': {
                'color_pattern': 'light_gray_dark_gray',
                'primary_colors': ['light_gray', 'dark_gray'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Swamp Buffalo': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.7
            },
            'Tarai': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            },
            'Toda': {
                'color_pattern': 'dark_gray_black',
                'primary_colors': ['dark_gray', 'black'],
                'texture': 'smooth',
                'contrast': 'low',
                'pattern_weight': 0.6
            }
        }
    
    def extract_breed_features(self, image):
        """Extract features specifically for breed identification"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for consistent analysis
            rgb_image = cv2.resize(rgb_image, (224, 224))
            
            features = {}
            
            # Color analysis
            features['color_histogram'] = self.analyze_color_distribution(rgb_image)
            features['dominant_colors'] = self.get_dominant_colors(rgb_image)
            features['color_variance'] = self.calculate_color_variance(rgb_image)
            
            # Pattern analysis
            features['pattern_analysis'] = self.analyze_patterns(rgb_image)
            features['contrast_analysis'] = self.analyze_contrast(rgb_image)
            
            # Texture analysis
            features['texture_features'] = self.analyze_texture(rgb_image)
            
            return features
            
        except Exception as e:
            print(f"Error extracting breed features: {e}")
            return None
    
    def analyze_color_distribution(self, image):
        """Analyze color distribution in the image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges
        color_ranges = {
            'black': (0, 0, 0, 180, 255, 30),
            'white': (0, 0, 200, 180, 30, 255),
            'gray': (0, 0, 30, 180, 30, 200),
            'brown': (10, 50, 20, 20, 255, 200),
            'red': (0, 50, 50, 10, 255, 255),
            'tan': (20, 30, 100, 30, 255, 200),
            'dark_brown': (10, 50, 0, 20, 255, 100),
            'light_gray': (0, 0, 100, 180, 30, 200),
            'dark_gray': (0, 0, 30, 180, 30, 100),
            'gold': (20, 100, 100, 30, 255, 255),
            'copper': (15, 80, 40, 25, 255, 150)
        }
        
        color_percentages = {}
        total_pixels = image.shape[0] * image.shape[1]
        
        for color_name, (h_min, s_min, v_min, h_max, s_max, v_max) in color_ranges.items():
            mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
            color_percentages[color_name] = np.sum(mask > 0) / total_pixels
        
        return color_percentages
    
    def get_dominant_colors(self, image):
        """Get the most dominant colors in the image"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        
        # Reduce image size for faster processing
        small_image = cv2.resize(image, (50, 50))
        pixels = small_image.reshape(-1, 3)
        
        # Find 5 dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get color centers
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Count occurrences
        label_counts = np.bincount(labels)
        dominant_colors = []
        
        for i, count in enumerate(label_counts):
            if count > 0:
                dominant_colors.append({
                    'color': colors[i].astype(int).tolist(),
                    'percentage': count / len(labels)
                })
        
        return sorted(dominant_colors, key=lambda x: x['percentage'], reverse=True)
    
    def calculate_color_variance(self, image):
        """Calculate color variance across the image"""
        # Convert to grayscale for variance calculation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.var(gray)
    
    def analyze_patterns(self, image):
        """Analyze patterns in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate pattern metrics
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Detect contours for pattern analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pattern_metrics = {
            'edge_density': edge_density,
            'contour_count': len(contours),
            'has_spots': edge_density > 0.1,  # High edge density might indicate spots
            'is_solid': edge_density < 0.05   # Low edge density might indicate solid color
        }
        
        return pattern_metrics
    
    def analyze_contrast(self, image):
        """Analyze contrast in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate contrast metrics
        contrast_metrics = {
            'std_dev': np.std(gray),
            'range': np.max(gray) - np.min(gray),
            'high_contrast': np.std(gray) > 50,
            'low_contrast': np.std(gray) < 20
        }
        
        return contrast_metrics
    
    def analyze_texture(self, image):
        """Analyze texture features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture metrics
        texture_metrics = {
            'smoothness': np.var(cv2.Laplacian(gray, cv2.CV_64F)),
            'uniformity': np.mean(gray),
            'is_smooth': np.var(cv2.Laplacian(gray, cv2.CV_64F)) < 100
        }
        
        return texture_metrics
    
    def classify_breed(self, image, animal_type):
        """Classify the specific breed based on image features"""
        if animal_type not in ['Cattle', 'Buffalo']:
            return {'breed': 'Unknown', 'confidence': 0.0}
        
        # Extract breed features
        features = self.extract_breed_features(image)
        if features is None:
            return {'breed': 'Unknown', 'confidence': 0.0}
        
        # If buffalo and a trained buffalo breed model is available, use it first
        if animal_type == 'Buffalo' and self.buffalo_breed_model is not None:
            try:
                # Build feature vector identical to training pipeline
                vec = self._extract_training_style_features(image)
                if vec is not None:
                    probs = self.buffalo_breed_model.predict_proba([vec])[0]
                    idx = int(np.argmax(probs))
                    breed = self.buffalo_breed_classes[idx] if self.buffalo_breed_classes else 'Unknown'
                    return {
                        'breed': breed,
                        'confidence': float(probs[idx])
                    }
            except Exception as e:
                print(f"Buffalo breed model inference failed, falling back to heuristics: {e}")
        
        # Heuristic fallback using color/texture rules
        # Get available breeds for this animal type
        available_breeds = list(self.breed_characteristics.keys())
        if animal_type == 'Cattle':
            available_breeds = [b for b in available_breeds if b not in ['Water Buffalo', 'African Buffalo', 'White bufflo', 'Asian buffalo', 'Wild bufflo', 'Bhadawari', 'Banni', 'Chilika', 'Godawari', 'Jaffarabadi', 'Jerangi', 'Kalahandi', 'Kujang', 'Manda', 'Marathwada', 'Mehsana', 'Murrah', 'Nagpuri', 'Nili Ravi', 'Pandharpuri', 'Paralakhemundi', 'Sambalpuri', 'South Kanara', 'Surti', 'Swamp Buffalo', 'Tarai', 'Toda']]
        else:  # Buffalo
            available_breeds = [b for b in available_breeds if b not in ['Holstein', 'Ayrshire', 'Brown Swiss', 'Jersey', 'Red Dane', 'Holstein Friesian', 'Brahman cow', 'Charolais', 'Limousin', 'Zebu']]
        
        best_breed = 'Unknown'
        best_confidence = 0.0
        for breed in available_breeds:
            if breed in self.breed_characteristics:
                confidence = self.calculate_breed_confidence(breed, features)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_breed = breed
        
        return {
            'breed': best_breed,
            'confidence': best_confidence,
            'features_used': features
        }
    
    def _extract_training_style_features(self, image):
        """Recompute the exact numeric feature vector used in training (train_buffalo_model.extract_features)."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            features = []
            # 1. RGB histograms (32 bins each)
            for i in range(3):
                hist = cv2.calcHist([rgb_image], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            # 2. HSV histograms (32 bins each)
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            # 3. LBP-like and histogram (16 bins)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            lbp = self._compute_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [16], [0, 16])
            features.extend(lbp_hist.flatten())
            # 4. Edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            # 5. Brightness/contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            features.extend([brightness, contrast])
            # 6. Shape
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                area = 0
                circularity = 0
            features.extend([area, circularity])
            # 7. Color moments
            for channel in range(3):
                channel_data = rgb_image[:, :, channel].flatten()
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                skewness = self._compute_skewness(channel_data)
                features.extend([mean, std, skewness])

            return np.array(features, dtype=float)
        except Exception:
            return None

    def _compute_lbp(self, image, radius=1, n_points=8):
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
                        binary_string += '1' if image[x, y] >= center else '0'
                    else:
                        binary_string += '0'
                lbp[i, j] = int(binary_string, 2)
        return lbp

    def _compute_skewness(self, data):
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def calculate_breed_confidence(self, breed, features):
        """Calculate confidence score for a specific breed"""
        if breed not in self.breed_characteristics:
            return 0.0
        
        char = self.breed_characteristics[breed]
        confidence = 0.0
        
        # Color matching
        color_hist = features['color_histogram']
        pattern_analysis = features['pattern_analysis']
        contrast_analysis = features['contrast_analysis']
        
        # Check color patterns
        if 'black_white_spotted' in char['color_pattern']:
            if color_hist.get('black', 0) > 0.1 and color_hist.get('white', 0) > 0.1:
                confidence += 0.4
            if pattern_analysis.get('has_spots', False):
                confidence += 0.3
        
        elif 'black_white_marked' in char['color_pattern']:
            if color_hist.get('black', 0) > 0.1 and color_hist.get('white', 0) > 0.1:
                confidence += 0.4
            if pattern_analysis.get('has_spots', False):
                confidence += 0.3
        
        elif 'jet_black' in char['color_pattern']:
            if color_hist.get('black', 0) > 0.7:
                confidence += 0.5
            if pattern_analysis.get('is_solid', False):
                confidence += 0.3
        
        elif 'copper_brown' in char['color_pattern']:
            if color_hist.get('brown', 0) > 0.3 or color_hist.get('copper', 0) > 0.3:
                confidence += 0.4
            if pattern_analysis.get('is_solid', False):
                confidence += 0.3
        
        elif 'solid' in char['color_pattern']:
            primary_color = char['primary_colors'][0]
            if color_hist.get(primary_color, 0) > 0.5:
                confidence += 0.4
            if pattern_analysis.get('is_solid', False):
                confidence += 0.3
        
        elif 'dark' in char['color_pattern']:
            if color_hist.get('black', 0) > 0.3 or color_hist.get('dark_gray', 0) > 0.3:
                confidence += 0.4
        
        elif 'light_gray_dark_gray' in char['color_pattern']:
            if color_hist.get('light_gray', 0) > 0.2 or color_hist.get('dark_gray', 0) > 0.2:
                confidence += 0.4
        
        # Check contrast
        if char['contrast'] == 'high' and contrast_analysis.get('high_contrast', False):
            confidence += 0.2
        elif char['contrast'] == 'low' and contrast_analysis.get('low_contrast', False):
            confidence += 0.2
        
        # Check texture
        texture_features = features['texture_features']
        if char['texture'] == 'smooth' and texture_features.get('is_smooth', False):
            confidence += 0.1
        
        # Apply pattern weight
        confidence *= char['pattern_weight']
        
        return min(confidence, 1.0)

# Global instance
breed_classifier = BreedClassifier()
