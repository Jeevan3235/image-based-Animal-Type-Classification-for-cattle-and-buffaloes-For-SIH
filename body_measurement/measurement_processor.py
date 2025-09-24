import cv2
import numpy as np
import math
from scipy import ndimage

class BodyMeasurementProcessor:
    def __init__(self):
        """
        Initialize body measurement processor
        """
        self.measurement_units = "pixels"  # Can be calibrated to real units
    
    def process_image(self, image_path):
        """
        Extract body measurements from animal image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Detect animal body
            body_contour = self.detect_animal_body(processed_image)
            if body_contour is None:
                return {"error": "Could not detect animal body"}
            
            # Extract measurements
            measurements = self.extract_measurements(body_contour, image.shape)
            
            return measurements
            
        except Exception as e:
            return {"error": f"Measurement extraction failed: {str(e)}"}
    
    def preprocess_image(self, image):
        """
        Preprocess image for better contour detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def detect_animal_body(self, processed_image):
        """
        Detect the main animal body contour
        """
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
            
            # Typical animal body proportions
            if area > 1000 and 0.3 < aspect_ratio < 3.0:
                valid_contours.append(contour)
        
        if not valid_contours:
            return None
        
        # Return the largest valid contour
        return max(valid_contours, key=cv2.contourArea)
    
    def extract_measurements(self, contour, image_shape):
        """
        Extract various body measurements from contour
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Basic measurements
        body_length = w
        height_at_withers = h
        
        # Calculate convex hull for more accurate measurements
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Estimate chest width (widest point)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = x + w//2, y + h//2
        
        # Calculate rump angle
        rump_angle = self.calculate_rump_angle(contour)
        
        # Estimate body volume (simplified)
        body_volume = hull_area * h  # Approximation
        
        measurements = {
            "body_length_pixels": int(body_length),
            "height_withers_pixels": int(height_at_withers),
            "chest_width_pixels": int(w * 0.6),  # Estimation
            "rump_angle_degrees": float(rump_angle),
            "body_area_pixels": int(hull_area),
            "estimated_volume": int(body_volume),
            "body_condition_score": self.calculate_body_condition(contour, hull_area),
            "contour_centroid": (int(cx), int(cy))
        }
        
        return measurements
    
    def calculate_rump_angle(self, contour):
        """
        Calculate rump angle from contour
        """
        try:
            # Fit ellipse to contour
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]  # Rotation angle
                return angle % 180  # Normalize to 0-180 degrees
        except:
            pass
        
        return 45.0  # Default value
    
    def calculate_body_condition(self, contour, area):
        """
        Simplified body condition score based on contour shape
        """
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.5
        
        # Map solidity to body condition score (1-5)
        bcs = 1 + (solidity - 0.5) * 8  # Rough mapping
        return max(1, min(5, bcs))  # Clamp between 1-5
    
    def calibrate_measurements(self, reference_length_cm, reference_length_pixels):
        """
        Calibrate pixel measurements to real-world units
        """
        self.pixels_to_cm = reference_length_cm / reference_length_pixels
        self.measurement_units = "cm"
