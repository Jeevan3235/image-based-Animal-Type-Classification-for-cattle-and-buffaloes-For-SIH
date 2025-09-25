import tensorflow as tf
import numpy as np
import cv2
import joblib
import yaml

class CattleBuffaloClassifier:
    def __init__(self, model_path=None):
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = self._build_model()
        
        self.class_names = ['cattle', 'buffalo']
        self.img_size = tuple(self.config['dataset']['image_size'])
    
    def _build_model(self):
        """Build CNN model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', 
                                 input_shape=(*self.img_size, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    
    def predict(self, image_path):
        """Make prediction on single image"""
        try:
            processed_img = self.preprocess_image(image_path)
            predictions = self.model.predict(processed_img)
            confidence = np.max(predictions[0])
            class_idx = np.argmax(predictions[0])
            
            return {
                'class': self.class_names[class_idx],
                'confidence': float(confidence),
                'all_predictions': predictions[0].tolist()
            }
        except Exception as e:
            return {'error': str(e)}
