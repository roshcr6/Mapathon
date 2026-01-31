"""
AI Training Module for Pavement Marking Classification

Trains a classifier to distinguish between:
- lane_line (from lines folder)
- crosswalk (from crosswalk folder)

Uses computer vision feature extraction and Random Forest classifier.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PavementMarkingClassifier:
    """
    Trained classifier for pavement marking types.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to saved model pickle file
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            'aspect_ratio', 'circularity', 'solidity', 
            'extent', 'orientation', 'hu_moments_0', 
            'hu_moments_1', 'hu_moments_2', 'hu_moments_3'
        ]
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def extract_features(self, contour: np.ndarray, image: np.ndarray = None) -> np.ndarray:
        """
        Extract shape features from a contour.
        
        Args:
            contour: Contour points
            image: Optional image for pixel-based features
            
        Returns:
            Feature vector
        """
        features = []
        
        # Basic shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0 or area == 0:
            return np.zeros(len(self.feature_names))
        
        # Aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if min(width, height) > 0:
            aspect_ratio = max(width, height) / min(width, height)
        else:
            aspect_ratio = 1.0
        features.append(min(aspect_ratio, 50))  # Cap at 50 to avoid outliers
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2)
        features.append(min(circularity, 1.0))
        
        # Solidity (area vs convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        features.append(solidity)
        
        # Extent (area vs bounding rectangle area)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        features.append(extent)
        
        # Orientation
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                orientation = ellipse[2] / 180.0  # Normalize to 0-1
            except:
                orientation = 0.5
        else:
            orientation = 0.5
        features.append(orientation)
        
        # Hu Moments (shape descriptors)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Use first 4 Hu moments, log-transformed
        for i in range(4):
            val = hu_moments[i]
            if val != 0:
                log_val = -np.sign(val) * np.log10(abs(val) + 1e-10)
            else:
                log_val = 0
            features.append(max(-20, min(20, log_val)))  # Clamp to avoid extremes
        
        return np.array(features, dtype=np.float64)
    
    def train(self, training_data_dir: Path):
        """
        Train classifier on labeled training data.
        
        Expects folder structure:
        - training_data_dir/
          - lines/  (images of lane lines)
          - crosswalk/  (images of crosswalks)
        
        Args:
            training_data_dir: Path to training data directory
        """
        logger.info("Starting AI training...")
        
        X_train = []
        y_train = []
        
        # Process lane lines
        lines_dir = training_data_dir / "lines"
        if lines_dir.exists():
            line_images = list(lines_dir.glob("*.png")) + list(lines_dir.glob("*.jpg"))
            logger.info(f"Found {len(line_images)} line training images")
            
            for img_path in line_images:
                features = self._extract_features_from_image(img_path)
                for feat in features:
                    X_train.append(feat)
                    y_train.append("lane_line")
        
        # Process crosswalks
        crosswalk_dir = training_data_dir / "crosswalk"
        if crosswalk_dir.exists():
            crosswalk_images = list(crosswalk_dir.glob("*.png")) + list(crosswalk_dir.glob("*.jpg"))
            logger.info(f"Found {len(crosswalk_images)} crosswalk training images")
            
            for img_path in crosswalk_images:
                features = self._extract_features_from_image(img_path)
                for feat in features:
                    X_train.append(feat)
                    y_train.append("crosswalk")
        
        if len(X_train) == 0:
            raise ValueError("No training data found!")
        
        logger.info(f"Total training samples: {len(X_train)}")
        logger.info(f"  - lane_line: {y_train.count('lane_line')}")
        logger.info(f"  - crosswalk: {y_train.count('crosswalk')}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Remove any invalid samples (NaN or Inf)
        valid_mask = np.all(np.isfinite(X_train), axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        logger.info(f"Valid samples after filtering: {len(X_train)}")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest classifier with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=200,       # More trees for better accuracy
            max_depth=15,           # Deeper trees
            min_samples_split=3,    # More sensitive splits
            min_samples_leaf=2,     # Smaller leaves allowed
            class_weight='balanced', # Handle class imbalance
            random_state=42,
            n_jobs=-1               # Use all CPU cores
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Print feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        logger.info("Feature importance:")
        for name, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {importance:.4f}")
        
        logger.info("✅ AI training complete!")
        
        return self.model, self.scaler
    
    def _extract_features_from_image(self, image_path: Path) -> List[np.ndarray]:
        """
        Extract features from all markings in an image with data augmentation.
        
        Args:
            image_path: Path to training image
            
        Returns:
            List of feature vectors
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return []
        
        features_list = []
        
        # Process original and augmented versions
        augmentations = [
            img,  # Original
            cv2.flip(img, 0),  # Vertical flip
            cv2.flip(img, 1),  # Horizontal flip
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),  # Rotate 90
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),  # Rotate -90
        ]
        
        for aug_img in augmentations:
            # Convert to grayscale
            gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
            
            # Try multiple thresholds for robustness
            for thresh in [180, 200, 220]:
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 50:  # Skip tiny noise
                        continue
                    
                    features = self.extract_features(contour, gray)
                    features_list.append(features)
        
        return features_list
    
    def predict(self, contour: np.ndarray, image: np.ndarray = None) -> str:
        """
        Predict marking type for a contour.
        
        Args:
            contour: Contour points
            image: Optional image
            
        Returns:
            Predicted class ("lane_line" or "crosswalk")
        """
        if self.model is None or self.scaler is None:
            # Fallback to rule-based
            return self._rule_based_classify(contour)
        
        features = self.extract_features(contour, image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
    
    def _rule_based_classify(self, contour: np.ndarray) -> str:
        """
        Fallback rule-based classification.
        
        Args:
            contour: Contour points
            
        Returns:
            Predicted class
        """
        # Calculate aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            return "lane_line"
        
        aspect_ratio = max(width, height) / min(width, height)
        
        # Lines are very elongated, crosswalks are more rectangular
        if aspect_ratio > 8:
            return "lane_line"
        else:
            # Check orientation for crosswalk bars
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.3 and 2 < aspect_ratio < 8:
                    return "crosswalk"
        
        return "lane_line"
    
    def save_model(self, model_path: str):
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train first!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {model_path}")


def train_ai_model(training_data_dir: str, output_model_path: str):
    """
    Train AI model on provided data.
    
    Args:
        training_data_dir: Directory with lines/ and crosswalk/ folders
        output_model_path: Where to save trained model
    """
    classifier = PavementMarkingClassifier()
    classifier.train(Path(training_data_dir))
    classifier.save_model(output_model_path)
    
    return classifier
