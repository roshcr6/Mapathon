"""
AI Training Module with PARALLEL PROCESSING
Uses all CPU cores for maximum training power.

Trains multiple classifier types simultaneously and picks the best one.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import logging
import time
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features_from_contour(contour: np.ndarray) -> np.ndarray:
    """Extract shape features from a contour (standalone function for parallel processing)."""
    features = []
    
    # Basic shape properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0 or area == 0:
        return np.zeros(9)
    
    # Aspect ratio
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if min(width, height) > 0:
        aspect_ratio = max(width, height) / min(width, height)
    else:
        aspect_ratio = 1.0
    features.append(min(aspect_ratio, 50))
    
    # Circularity
    circularity = 4 * np.pi * area / (perimeter ** 2)
    features.append(min(circularity, 1.0))
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    features.append(solidity)
    
    # Extent
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    features.append(extent)
    
    # Orientation
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            orientation = ellipse[2] / 180.0
        except:
            orientation = 0.5
    else:
        orientation = 0.5
    features.append(orientation)
    
    # Hu Moments
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(4):
        val = hu_moments[i]
        if val != 0:
            log_val = -np.sign(val) * np.log10(abs(val) + 1e-10)
        else:
            log_val = 0
        features.append(max(-20, min(20, log_val)))
    
    return np.array(features, dtype=np.float64)


def process_single_image(args: Tuple[Path, str]) -> List[Tuple[np.ndarray, str]]:
    """Process a single image and extract features (for parallel execution)."""
    image_path, label = args
    results = []
    
    img = cv2.imread(str(image_path))
    if img is None:
        return results
    
    # More augmentations for better training
    augmentations = [
        img,
        cv2.flip(img, 0),
        cv2.flip(img, 1),
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(cv2.flip(img, 0), 1),  # Both flips
    ]
    
    # Add brightness variations
    for gamma in [0.8, 1.2]:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        augmentations.append(cv2.LUT(img, table))
    
    for aug_img in augmentations:
        gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
        
        # Multiple thresholds
        for thresh in [170, 180, 190, 200, 210, 220, 230]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 30:
                    continue
                
                features = extract_features_from_contour(contour)
                if np.all(np.isfinite(features)):
                    results.append((features, label))
    
    return results


def train_classifier(args: Tuple[str, Any, np.ndarray, np.ndarray]) -> Tuple[str, float, Any]:
    """Train a single classifier and return its cross-validation score."""
    name, clf, X, y = args
    try:
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=1)
        mean_score = scores.mean()
        
        # Fit on full data
        clf.fit(X, y)
        
        return (name, mean_score, clf)
    except Exception as e:
        logger.warning(f"Classifier {name} failed: {e}")
        return (name, 0.0, None)


class ParallelPavementClassifier:
    """
    High-performance classifier using parallel processing.
    Trains multiple models and selects the best one.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.best_model_name = None
        self.feature_names = [
            'aspect_ratio', 'circularity', 'solidity',
            'extent', 'orientation', 'hu_moments_0',
            'hu_moments_1', 'hu_moments_2', 'hu_moments_3'
        ]
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train_parallel(self, training_data_dir: Path):
        """
        Train using parallel processing on all CPU cores.
        """
        start_time = time.time()
        n_cores = mp.cpu_count()
        logger.info(f"🚀 Starting PARALLEL AI training with {n_cores} CPU cores...")
        
        # Collect all image paths
        image_tasks = []
        
        lines_dir = training_data_dir / "lines"
        if lines_dir.exists():
            for img_path in list(lines_dir.glob("*.png")) + list(lines_dir.glob("*.jpg")):
                image_tasks.append((img_path, "lane_line"))
            logger.info(f"Found {len([t for t in image_tasks if t[1] == 'lane_line'])} line images")
        
        crosswalk_dir = training_data_dir / "crosswalk"
        if crosswalk_dir.exists():
            for img_path in list(crosswalk_dir.glob("*.png")) + list(crosswalk_dir.glob("*.jpg")):
                image_tasks.append((img_path, "crosswalk"))
            logger.info(f"Found {len([t for t in image_tasks if t[1] == 'crosswalk'])} crosswalk images")
        
        if not image_tasks:
            raise ValueError("No training data found!")
        
        # Step 1: Parallel feature extraction
        logger.info("⚡ Extracting features in parallel...")
        X_train = []
        y_train = []
        
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(process_single_image, task) for task in image_tasks]
            
            for future in as_completed(futures):
                for features, label in future.result():
                    X_train.append(features)
                    y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        logger.info(f"📊 Total training samples: {len(X_train)}")
        logger.info(f"   - lane_line: {np.sum(y_train == 'lane_line')}")
        logger.info(f"   - crosswalk: {np.sum(y_train == 'crosswalk')}")
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Step 2: Train multiple classifiers in parallel
        logger.info("🧠 Training multiple AI models in parallel...")
        
        classifiers = [
            ("RandomForest-500", RandomForestClassifier(
                n_estimators=500, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, class_weight='balanced', n_jobs=-1, random_state=42
            )),
            ("RandomForest-1000", RandomForestClassifier(
                n_estimators=1000, max_depth=25, min_samples_split=2,
                min_samples_leaf=1, class_weight='balanced', n_jobs=-1, random_state=42
            )),
            ("GradientBoosting", GradientBoostingClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42
            )),
            ("AdaBoost", AdaBoostClassifier(
                n_estimators=200, learning_rate=0.5, random_state=42
            )),
            ("MLP-Deep", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64), max_iter=1000,
                early_stopping=True, random_state=42
            )),
            ("MLP-Wide", MLPClassifier(
                hidden_layer_sizes=(512, 256), max_iter=1000,
                early_stopping=True, random_state=42
            )),
        ]
        
        # Train all classifiers in parallel
        results = []
        with ThreadPoolExecutor(max_workers=len(classifiers)) as executor:
            tasks = [(name, clf, X_scaled, y_train) for name, clf in classifiers]
            futures = [executor.submit(train_classifier, task) for task in tasks]
            
            for future in as_completed(futures):
                name, score, clf = future.result()
                if clf is not None:
                    results.append((name, score, clf))
                    logger.info(f"   {name}: {score:.4f} accuracy")
        
        # Select best model
        if results:
            best = max(results, key=lambda x: x[1])
            self.best_model_name = best[0]
            self.model = best[2]
            
            logger.info(f"\n🏆 Best model: {self.best_model_name} with {best[1]:.4f} accuracy")
        else:
            # Fallback
            logger.info("Using fallback RandomForest...")
            self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1)
            self.model.fit(X_scaled, y_train)
            self.best_model_name = "RandomForest-fallback"
        
        # Training statistics
        train_time = time.time() - start_time
        logger.info(f"\n⏱️ Total training time: {train_time:.2f}s")
        logger.info(f"📈 Training samples per second: {len(X_train) / train_time:.0f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            logger.info("\n📊 Feature importance:")
            for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"   {name}: {imp:.4f}")
        
        logger.info("\n✅ PARALLEL AI training complete!")
        return self.model, self.scaler
    
    def predict(self, contour: np.ndarray, image: np.ndarray = None) -> str:
        """Predict marking type."""
        if self.model is None or self.scaler is None:
            return self._rule_based_classify(contour)
        
        features = extract_features_from_contour(contour)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(features_scaled)[0]
    
    def _rule_based_classify(self, contour: np.ndarray) -> str:
        """Fallback rule-based classification."""
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width == 0 or height == 0:
            return "lane_line"
        aspect_ratio = max(width, height) / min(width, height)
        return "lane_line" if aspect_ratio > 8 else "crosswalk"
    
    def save_model(self, model_path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.best_model_name
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data.get('model_type', 'unknown')
        
        logger.info(f"📂 Model loaded: {self.best_model_name}")


def train_ai_parallel(training_data_dir: str, output_model_path: str):
    """
    Train AI with full parallel processing power.
    """
    classifier = ParallelPavementClassifier()
    classifier.train_parallel(Path(training_data_dir))
    classifier.save_model(output_model_path)
    return classifier


if __name__ == "__main__":
    # Test parallel training
    import sys
    training_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    output_path = sys.argv[2] if len(sys.argv) > 2 else "trained_model_parallel.pkl"
    
    train_ai_parallel(training_dir, output_path)
