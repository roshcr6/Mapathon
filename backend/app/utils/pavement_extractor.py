"""
Pavement Marking Extraction Module

This module handles the AI-powered extraction of pavement markings from
satellite/aerial imagery exported from QGIS.

Pipeline:
1. Load georeferenced satellite image
2. Preprocess (contrast enhancement, noise reduction)
3. Detect high-brightness regions (pavement markings)
4. Apply morphological operations
5. Extract contours and classify markings
6. Convert to georeferenced vector data
7. Export as GeoJSON
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import linemerge
import json
import logging
import time

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Import AI classifier
try:
    from .ai_trainer import PavementMarkingClassifier
    HAS_AI_CLASSIFIER = True
except Exception as e:
    HAS_AI_CLASSIFIER = False
    import warnings
    warnings.warn(f"AI Classifier not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GeoBounds:
    """Geographic bounding box."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


@dataclass
class DetectedMarking:
    """Represents a detected pavement marking."""
    contour: np.ndarray
    marking_type: str
    confidence: float
    pixel_coords: List[Tuple[int, int]]
    geo_coords: Optional[List[Tuple[float, float]]] = None


class PavementExtractor:
    """
    AI-powered pavement marking extractor.
    
    Uses computer vision and deep learning techniques to detect
    and extract pavement markings from satellite imagery.
    """
    
    def __init__(
        self,
        threshold: int = 200,
        min_area: int = 100,
        geo_bounds: Optional[GeoBounds] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the pavement extractor.
        
        Args:
            threshold: Brightness threshold for marking detection (0-255)
            min_area: Minimum contour area to consider as valid marking
            geo_bounds: Geographic bounds for coordinate transformation
            model_path: Path to trained AI model (optional)
        """
        self.threshold = threshold
        self.min_area = min_area
        self.geo_bounds = geo_bounds
        self.image = None
        self.image_shape = None
        self.geo_transform = None
        
        # Load AI classifier if available
        self.classifier = None
        if HAS_AI_CLASSIFIER:
            # Try to load trained model
            if model_path is None:
                # Default model path
                model_path = Path(__file__).parent.parent.parent / "data" / "trained_model.pkl"
            
            if Path(model_path).exists():
                try:
                    self.classifier = PavementMarkingClassifier(str(model_path))
                    logger.info(f"✅ AI classifier loaded from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load AI model: {e}")
                    self.classifier = None
            else:
                logger.info("No trained model found. Using rule-based classification.")
                self.classifier = PavementMarkingClassifier()  # Will use rule-based fallback
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load satellite image and extract geo-referencing info if available.
        
        Args:
            image_path: Path to the satellite image
            
        Returns:
            Loaded image as numpy array
        """
        path = Path(image_path)
        
        # Try to load with rasterio for georeferenced images
        if HAS_RASTERIO and path.suffix.lower() in ['.tif', '.tiff', '.geotiff']:
            try:
                with rasterio.open(image_path) as src:
                    # Read image data
                    if src.count >= 3:
                        # RGB image
                        r = src.read(1)
                        g = src.read(2)
                        b = src.read(3)
                        self.image = np.dstack([r, g, b])
                    else:
                        self.image = src.read(1)
                    
                    # Extract geo-transform
                    self.geo_transform = src.transform
                    bounds = src.bounds
                    self.geo_bounds = GeoBounds(
                        min_lat=bounds.bottom,
                        max_lat=bounds.top,
                        min_lon=bounds.left,
                        max_lon=bounds.right
                    )
                    logger.info(f"Loaded georeferenced image with bounds: {self.geo_bounds}")
            except Exception as e:
                logger.warning(f"Failed to load with rasterio: {e}. Falling back to OpenCV.")
                self.image = cv2.imread(str(image_path))
                if self.image is not None:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            # Load with OpenCV for regular images
            self.image = cv2.imread(str(image_path))
            if self.image is not None:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        self.image_shape = self.image.shape[:2]
        logger.info(f"Loaded image with shape: {self.image_shape}")
        
        return self.image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for pavement marking detection.
        
        Steps:
        1. Convert to grayscale
        2. Apply CLAHE for contrast enhancement
        3. Gaussian blur for noise reduction
        4. Edge-preserving filter
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Bilateral filter for edge preservation
        filtered = cv2.bilateralFilter(blurred, 9, 75, 75)
        
        return filtered
    
    def detect_markings_traditional(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Detect ONLY bright white pavement markings on dark asphalt.
        Uses strict thresholding to avoid false positives.
        
        Args:
            preprocessed: Preprocessed grayscale image
            
        Returns:
            Binary mask of detected markings
        """
        # STRICT threshold - only detect BRIGHT white markings
        # This prevents detecting buildings, sidewalks, etc.
        _, binary = cv2.threshold(
            preprocessed, 
            self.threshold + 20,  # Higher threshold to be selective
            255, 
            cv2.THRESH_BINARY
        )
        
        # Remove very small noise
        kernel_tiny = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny, iterations=2)
        
        # Only keep line-like structures (remove blobs)
        kernel_hline = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        
        hlines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_hline)
        vlines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_vline)
        
        # Combine horizontal and vertical lines
        lines_only = cv2.bitwise_or(hlines, vlines)
        
        return lines_only
    
    def detect_markings_canny(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Detect line edges - DISABLED to reduce false positives.
        
        Returns empty mask to use only brightness-based detection.
        
        Args:
            preprocessed: Preprocessed grayscale image
            
        Returns:
            Empty binary mask
        """
        # Disabled to avoid detecting building edges, shadows, etc.
        return np.zeros_like(preprocessed)
    
    def combine_detections(
        self, 
        threshold_mask: np.ndarray, 
        edge_mask: np.ndarray
    ) -> np.ndarray:
        """
        Combine threshold-based and edge-based detections.
        
        Args:
            threshold_mask: Mask from threshold detection
            edge_mask: Mask from edge detection
            
        Returns:
            Combined binary mask
        """
        # Use threshold as primary, edges as secondary
        combined = cv2.bitwise_or(threshold_mask, edge_mask)
        
        # Clean up combined mask
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_contours(self, mask: np.ndarray) -> List[DetectedMarking]:
        """
        Extract contours from binary mask and classify markings.
        
        Args:
            mask: Binary mask of detected markings
            
        Returns:
            List of detected marking objects
        """
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        markings = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Strict size filter
            if area < self.min_area:
                continue
            
            # Skip very large regions (buildings, not markings)
            if area > mask.shape[0] * mask.shape[1] * 0.01:
                continue
            
            # CRITICAL: Check if shape is actually a LINE
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            
            aspect_ratio = max(width, height) / min(width, height)
            
            # ONLY accept elongated shapes (lines must be at least 3:1 ratio)
            if aspect_ratio < 3:
                continue
            
            # Classify marking type based on shape
            marking_type = self._classify_marking(contour, area)
            
            # Calculate confidence based on contour properties
            confidence = self._calculate_confidence(contour, area)
            
            # Extract pixel coordinates
            pixel_coords = [(int(p[0][0]), int(p[0][1])) for p in contour]
            
            marking = DetectedMarking(
                contour=contour,
                marking_type=marking_type,
                confidence=confidence,
                pixel_coords=pixel_coords
            )
            
            markings.append(marking)
        
        logger.info(f"Extracted {len(markings)} marking contours")
        return markings
    
    def _classify_marking(self, contour: np.ndarray, area: float) -> str:
        """
        Classify marking using trained AI model or rule-based fallback.
        
        Args:
            contour: Contour points
            area: Contour area
            
        Returns:
            Marking type: "lane_line" or "crosswalk"
        """
        # Use AI classifier if available
        if self.classifier is not None:
            return self.classifier.predict(contour, self.image)
        
        # Fallback: rule-based classification
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            return "lane_line"
        
        aspect_ratio = max(width, height) / min(width, height)
        
        # Lines are very elongated (>8:1)
        # Crosswalks are moderately elongated (2:1 to 8:1)
        if aspect_ratio > 8:
            return "lane_line"
        elif 2 < aspect_ratio <= 8:
            # Could be crosswalk bar
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.4:  # Low circularity = rectangular
                    return "crosswalk"
        
        return "lane_line"
    
    def _calculate_confidence(self, contour: np.ndarray, area: float) -> float:
        """
        Calculate detection confidence score.
        
        Args:
            contour: Contour points
            area: Contour area
            
        Returns:
            Confidence score (0-1)
        """
        # Larger areas generally indicate more confident detections
        area_score = min(1.0, area / 5000)
        
        # More complex contours are more likely to be real markings
        perimeter = cv2.arcLength(contour, True)
        complexity_score = min(1.0, perimeter / 500)
        
        # Combine scores
        confidence = 0.6 * area_score + 0.4 * complexity_score
        
        return min(1.0, confidence)
    
    def pixel_to_geo(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            
        Returns:
            Tuple of (longitude, latitude)
        """
        if self.geo_bounds is None or self.image_shape is None:
            return (x, y)
        
        height, width = self.image_shape
        
        # Linear interpolation from pixel to geo coords
        lon = self.geo_bounds.min_lon + (x / width) * (
            self.geo_bounds.max_lon - self.geo_bounds.min_lon
        )
        lat = self.geo_bounds.max_lat - (y / height) * (
            self.geo_bounds.max_lat - self.geo_bounds.min_lat
        )
        
        return (lon, lat)
    
    def markings_to_geojson(self, markings: List[DetectedMarking]) -> Dict[str, Any]:
        """
        Convert detected markings to GeoJSON format.
        
        Args:
            markings: List of detected markings
            
        Returns:
            GeoJSON FeatureCollection dictionary
        """
        features = []
        
        for i, marking in enumerate(markings):
            # Convert pixel coordinates to geo coordinates
            geo_coords = [
                self.pixel_to_geo(x, y) for x, y in marking.pixel_coords
            ]
            
            # Simplify contour for cleaner output
            if len(geo_coords) > 4:
                epsilon = cv2.arcLength(marking.contour, True) * 0.02
                approx = cv2.approxPolyDP(marking.contour, epsilon, True)
                geo_coords = [
                    self.pixel_to_geo(int(p[0][0]), int(p[0][1])) 
                    for p in approx
                ]
            
            # Close the polygon if needed
            if len(geo_coords) >= 3:
                if geo_coords[0] != geo_coords[-1]:
                    geo_coords.append(geo_coords[0])
                
                # Create polygon geometry
                try:
                    polygon = Polygon(geo_coords)
                    if polygon.is_valid:
                        geometry = mapping(polygon)
                    else:
                        # Try to fix invalid polygon
                        polygon = polygon.buffer(0)
                        geometry = mapping(polygon)
                except Exception as e:
                    logger.warning(f"Failed to create polygon for marking {i}: {e}")
                    continue
            else:
                # Create line geometry for simple markings
                try:
                    line = LineString(geo_coords)
                    geometry = mapping(line)
                except Exception as e:
                    logger.warning(f"Failed to create line for marking {i}: {e}")
                    continue
            
            feature = {
                "type": "Feature",
                "id": i,
                "geometry": geometry,
                "properties": {
                    "marking_type": marking.marking_type,
                    "confidence": round(marking.confidence, 3),
                    "area_pixels": float(cv2.contourArea(marking.contour))
                }
            }
            
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_features": len(features),
                "image_shape": list(self.image_shape) if self.image_shape else None,
                "geo_bounds": {
                    "min_lat": self.geo_bounds.min_lat if self.geo_bounds else None,
                    "max_lat": self.geo_bounds.max_lat if self.geo_bounds else None,
                    "min_lon": self.geo_bounds.min_lon if self.geo_bounds else None,
                    "max_lon": self.geo_bounds.max_lon if self.geo_bounds else None
                } if self.geo_bounds else None
            }
        }
        
        return geojson
    
    def process(self, image_path: str) -> Tuple[Dict[str, Any], float]:
        """
        Run the complete pavement extraction pipeline.
        
        Args:
            image_path: Path to the satellite image
            
        Returns:
            Tuple of (GeoJSON dict, processing time in seconds)
        """
        start_time = time.time()
        
        logger.info(f"Starting pavement extraction from: {image_path}")
        
        # Load image
        image = self.load_image(image_path)
        
        # Preprocess
        logger.info("Preprocessing image...")
        preprocessed = self.preprocess(image)
        
        # Detect markings using multiple methods
        logger.info("Detecting markings...")
        threshold_mask = self.detect_markings_traditional(preprocessed)
        edge_mask = self.detect_markings_canny(preprocessed)
        
        # Combine detections
        combined_mask = self.combine_detections(threshold_mask, edge_mask)
        
        # Extract contours
        logger.info("Extracting contours...")
        markings = self.extract_contours(combined_mask)
        
        # Convert to GeoJSON
        logger.info("Converting to GeoJSON...")
        geojson = self.markings_to_geojson(markings)
        
        processing_time = time.time() - start_time
        logger.info(f"Extraction complete. Found {len(markings)} markings in {processing_time:.2f}s")
        
        return geojson, processing_time
    
    def save_geojson(self, geojson: Dict[str, Any], output_path: str) -> str:
        """
        Save GeoJSON to file.
        
        Args:
            geojson: GeoJSON dictionary
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Saved GeoJSON to: {output_path}")
        return str(output_path)
    
    def get_debug_visualization(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a debug visualization showing detected markings.
        
        Args:
            mask: Binary detection mask
            
        Returns:
            Visualization image
        """
        if self.image is None:
            return mask
        
        # Create colored overlay
        vis = self.image.copy()
        
        # Draw mask in red
        vis[mask > 0] = [255, 0, 0]
        
        # Blend with original
        blended = cv2.addWeighted(self.image, 0.7, vis, 0.3, 0)
        
        return blended


def extract_pavement_markings(
    image_path: str,
    output_path: str,
    threshold: int = 200,
    min_area: int = 100,
    geo_bounds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenience function to extract pavement markings from an image.
    
    Args:
        image_path: Path to input satellite image
        output_path: Path for output GeoJSON
        threshold: Detection threshold
        min_area: Minimum contour area
        geo_bounds: Optional geo bounds dict
        
    Returns:
        Result dictionary with status and paths
    """
    bounds = None
    if geo_bounds:
        bounds = GeoBounds(
            min_lat=geo_bounds.get('min_lat', 0),
            max_lat=geo_bounds.get('max_lat', 0),
            min_lon=geo_bounds.get('min_lon', 0),
            max_lon=geo_bounds.get('max_lon', 0)
        )
    
    extractor = PavementExtractor(
        threshold=threshold,
        min_area=min_area,
        geo_bounds=bounds
    )
    
    geojson, processing_time = extractor.process(image_path)
    saved_path = extractor.save_geojson(geojson, output_path)
    
    return {
        "success": True,
        "geojson_path": saved_path,
        "feature_count": len(geojson.get("features", [])),
        "processing_time": processing_time
    }
