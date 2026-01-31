"""
Enhanced Pavement Marking Detector V2

This module uses advanced computer vision to detect pavement markings:
1. Color-based detection (white/yellow markings on dark road)
2. Adaptive thresholding for varying lighting
3. Line detection using Hough Transform
4. Shape analysis for marking classification

HOW THE AI SEES THE ROAD:
- Road surface = dark gray/black pixels (brightness < 100)
- Lane lines = bright white pixels (brightness > 200)
- Crosswalks = repeating white stripes pattern
- The AI looks for HIGH CONTRAST between road and markings
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import json

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
class PavementMarking:
    """Detected pavement marking with all properties."""
    contour: np.ndarray
    marking_type: str  # lane_line, crosswalk, stop_line, arrow
    confidence: float
    center: Tuple[int, int]
    area: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h


class EnhancedPavementDetector:
    """
    Enhanced pavement marking detector using multiple detection strategies.
    
    Detection Pipeline:
    1. Convert to multiple color spaces (RGB, HSV, LAB)
    2. Detect white markings using brightness
    3. Detect yellow markings using color
    4. Apply road mask to focus only on road areas
    5. Use morphology to clean up detections
    6. Extract and classify contours
    """
    
    def __init__(
        self,
        white_threshold: int = 180,  # Lower = more sensitive
        min_area: int = 50,
        geo_bounds: Optional[GeoBounds] = None
    ):
        self.white_threshold = white_threshold
        self.min_area = min_area
        self.geo_bounds = geo_bounds
        self.image = None
        self.image_shape = None
        self.debug_images = {}
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_shape = self.image.shape[:2]
        logger.info(f"Loaded image: {self.image_shape}")
        return self.image
        
    def detect_white_markings(self, image: np.ndarray) -> np.ndarray:
        """
        Detect white pavement markings.
        
        White markings are: bright (>180) + low saturation (<50)
        """
        # Convert to HSV and grayscale
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Find bright areas
        _, bright_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # White = low saturation (< 50) + high value
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_threshold]),
            np.array([180, 50, 255])
        )
        
        # Combine: bright AND white
        markings = cv2.bitwise_and(bright_mask, white_mask)
        
        # Morphological cleanup
        kernel = np.ones((2, 2), np.uint8)
        markings = cv2.morphologyEx(markings, cv2.MORPH_OPEN, kernel)
        
        # Connect nearby pixels
        kernel2 = np.ones((3, 3), np.uint8)
        markings = cv2.morphologyEx(markings, cv2.MORPH_CLOSE, kernel2)
        
        self.debug_images['white_detection'] = markings
        return markings
    
    def detect_yellow_markings(self, image: np.ndarray) -> np.ndarray:
        """
        Detect yellow pavement markings.
        
        Yellow markings have:
        - Hue between 15-35 in HSV
        - High saturation
        - Medium to high value
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Yellow color range
        yellow_mask = cv2.inRange(
            hsv,
            np.array([15, 100, 100]),
            np.array([35, 255, 255])
        )
        
        self.debug_images['yellow_detection'] = yellow_mask
        return yellow_mask
    
    def create_road_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create mask for road surface (dark asphalt areas).
        This helps focus detection on actual roads.
        
        Roads are typically:
        - Dark gray/black (brightness 30-120)
        - Low saturation (not colored)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Road is darker than surroundings
        # Use adaptive thresholding to find dark regions
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Dark areas are potential road
        _, dark_mask = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Fill holes and expand
        kernel = np.ones((15, 15), np.uint8)
        road_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.dilate(road_mask, kernel, iterations=2)
        
        self.debug_images['road_mask'] = road_mask
        return road_mask
    
    def clean_detections(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean detection mask."""
        # Remove noise
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Connect nearby regions
        kernel_medium = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        return cleaned
    
    def extract_markings(self, mask: np.ndarray) -> List[PavementMarking]:
        """Extract and classify marking contours - BALANCED FILTERING."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markings = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip too small (noise)
            if area < self.min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            
            # FILTER 1: Skip if NOT elongated (buildings are boxy)
            # Accept markings with aspect ratio > 2
            if aspect_ratio < 2.0:
                continue
            
            # FILTER 2: Skip if too thick (buildings are wide)
            # Crosswalks can be up to 150px wide, lane lines < 30px
            if min(w, h) > 150:
                continue
            
            # FILTER 3: Skip if HUGE area (> 1% of image = building)
            max_area = self.image_shape[0] * self.image_shape[1] * 0.01
            if area > max_area:
                continue
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            
            # FILTER 4: Must be fairly solid (> 0.5) - not hollow building outline
            if solidity < 0.5:
                continue
            
            center = (x + w // 2, y + h // 2)
            
            # Classify based on shape
            marking_type, confidence = self._classify_shape(contour, area, w, h)
            
            marking = PavementMarking(
                contour=contour,
                marking_type=marking_type,
                confidence=confidence,
                center=center,
                area=area,
                bounding_box=(x, y, w, h)
            )
            markings.append(marking)
        
        logger.info(f"Extracted {len(markings)} markings")
        return markings
    
    def _classify_shape(self, contour, area, w, h) -> Tuple[str, float]:
        """Classify marking based on shape analysis."""
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        
        # Calculate solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        
        # Calculate perimeter and circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
        
        confidence = 0.7
        
        # Very elongated + thin = lane line
        if aspect_ratio > 8 and min(w, h) < 20:
            return "lane_line", min(0.95, confidence + 0.2)
        
        # Moderately elongated + wider = crosswalk bar
        elif 3 <= aspect_ratio <= 8 and min(w, h) > 10:
            if solidity > 0.7:
                return "crosswalk", confidence + 0.15
        
        # Wide and short = stop line
        elif w > h * 4 and solidity > 0.8:
            return "stop_line", confidence + 0.1
        
        # Default to lane line if elongated enough
        return "lane_line", confidence * 0.9
    
    def detect(self, image_path: str = None) -> List[PavementMarking]:
        """
        Run full detection pipeline.
        
        Returns list of detected pavement markings.
        """
        if image_path:
            self.load_image(image_path)
        
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Step 1: Detect white markings
        white_mask = self.detect_white_markings(self.image)
        
        # Step 2: Detect yellow markings
        yellow_mask = self.detect_yellow_markings(self.image)
        
        # Step 3: Combine all markings
        all_markings_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Step 4: Detect road areas and filter to road context
        road_mask = self.detect_roads(self.image)
        
        # Only keep markings that are near roads (within 30 pixels)
        kernel = np.ones((30, 30), np.uint8)
        road_expanded = cv2.dilate(road_mask, kernel, iterations=1)
        
        # Filter to road areas
        road_markings = cv2.bitwise_and(all_markings_mask, road_expanded)
        
        # Step 5: Clean up
        cleaned = self.clean_detections(road_markings)
        self.debug_images['final_mask'] = cleaned
        
        # Step 6: Extract markings
        markings = self.extract_markings(cleaned)
        
        return markings
    
    def detect_roads(self, image: np.ndarray) -> np.ndarray:
        """
        Detect road surfaces (dark gray asphalt areas).
        Roads are dark (brightness 40-130) and low saturation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Roads are medium-dark gray (40-130 brightness)
        road_brightness = cv2.inRange(gray, 40, 130)
        
        # Roads have low saturation (not colorful)
        low_saturation = cv2.inRange(hsv[:,:,1], 0, 60)
        
        # Combine
        road_mask = cv2.bitwise_and(road_brightness, low_saturation)
        
        # Morphological operations to clean up
        kernel = np.ones((15, 15), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        self.debug_images['road_mask'] = road_mask
        return road_mask
    
    def pixel_to_geo(self, x: int, y: int) -> Tuple[float, float]:
        """Convert pixel to geographic coordinates."""
        if self.geo_bounds is None or self.image_shape is None:
            return (float(x), float(y))
        
        height, width = self.image_shape
        lon = self.geo_bounds.min_lon + (x / width) * (self.geo_bounds.max_lon - self.geo_bounds.min_lon)
        lat = self.geo_bounds.max_lat - (y / height) * (self.geo_bounds.max_lat - self.geo_bounds.min_lat)
        return (lon, lat)
    
    def to_geojson(self, markings: List[PavementMarking]) -> Dict[str, Any]:
        """Convert markings to GeoJSON format."""
        features = []
        
        for i, m in enumerate(markings):
            # Convert contour to geo coordinates
            coords = []
            for point in m.contour:
                px, py = point[0]
                lon, lat = self.pixel_to_geo(int(px), int(py))
                coords.append([lon, lat])
            
            # Close the polygon
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            
            feature = {
                "type": "Feature",
                "id": i,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {
                    "marking_type": m.marking_type,
                    "confidence": round(m.confidence, 3),
                    "area_pixels": round(m.area, 1),
                    "center_x": m.center[0],
                    "center_y": m.center[1]
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_markings": len(markings),
                "detection_threshold": self.white_threshold,
                "geo_bounds": {
                    "min_lat": self.geo_bounds.min_lat if self.geo_bounds else None,
                    "max_lat": self.geo_bounds.max_lat if self.geo_bounds else None,
                    "min_lon": self.geo_bounds.min_lon if self.geo_bounds else None,
                    "max_lon": self.geo_bounds.max_lon if self.geo_bounds else None
                } if self.geo_bounds else None
            }
        }
        
        return geojson
    
    def create_visualization(self, markings: List[PavementMarking], output_path: str = None) -> np.ndarray:
        """
        Create visualization showing what the AI detected.
        
        Colors:
        - Yellow boxes = lane lines
        - White boxes = crosswalks
        - Green boxes = stop lines
        """
        vis = self.image.copy()
        
        colors = {
            "lane_line": (255, 255, 0),      # Yellow
            "crosswalk": (255, 255, 255),    # White
            "stop_line": (0, 255, 0),        # Green
            "unknown": (255, 128, 0)         # Orange
        }
        
        for m in markings:
            color = colors.get(m.marking_type, colors["unknown"])
            
            # Draw contour
            cv2.drawContours(vis, [m.contour], -1, color, 2)
            
            # Draw bounding box
            x, y, w, h = m.bounding_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            
            # Add label
            label = f"{m.marking_type[:4]} {m.confidence:.0%}"
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add summary
        summary = f"Detected: {len(markings)} markings"
        cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved visualization to {output_path}")
        
        return vis


def extract_pavement_markings_v2(
    image_path: str,
    output_geojson_path: str,
    geo_bounds: Dict[str, float] = None,
    threshold: int = 180,
    min_area: int = 50,
    save_visualization: bool = True
) -> Dict[str, Any]:
    """
    Main function to extract pavement markings from satellite image.
    
    Args:
        image_path: Path to satellite image
        output_geojson_path: Where to save GeoJSON
        geo_bounds: Geographic bounds dict with min_lat, max_lat, min_lon, max_lon
        threshold: Detection sensitivity (lower = more detections)
        min_area: Minimum marking size in pixels
        save_visualization: Whether to save debug image
        
    Returns:
        Dict with detection results
    """
    # Setup bounds
    bounds = None
    if geo_bounds:
        bounds = GeoBounds(
            min_lat=geo_bounds.get('min_lat', 0),
            max_lat=geo_bounds.get('max_lat', 0),
            min_lon=geo_bounds.get('min_lon', 0),
            max_lon=geo_bounds.get('max_lon', 0)
        )
    
    # Create detector
    detector = EnhancedPavementDetector(
        white_threshold=threshold,
        min_area=min_area,
        geo_bounds=bounds
    )
    
    # Run detection
    markings = detector.detect(image_path)
    
    # Convert to GeoJSON
    geojson = detector.to_geojson(markings)
    
    # Save GeoJSON
    output_path = Path(output_geojson_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    # Save visualization
    if save_visualization:
        vis_path = str(output_path).replace('.geojson', '_detection.png')
        detector.create_visualization(markings, vis_path)
    
    # Count by type
    type_counts = {}
    for m in markings:
        type_counts[m.marking_type] = type_counts.get(m.marking_type, 0) + 1
    
    result = {
        "success": True,
        "geojson_path": str(output_path),
        "total_markings": len(markings),
        "by_type": type_counts,
        "image_size": detector.image_shape,
        "threshold_used": threshold
    }
    
    logger.info(f"Detection complete: {len(markings)} markings found")
    return result
