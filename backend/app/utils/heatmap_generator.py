"""
Traffic Heatmap Generator Module

This module processes CCTV video footage to generate traffic density heatmaps.

Pipeline:
1. Load CCTV video file
2. Extract frames at specified sample rate
3. Detect motion/activity using background subtraction
4. Accumulate motion density into grid
5. Normalize and export as heatmap JSON
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
import time

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
class HeatmapConfig:
    """Configuration for heatmap generation."""
    grid_size: int = 50
    frame_sample_rate: int = 5
    motion_threshold: int = 25
    blur_kernel_size: int = 21
    min_contour_area: int = 500


class MotionDetector:
    """
    Motion detection using background subtraction.
    
    Uses MOG2 background subtractor for robust motion detection
    in varying lighting conditions.
    """
    
    def __init__(self, history: int = 500, threshold: int = 16):
        """
        Initialize motion detector.
        
        Args:
            history: Number of frames to use for background model
            threshold: Threshold for background subtraction
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion in a frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of motion regions
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadows are marked as 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        return fg_mask


class OpticalFlowDetector:
    """
    Motion detection using dense optical flow.
    
    Provides more accurate motion vectors but is computationally
    more expensive than background subtraction.
    """
    
    def __init__(self):
        """Initialize optical flow detector."""
        self.prev_gray = None
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion using Farneback optical flow.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Motion magnitude map
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Normalize to 0-255
        magnitude = np.clip(magnitude * 10, 0, 255).astype(np.uint8)
        
        self.prev_gray = gray
        
        return magnitude


class HeatmapGenerator:
    """
    Generates traffic density heatmaps from CCTV video.
    
    Processes video frames to detect motion/activity and
    accumulates into a spatial heatmap grid.
    """
    
    def __init__(
        self,
        config: Optional[HeatmapConfig] = None,
        geo_bounds: Optional[GeoBounds] = None
    ):
        """
        Initialize heatmap generator.
        
        Args:
            config: Heatmap configuration
            geo_bounds: Geographic bounds for the video area
        """
        self.config = config or HeatmapConfig()
        self.geo_bounds = geo_bounds
        self.motion_detector = MotionDetector()
        self.heatmap_grid = None
        self.frame_count = 0
        self.video_shape = None
    
    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """
        Load video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            OpenCV VideoCapture object
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.video_shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        )
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Loaded video: {self.video_shape[1]}x{self.video_shape[0]}, "
                   f"{total_frames} frames, {fps:.2f} FPS")
        
        return cap
    
    def initialize_grid(self):
        """Initialize the heatmap accumulation grid."""
        self.heatmap_grid = np.zeros(
            (self.config.grid_size, self.config.grid_size),
            dtype=np.float64
        )
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and update heatmap grid.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Motion mask for this frame
        """
        # Detect motion
        motion_mask = self.motion_detector.detect(frame)
        
        # Resize motion mask to grid size
        grid_motion = cv2.resize(
            motion_mask,
            (self.config.grid_size, self.config.grid_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Accumulate into heatmap (normalize by 255 to get 0-1 range)
        self.heatmap_grid += grid_motion.astype(np.float64) / 255.0
        self.frame_count += 1
        
        return motion_mask
    
    def process_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Process entire video to generate heatmap.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (normalized heatmap array, frames processed)
        """
        cap = self.load_video(video_path)
        self.initialize_grid()
        
        frame_idx = 0
        processed_count = 0
        
        logger.info("Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample frames according to rate
            if frame_idx % self.config.frame_sample_rate == 0:
                self.process_frame(frame)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} frames...")
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"Video processing complete. Processed {processed_count} frames.")
        
        # Normalize heatmap
        if self.frame_count > 0:
            self.heatmap_grid /= self.frame_count
        
        return self.heatmap_grid, processed_count
    
    def grid_to_geo(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid cell to geographic coordinates (cell center).
        
        Args:
            row: Grid row index
            col: Grid column index
            
        Returns:
            Tuple of (longitude, latitude)
        """
        if self.geo_bounds is None:
            # Return normalized coordinates if no geo bounds
            return (
                col / self.config.grid_size,
                1.0 - row / self.config.grid_size
            )
        
        # Calculate cell center in normalized coords (0-1)
        norm_x = (col + 0.5) / self.config.grid_size
        norm_y = (row + 0.5) / self.config.grid_size
        
        # Convert to geographic coordinates
        lon = self.geo_bounds.min_lon + norm_x * (
            self.geo_bounds.max_lon - self.geo_bounds.min_lon
        )
        lat = self.geo_bounds.max_lat - norm_y * (
            self.geo_bounds.max_lat - self.geo_bounds.min_lat
        )
        
        return (lon, lat)
    
    def heatmap_to_json(self, heatmap: np.ndarray) -> Dict[str, Any]:
        """
        Convert heatmap array to JSON format.
        
        Args:
            heatmap: Normalized heatmap array
            
        Returns:
            Heatmap data dictionary
        """
        points = []
        
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                intensity = float(heatmap[row, col])
                
                # Skip near-zero intensities to reduce data size
                if intensity < 0.001:
                    continue
                
                lon, lat = self.grid_to_geo(row, col)
                
                points.append({
                    "lat": lat,
                    "lon": lon,
                    "intensity": round(intensity, 4)
                })
        
        # Calculate statistics
        non_zero = heatmap[heatmap > 0]
        max_intensity = float(np.max(heatmap)) if len(non_zero) > 0 else 0
        min_intensity = float(np.min(non_zero)) if len(non_zero) > 0 else 0
        mean_intensity = float(np.mean(non_zero)) if len(non_zero) > 0 else 0
        
        result = {
            "bounds": {
                "min_lat": self.geo_bounds.min_lat if self.geo_bounds else 0,
                "max_lat": self.geo_bounds.max_lat if self.geo_bounds else 1,
                "min_lon": self.geo_bounds.min_lon if self.geo_bounds else 0,
                "max_lon": self.geo_bounds.max_lon if self.geo_bounds else 1
            },
            "grid_size": self.config.grid_size,
            "points": points,
            "statistics": {
                "max_intensity": round(max_intensity, 4),
                "min_intensity": round(min_intensity, 4),
                "mean_intensity": round(mean_intensity, 4),
                "total_points": len(points),
                "frames_processed": self.frame_count
            }
        }
        
        return result
    
    def generate_geojson_heatmap(self, heatmap: np.ndarray) -> Dict[str, Any]:
        """
        Convert heatmap to GeoJSON format with grid cells as polygons.
        
        Args:
            heatmap: Normalized heatmap array
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                intensity = float(heatmap[row, col])
                
                if intensity < 0.001:
                    continue
                
                # Calculate cell corners
                if self.geo_bounds:
                    cell_width = (self.geo_bounds.max_lon - self.geo_bounds.min_lon) / self.config.grid_size
                    cell_height = (self.geo_bounds.max_lat - self.geo_bounds.min_lat) / self.config.grid_size
                    
                    min_lon = self.geo_bounds.min_lon + col * cell_width
                    max_lon = min_lon + cell_width
                    max_lat = self.geo_bounds.max_lat - row * cell_height
                    min_lat = max_lat - cell_height
                else:
                    cell_size = 1.0 / self.config.grid_size
                    min_lon = col * cell_size
                    max_lon = min_lon + cell_size
                    max_lat = 1.0 - row * cell_size
                    min_lat = max_lat - cell_size
                
                # Create polygon coordinates (GeoJSON format: [[lon, lat], ...])
                coordinates = [[
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]  # Close the polygon
                ]]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    },
                    "properties": {
                        "intensity": round(intensity, 4),
                        "row": row,
                        "col": col
                    }
                }
                
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "grid_size": self.config.grid_size,
                "frames_processed": self.frame_count
            }
        }
        
        return geojson
    
    def process(self, video_path: str) -> Tuple[Dict[str, Any], float]:
        """
        Run the complete heatmap generation pipeline.
        
        Args:
            video_path: Path to CCTV video file
            
        Returns:
            Tuple of (heatmap JSON dict, processing time)
        """
        start_time = time.time()
        
        logger.info(f"Starting heatmap generation from: {video_path}")
        
        # Process video
        heatmap, frames_processed = self.process_video(video_path)
        
        # Apply Gaussian blur for smoother visualization
        heatmap_smooth = cv2.GaussianBlur(
            heatmap.astype(np.float32),
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            0
        )
        
        # Normalize to 0-1 range
        if np.max(heatmap_smooth) > 0:
            heatmap_smooth = heatmap_smooth / np.max(heatmap_smooth)
        
        # Convert to JSON
        logger.info("Converting to JSON format...")
        heatmap_json = self.heatmap_to_json(heatmap_smooth)
        
        processing_time = time.time() - start_time
        logger.info(f"Heatmap generation complete in {processing_time:.2f}s")
        
        return heatmap_json, processing_time
    
    def save_heatmap(self, heatmap_json: Dict[str, Any], output_path: str) -> str:
        """
        Save heatmap to JSON file.
        
        Args:
            heatmap_json: Heatmap data dictionary
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(heatmap_json, f, indent=2)
        
        logger.info(f"Saved heatmap to: {output_path}")
        return str(output_path)


def generate_traffic_heatmap(
    video_path: str,
    output_path: str,
    grid_size: int = 50,
    frame_sample_rate: int = 5,
    geo_bounds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate traffic heatmap from video.
    
    Args:
        video_path: Path to input video
        output_path: Path for output JSON
        grid_size: Resolution of heatmap grid
        frame_sample_rate: Process every Nth frame
        geo_bounds: Optional geo bounds dict
        
    Returns:
        Result dictionary with status and paths
    """
    config = HeatmapConfig(
        grid_size=grid_size,
        frame_sample_rate=frame_sample_rate
    )
    
    bounds = None
    if geo_bounds:
        bounds = GeoBounds(
            min_lat=geo_bounds.get('min_lat', 0),
            max_lat=geo_bounds.get('max_lat', 1),
            min_lon=geo_bounds.get('min_lon', 0),
            max_lon=geo_bounds.get('max_lon', 1)
        )
    
    generator = HeatmapGenerator(config=config, geo_bounds=bounds)
    heatmap_json, processing_time = generator.process(video_path)
    saved_path = generator.save_heatmap(heatmap_json, output_path)
    
    return {
        "success": True,
        "heatmap_path": saved_path,
        "total_frames_processed": heatmap_json["statistics"]["frames_processed"],
        "total_points": heatmap_json["statistics"]["total_points"],
        "processing_time": processing_time
    }


def create_demo_heatmap(
    output_path: str,
    grid_size: int = 50,
    geo_bounds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create a demo heatmap without video input.
    
    Generates synthetic heatmap data for testing purposes.
    
    Args:
        output_path: Path for output JSON
        grid_size: Resolution of heatmap grid
        geo_bounds: Optional geo bounds dict
        
    Returns:
        Result dictionary
    """
    if geo_bounds is None:
        geo_bounds = {
            "min_lat": 40.7128,
            "max_lat": 40.7138,
            "min_lon": -74.0060,
            "max_lon": -74.0050
        }
    
    points = []
    
    # Generate synthetic traffic patterns
    np.random.seed(42)
    
    for _ in range(200):
        # Create clusters around typical traffic areas
        cluster_centers = [
            (0.3, 0.5),  # Left lane
            (0.5, 0.5),  # Center
            (0.7, 0.5),  # Right lane
            (0.5, 0.3),  # Intersection top
            (0.5, 0.7),  # Intersection bottom
        ]
        
        center = cluster_centers[np.random.randint(len(cluster_centers))]
        x = center[0] + np.random.normal(0, 0.1)
        y = center[1] + np.random.normal(0, 0.1)
        
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        lon = geo_bounds["min_lon"] + x * (geo_bounds["max_lon"] - geo_bounds["min_lon"])
        lat = geo_bounds["min_lat"] + y * (geo_bounds["max_lat"] - geo_bounds["min_lat"])
        
        intensity = np.random.uniform(0.3, 1.0)
        
        points.append({
            "lat": lat,
            "lon": lon,
            "intensity": round(intensity, 4)
        })
    
    heatmap_json = {
        "bounds": geo_bounds,
        "grid_size": grid_size,
        "points": points,
        "statistics": {
            "max_intensity": 1.0,
            "min_intensity": 0.3,
            "mean_intensity": 0.65,
            "total_points": len(points),
            "frames_processed": 0
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(heatmap_json, f, indent=2)
    
    return {
        "success": True,
        "heatmap_path": str(output_path),
        "total_points": len(points),
        "processing_time": 0.1
    }
