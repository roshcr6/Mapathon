"""
Vehicle Detection and Traffic Flow Analyzer

Advanced AI module for:
1. Vehicle detection using background subtraction and contour analysis
2. Vehicle tracking across frames
3. Traffic flow analysis and path extraction
4. Heatmap generation based on traffic density

This module processes CCTV footage to understand traffic patterns.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for detected object."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class TrackedVehicle:
    """Represents a tracked vehicle across frames."""
    id: int
    positions: List[Tuple[int, int]] = field(default_factory=list)
    bboxes: List[BoundingBox] = field(default_factory=list)
    last_seen: int = 0
    velocity: Tuple[float, float] = (0, 0)
    
    def add_detection(self, bbox: BoundingBox, frame_id: int):
        """Add new detection."""
        self.bboxes.append(bbox)
        self.positions.append(bbox.center)
        
        # Calculate velocity
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            # Smooth velocity with exponential moving average
            alpha = 0.3
            self.velocity = (
                alpha * dx + (1 - alpha) * self.velocity[0],
                alpha * dy + (1 - alpha) * self.velocity[1]
            )
        
        self.last_seen = frame_id
    
    def predict_position(self) -> Tuple[int, int]:
        """Predict next position based on velocity."""
        if not self.positions:
            return (0, 0)
        
        last_pos = self.positions[-1]
        return (
            int(last_pos[0] + self.velocity[0]),
            int(last_pos[1] + self.velocity[1])
        )
    
    @property
    def path_length(self) -> float:
        """Total path length traveled."""
        if len(self.positions) < 2:
            return 0
        
        total = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)
        
        return total


class VehicleDetector:
    """
    Detects vehicles in video frames using background subtraction
    and contour analysis.
    """
    
    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 50000,
        history: int = 500,
        var_threshold: int = 50
    ):
        """
        Initialize vehicle detector.
        
        Args:
            min_area: Minimum contour area to consider as vehicle
            max_area: Maximum contour area
            history: Number of frames for background model
            var_threshold: Threshold for background subtraction
        """
        self.min_area = min_area
        self.max_area = max_area
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )
        
        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: BGR video frame
            
        Returns:
            List of bounding boxes for detected vehicles
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (marked as 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (vehicles are roughly rectangular)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
            
            detections.append(BoundingBox(x, y, w, h))
        
        return detections
    
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get the foreground mask for visualization."""
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)
        return fg_mask


class VehicleTracker:
    """
    Tracks vehicles across frames using simple IoU-based matching.
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        min_iou: float = 0.3
    ):
        """
        Initialize tracker.
        
        Args:
            max_disappeared: Max frames before removing track
            min_iou: Minimum IoU for matching
        """
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.vehicles: Dict[int, TrackedVehicle] = {}
        self.next_id = 0
        self.frame_id = 0
    
    def update(self, detections: List[BoundingBox]) -> Dict[int, TrackedVehicle]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detected bounding boxes
            
        Returns:
            Dictionary of active tracked vehicles
        """
        self.frame_id += 1
        
        if not detections:
            # Remove stale tracks
            self._remove_stale_tracks()
            return self.vehicles
        
        if not self.vehicles:
            # No existing tracks - create new ones
            for det in detections:
                self._create_track(det)
            return self.vehicles
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_id, det in matched:
            self.vehicles[track_id].add_detection(det, self.frame_id)
        
        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            self._create_track(det)
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        return self.vehicles
    
    def _create_track(self, bbox: BoundingBox):
        """Create a new track."""
        vehicle = TrackedVehicle(id=self.next_id)
        vehicle.add_detection(bbox, self.frame_id)
        self.vehicles[self.next_id] = vehicle
        self.next_id += 1
    
    def _match_detections(
        self, 
        detections: List[BoundingBox]
    ) -> Tuple[List, List, List]:
        """Match detections to existing tracks using IoU."""
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.vehicles.keys())
        
        if not self.vehicles or not detections:
            return matched, [detections[i] for i in unmatched_dets], unmatched_tracks
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.vehicles), len(detections)))
        track_ids = list(self.vehicles.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.vehicles[track_id]
            if track.bboxes:
                last_bbox = track.bboxes[-1]
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = last_bbox.iou(det)
        
        # Greedy matching
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = np.max(iou_matrix)
            if max_iou < self.min_iou:
                break
            
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = max_idx
            
            track_id = track_ids[track_idx]
            matched.append((track_id, detections[det_idx]))
            
            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if track_id in unmatched_tracks:
                unmatched_tracks.remove(track_id)
            
            # Zero out matched row and column
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0
        
        return matched, [detections[i] for i in unmatched_dets], unmatched_tracks
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        for track_id, vehicle in self.vehicles.items():
            if self.frame_id - vehicle.last_seen > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.vehicles[track_id]


class TrafficFlowAnalyzer:
    """
    Analyzes traffic flow patterns from tracked vehicles.
    Generates heatmaps and flow statistics.
    """
    
    def __init__(self, grid_size: int = 50):
        """
        Initialize analyzer.
        
        Args:
            grid_size: Size of the heatmap grid
        """
        self.grid_size = grid_size
        self.density_grid = None
        self.flow_grid = None
        self.video_shape = None
        self.all_paths: List[List[Tuple[int, int]]] = []
        self.frame_count = 0
    
    def initialize(self, video_shape: Tuple[int, int]):
        """Initialize grids based on video dimensions."""
        self.video_shape = video_shape
        self.density_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        self.flow_grid = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float64)
    
    def update(self, vehicles: Dict[int, TrackedVehicle]):
        """
        Update flow analysis with current vehicle states.
        
        Args:
            vehicles: Dictionary of tracked vehicles
        """
        self.frame_count += 1
        
        for vehicle in vehicles.values():
            if len(vehicle.positions) < 2:
                continue
            
            # Get current position
            pos = vehicle.positions[-1]
            
            # Convert to grid coordinates
            grid_x = int(pos[0] / self.video_shape[1] * self.grid_size)
            grid_y = int(pos[1] / self.video_shape[0] * self.grid_size)
            
            grid_x = np.clip(grid_x, 0, self.grid_size - 1)
            grid_y = np.clip(grid_y, 0, self.grid_size - 1)
            
            # Update density
            self.density_grid[grid_y, grid_x] += 1
            
            # Update flow direction
            self.flow_grid[grid_y, grid_x, 0] += vehicle.velocity[0]
            self.flow_grid[grid_y, grid_x, 1] += vehicle.velocity[1]
    
    def add_completed_path(self, vehicle: TrackedVehicle):
        """Add a completed vehicle path for analysis."""
        if len(vehicle.positions) > 5:
            self.all_paths.append(vehicle.positions.copy())
    
    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """Get the density heatmap."""
        if self.density_grid is None:
            return np.zeros((self.grid_size, self.grid_size))
        
        heatmap = self.density_grid.copy()
        
        if normalize and np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Apply Gaussian blur for smoother visualization
        heatmap = cv2.GaussianBlur(
            heatmap.astype(np.float32),
            (11, 11),
            0
        )
        
        if normalize and np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def get_flow_field(self) -> np.ndarray:
        """Get the flow direction field."""
        if self.flow_grid is None:
            return np.zeros((self.grid_size, self.grid_size, 2))
        
        # Normalize flow vectors
        flow = self.flow_grid.copy()
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitude[magnitude == 0] = 1
        flow[..., 0] /= magnitude
        flow[..., 1] /= magnitude
        
        return flow
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get traffic statistics."""
        return {
            "total_paths": len(self.all_paths),
            "frames_processed": self.frame_count,
            "average_path_length": np.mean([len(p) for p in self.all_paths]) if self.all_paths else 0,
            "grid_size": self.grid_size
        }


class TrafficVideoProcessor:
    """
    Main class for processing CCTV footage.
    Combines detection, tracking, and analysis.
    """
    
    def __init__(
        self,
        grid_size: int = 50,
        frame_sample_rate: int = 2,
        min_vehicle_area: int = 500
    ):
        """
        Initialize processor.
        
        Args:
            grid_size: Heatmap grid resolution
            frame_sample_rate: Process every Nth frame
            min_vehicle_area: Minimum area to consider as vehicle
        """
        self.detector = VehicleDetector(min_area=min_vehicle_area)
        self.tracker = VehicleTracker()
        self.analyzer = TrafficFlowAnalyzer(grid_size=grid_size)
        self.frame_sample_rate = frame_sample_rate
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None for all)
            progress_callback: Callback for progress updates
            
        Returns:
            Processing results including heatmap data
        """
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing video: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
        
        # Initialize analyzer
        self.analyzer.initialize((height, width))
        
        frame_idx = 0
        processed_count = 0
        previous_vehicles = set()
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            # Sample frames
            if frame_idx % self.frame_sample_rate == 0:
                # Detect vehicles
                detections = self.detector.detect(frame)
                
                # Update tracker
                vehicles = self.tracker.update(detections)
                
                # Check for completed tracks
                current_ids = set(vehicles.keys())
                completed_ids = previous_vehicles - current_ids
                
                for vid in completed_ids:
                    if vid in self.tracker.vehicles:
                        self.analyzer.add_completed_path(self.tracker.vehicles[vid])
                
                previous_vehicles = current_ids
                
                # Update analyzer
                self.analyzer.update(vehicles)
                
                processed_count += 1
                
                if progress_callback and processed_count % 50 == 0:
                    progress = frame_idx / total_frames * 100
                    progress_callback(progress, processed_count)
            
            frame_idx += 1
        
        cap.release()
        
        # Add remaining paths
        for vehicle in self.tracker.vehicles.values():
            self.analyzer.add_completed_path(vehicle)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Processed {processed_count} frames in {processing_time:.2f}s")
        
        return {
            "success": True,
            "heatmap": self.analyzer.get_heatmap(),
            "flow_field": self.analyzer.get_flow_field(),
            "statistics": self.analyzer.get_statistics(),
            "video_info": {
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "fps": fps
            },
            "processing_time": processing_time
        }
    
    def heatmap_to_json(
        self,
        heatmap: np.ndarray,
        geo_bounds: Dict[str, float],
        road_center_offset: Tuple[float, float] = (0.5, 0.5),
        road_width_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """
        Convert heatmap array to JSON format with geo coordinates.
        Maps video traffic to a specific road area within the satellite bounds.
        
        Args:
            heatmap: 2D heatmap array
            geo_bounds: Geographic bounds for the area
            road_center_offset: (x, y) position of road center (0-1 normalized)
            road_width_ratio: How much of the bounds the road occupies (0-1)
            
        Returns:
            Heatmap data in JSON-serializable format
        """
        points = []
        grid_size = heatmap.shape[0]
        
        # Calculate road bounds within the satellite area
        # Instead of spanning the entire area, concentrate on a road section
        lat_range = geo_bounds["max_lat"] - geo_bounds["min_lat"]
        lon_range = geo_bounds["max_lon"] - geo_bounds["min_lon"]
        
        # Center the road within the satellite area
        road_lat_size = lat_range * road_width_ratio
        road_lon_size = lon_range * road_width_ratio
        
        road_center_lat = geo_bounds["min_lat"] + lat_range * road_center_offset[1]
        road_center_lon = geo_bounds["min_lon"] + lon_range * road_center_offset[0]
        
        road_bounds = {
            "min_lat": road_center_lat - road_lat_size / 2,
            "max_lat": road_center_lat + road_lat_size / 2,
            "min_lon": road_center_lon - road_lon_size / 2,
            "max_lon": road_center_lon + road_lon_size / 2
        }
        
        for row in range(grid_size):
            for col in range(grid_size):
                intensity = float(heatmap[row, col])
                
                if intensity < 0.01:
                    continue
                
                # Convert grid to geo coordinates (within road bounds)
                norm_x = (col + 0.5) / grid_size
                norm_y = (row + 0.5) / grid_size
                
                lon = road_bounds["min_lon"] + norm_x * (
                    road_bounds["max_lon"] - road_bounds["min_lon"]
                )
                lat = road_bounds["max_lat"] - norm_y * (
                    road_bounds["max_lat"] - road_bounds["min_lat"]
                )
                
                points.append({
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "intensity": round(intensity, 4)
                })
        
        return {
            "bounds": geo_bounds,
            "grid_size": grid_size,
            "points": points,
            "statistics": {
                "max_intensity": float(np.max(heatmap)),
                "min_intensity": float(np.min(heatmap[heatmap > 0])) if np.any(heatmap > 0) else 0,
                "mean_intensity": float(np.mean(heatmap[heatmap > 0])) if np.any(heatmap > 0) else 0,
                "total_points": len(points)
            }
        }


def process_traffic_video(
    video_path: str,
    output_path: str,
    geo_bounds: Dict[str, float],
    grid_size: int = 50,
    frame_sample_rate: int = 2,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a traffic video.
    
    Args:
        video_path: Path to input video
        output_path: Path for output JSON
        geo_bounds: Geographic bounds for mapping
        grid_size: Heatmap resolution
        frame_sample_rate: Process every Nth frame
        max_frames: Maximum frames to process
        
    Returns:
        Processing results
    """
    processor = TrafficVideoProcessor(
        grid_size=grid_size,
        frame_sample_rate=frame_sample_rate
    )
    
    def progress_callback(percent, frames):
        logger.info(f"Progress: {percent:.1f}% ({frames} frames)")
    
    result = processor.process_video(
        video_path,
        max_frames=max_frames,
        progress_callback=progress_callback
    )
    
    if result["success"]:
        # Concentrate heatmap on the middle 40% of the area (typical road width)
        # Centered at 50% horizontally and vertically
        heatmap_json = processor.heatmap_to_json(
            result["heatmap"], 
            geo_bounds,
            road_center_offset=(0.5, 0.5),  # Center of the satellite image
            road_width_ratio=0.4  # Road occupies 40% of the bounds
        )
        heatmap_json["statistics"]["frames_processed"] = result["statistics"]["frames_processed"]
        heatmap_json["statistics"]["total_paths"] = result["statistics"]["total_paths"]
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(heatmap_json, f, indent=2)
        
        result["heatmap_json"] = heatmap_json
        result["output_path"] = str(output_path)
    
    return result
