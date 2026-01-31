"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class GeoBounds(BaseModel):
    """Geographic bounding box coordinates."""
    min_lat: float = Field(..., description="Minimum latitude")
    max_lat: float = Field(..., description="Maximum latitude")
    min_lon: float = Field(..., description="Minimum longitude")
    max_lon: float = Field(..., description="Maximum longitude")


class PavementExtractionRequest(BaseModel):
    """Request model for pavement extraction."""
    threshold: Optional[int] = Field(200, ge=0, le=255, description="Brightness threshold for detection")
    min_area: Optional[int] = Field(100, ge=0, description="Minimum contour area to consider")
    geo_bounds: Optional[GeoBounds] = None


class PavementExtractionResponse(BaseModel):
    """Response model for pavement extraction."""
    success: bool
    message: str
    geojson_path: Optional[str] = None
    feature_count: Optional[int] = None
    processing_time: Optional[float] = None


class HeatmapGenerationRequest(BaseModel):
    """Request model for heatmap generation."""
    grid_size: Optional[int] = Field(50, ge=10, le=200, description="Grid resolution")
    frame_sample_rate: Optional[int] = Field(5, ge=1, le=30, description="Process every Nth frame")
    geo_bounds: Optional[GeoBounds] = None


class HeatmapGenerationResponse(BaseModel):
    """Response model for heatmap generation."""
    success: bool
    message: str
    heatmap_path: Optional[str] = None
    total_frames_processed: Optional[int] = None
    processing_time: Optional[float] = None


class GeoJSONFeature(BaseModel):
    """GeoJSON Feature model."""
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]


class GeoJSONCollection(BaseModel):
    """GeoJSON FeatureCollection model."""
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


class HeatmapPoint(BaseModel):
    """Single heatmap data point."""
    lat: float
    lon: float
    intensity: float


class HeatmapData(BaseModel):
    """Complete heatmap data structure."""
    bounds: GeoBounds
    grid_size: int
    points: List[HeatmapPoint]
    max_intensity: float
    min_intensity: float


class MarkingType(str, Enum):
    """Types of pavement markings detected."""
    LANE_LINE = "lane_line"
    CROSSWALK = "crosswalk"
    ARROW = "arrow"
    STOP_LINE = "stop_line"
    UNKNOWN = "unknown"
