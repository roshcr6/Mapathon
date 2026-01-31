"""
Configuration settings for the Mapathon backend.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
FOOTAGE_DIR = BASE_DIR.parent / "footage"  # Points to mapathon/footage

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
MODEL_CONFIDENCE_THRESHOLD = 0.5
PAVEMENT_DETECTION_THRESHOLD = 200  # Brightness threshold for pavement markings

# Heatmap settings
HEATMAP_GRID_SIZE = 50  # Grid resolution for heatmap
FRAME_SAMPLE_RATE = 5  # Process every Nth frame

# GeoJSON output paths
GEOJSON_OUTPUT_PATH = OUTPUT_DIR / "pavement_markings.geojson"
HEATMAP_OUTPUT_PATH = OUTPUT_DIR / "traffic_heatmap.json"

# Default geo bounds (will be overridden by actual georeferenced data)
DEFAULT_BOUNDS = {
    "min_lat": 40.7128,
    "max_lat": 40.7138,
    "min_lon": -74.0060,
    "max_lon": -74.0050
}

# CORS settings
CORS_ORIGINS = [
    "http://localhost:5176",
    "http://127.0.0.1:5176",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
