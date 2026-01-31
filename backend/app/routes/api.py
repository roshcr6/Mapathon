"""
Complete API Routes for Mapathon Backend

Provides endpoints for:
- Satellite data download from QGIS sources
- AI-powered pavement marking extraction
- CCTV video processing and traffic analysis
- Heatmap generation and overlay
- Data retrieval and visualization
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from typing import Optional
import shutil
import json
import os
import time
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ..config import UPLOAD_DIR, OUTPUT_DIR, GEOJSON_OUTPUT_PATH, HEATMAP_OUTPUT_PATH, DEFAULT_BOUNDS, DATA_DIR, BASE_DIR
from ..models.schemas import (
    PavementExtractionResponse,
    HeatmapGenerationResponse,
    GeoBounds
)
from ..utils.pavement_extractor import extract_pavement_markings
from ..utils.heatmap_generator import generate_traffic_heatmap, create_demo_heatmap
from ..utils.satellite_downloader import SatelliteDataManager
from ..utils.traffic_analyzer import process_traffic_video

router = APIRouter()

# Initialize satellite data manager
satellite_manager = SatelliteDataManager(str(DATA_DIR / "satellite"))


# =============================================================================
# Satellite Data Endpoints
# =============================================================================

@router.get("/satellite/locations")
async def list_satellite_locations():
    """List available predefined satellite locations."""
    return {
        "locations": satellite_manager.list_locations(),
        "message": "Use location key with /satellite/download endpoint"
    }


@router.post("/satellite/download")
async def download_satellite(
    location: str = Form("times_square_nyc"),
    custom_min_lat: Optional[float] = Form(None),
    custom_max_lat: Optional[float] = Form(None),
    custom_min_lon: Optional[float] = Form(None),
    custom_max_lon: Optional[float] = Form(None),
    zoom: int = Form(18)
):
    """Download satellite imagery for a location."""
    try:
        if all([custom_min_lat, custom_max_lat, custom_min_lon, custom_max_lon]):
            image_path, bounds = satellite_manager.download_custom_area(
                min_lat=custom_min_lat,
                max_lat=custom_max_lat,
                min_lon=custom_min_lon,
                max_lon=custom_max_lon,
                name="custom_area",
                zoom=zoom
            )
            location_name = "Custom Area"
        else:
            if location not in satellite_manager.LOCATIONS:
                raise HTTPException(status_code=400, detail=f"Unknown location")
            image_path, bounds = satellite_manager.download_location(location)
            location_name = satellite_manager.LOCATIONS[location]["name"]
        
        return {
            "success": True,
            "message": f"Downloaded satellite imagery for {location_name}",
            "image_path": image_path,
            "bounds": bounds.to_dict(),
            "location": location_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-footage")
async def process_existing_footage(
    grid_size: int = Form(50),
    frame_sample_rate: int = Form(2),
    max_frames: Optional[int] = Form(500),
    min_lat: Optional[float] = Form(None),
    max_lat: Optional[float] = Form(None),
    min_lon: Optional[float] = Form(None),
    max_lon: Optional[float] = Form(None)
):
    """Process the pre-loaded footage from the footage folder."""
    footage_dir = BASE_DIR.parent / "footage"
    video_files = list(footage_dir.glob("*.mov")) + list(footage_dir.glob("*.mp4")) + list(footage_dir.glob("*.avi"))
    
    if not video_files:
        raise HTTPException(status_code=404, detail="No video files found in footage folder")
    
    video_path = str(video_files[0])
    geo_bounds = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon} if all([min_lat, max_lat, min_lon, max_lon]) else DEFAULT_BOUNDS
    
    try:
        result = process_traffic_video(
            video_path=video_path,
            output_path=str(HEATMAP_OUTPUT_PATH),
            geo_bounds=geo_bounds,
            grid_size=grid_size,
            frame_sample_rate=frame_sample_rate,
            max_frames=max_frames
        )
        return {
            "success": True,
            "message": f"Processed footage: {Path(video_path).name}",
            "heatmap_path": result["output_path"],
            "statistics": result["statistics"],
            "processing_time": result["processing_time"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-complete-pipeline")
async def run_complete_pipeline(
    location: str = Form("times_square_nyc"),
    process_video: bool = Form(True),
    threshold: int = Form(200),
    grid_size: int = Form(50),
    max_frames: Optional[int] = Form(500)
):
    """Run complete pipeline: satellite download, pavement extraction, traffic analysis."""
    results = {"satellite": None, "pavement": None, "traffic": None, "success": True}
    start_time = time.time()
    
    try:
        # Step 1: Download satellite imagery
        loc_info = satellite_manager.get_location(location)
        if not loc_info:
            raise HTTPException(status_code=400, detail=f"Unknown location: {location}")
        
        image_path, bounds = satellite_manager.download_location(location)
        results["satellite"] = {"image_path": image_path, "bounds": bounds.to_dict(), "location": loc_info["name"]}
        
        # Step 2: Extract pavement markings
        pavement_result = extract_pavement_markings(
            image_path=image_path, output_path=str(GEOJSON_OUTPUT_PATH),
            threshold=threshold, min_area=100, geo_bounds=bounds.to_dict()
        )
        results["pavement"] = pavement_result
        
        # Step 3: Process CCTV footage - ALWAYS use real video
        footage_dir = BASE_DIR.parent / "footage"
        video_files = list(footage_dir.glob("*.mov")) + list(footage_dir.glob("*.mp4"))
        
        if not video_files:
            raise HTTPException(
                status_code=400, 
                detail=f"No video files found in {footage_dir}. Please add your CCTV footage (.mov or .mp4) to the footage folder."
            )
        
        logger.info(f"Processing real video: {video_files[0]}")
        traffic_result = process_traffic_video(
            video_path=str(video_files[0]), 
            output_path=str(HEATMAP_OUTPUT_PATH),
            geo_bounds=bounds.to_dict(), 
            grid_size=grid_size,
            frame_sample_rate=2, 
            max_frames=max_frames
        )
        results["traffic"] = {
            "video_file": str(video_files[0].name),
            "statistics": traffic_result["statistics"], 
            "processing_time": traffic_result["processing_time"]
        }
        
        results["total_processing_time"] = time.time() - start_time
        results["bounds"] = bounds.to_dict()
        
        # Ensure all numeric values are JSON serializable
        results["pavement_features"] = pavement_result.get("feature_count", 0)
        results["frames_processed"] = traffic_result.get("statistics", {}).get("frames_processed", 0)
        
        return JSONResponse(content=results)
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        
        # Log full error
        logger.error(f"Pipeline error: {error_msg}")
        logger.error(f"Traceback: {traceback_str}")
        
        # Print to console for debugging
        print(f"\n❌ PIPELINE ERROR: {error_msg}")
        print(f"Traceback:\n{traceback_str}\n")
        
        # Return clean JSON error with proper string
        raise HTTPException(
            status_code=500, 
            detail=f"Pipeline failed: {error_msg}"
        )


# =============================================================================
# Original Endpoints (updated)
# =============================================================================


@router.post("/extract-pavement", response_model=PavementExtractionResponse)
async def extract_pavement(
    file: UploadFile = File(...),
    threshold: int = Form(200),
    min_area: int = Form(100),
    min_lat: Optional[float] = Form(None),
    max_lat: Optional[float] = Form(None),
    min_lon: Optional[float] = Form(None),
    max_lon: Optional[float] = Form(None)
):
    """
    Extract pavement markings from uploaded satellite imagery.
    
    This endpoint accepts a QGIS-exported satellite image and processes it
    to detect pavement markings (lanes, crosswalks, arrows, etc.).
    
    Parameters:
    - file: Satellite image file (PNG, TIFF, JPG)
    - threshold: Brightness threshold for detection (0-255)
    - min_area: Minimum contour area to consider
    - min_lat, max_lat, min_lon, max_lon: Geographic bounds for georeferencing
    
    Returns:
    - GeoJSON file path with detected markings
    - Feature count and processing statistics
    """
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.geotiff'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"satellite_image{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Prepare geo bounds
    geo_bounds = None
    if all([min_lat, max_lat, min_lon, max_lon]):
        geo_bounds = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        }
    else:
        # Use default bounds for demo
        geo_bounds = DEFAULT_BOUNDS
    
    # Process image
    try:
        result = extract_pavement_markings(
            image_path=str(upload_path),
            output_path=str(GEOJSON_OUTPUT_PATH),
            threshold=threshold,
            min_area=min_area,
            geo_bounds=geo_bounds
        )
        
        return PavementExtractionResponse(
            success=True,
            message="Pavement markings extracted successfully",
            geojson_path=result["geojson_path"],
            feature_count=result["feature_count"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@router.post("/generate-heatmap", response_model=HeatmapGenerationResponse)
async def generate_heatmap(
    file: UploadFile = File(...),
    grid_size: int = Form(50),
    frame_sample_rate: int = Form(5),
    min_lat: Optional[float] = Form(None),
    max_lat: Optional[float] = Form(None),
    min_lon: Optional[float] = Form(None),
    max_lon: Optional[float] = Form(None)
):
    """
    Generate traffic heatmap from CCTV video.
    
    This endpoint accepts a video file and processes it to detect
    traffic patterns and generate a spatial heatmap.
    
    Parameters:
    - file: Video file (MP4, AVI, MOV)
    - grid_size: Resolution of the heatmap grid (10-200)
    - frame_sample_rate: Process every Nth frame (1-30)
    - min_lat, max_lat, min_lon, max_lon: Geographic bounds
    
    Returns:
    - Heatmap JSON file path
    - Processing statistics
    """
    # Validate file type
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"cctv_video{file_ext}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Prepare geo bounds
    geo_bounds = None
    if all([min_lat, max_lat, min_lon, max_lon]):
        geo_bounds = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        }
    else:
        geo_bounds = DEFAULT_BOUNDS
    
    # Process video
    try:
        result = generate_traffic_heatmap(
            video_path=str(upload_path),
            output_path=str(HEATMAP_OUTPUT_PATH),
            grid_size=grid_size,
            frame_sample_rate=frame_sample_rate,
            geo_bounds=geo_bounds
        )
        
        return HeatmapGenerationResponse(
            success=True,
            message="Traffic heatmap generated successfully",
            heatmap_path=result["heatmap_path"],
            total_frames_processed=result["total_frames_processed"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Heatmap generation failed: {str(e)}"
        )


@router.post("/generate-demo-heatmap", response_model=HeatmapGenerationResponse)
async def generate_demo_heatmap_endpoint(
    grid_size: int = Form(50),
    min_lat: Optional[float] = Form(None),
    max_lat: Optional[float] = Form(None),
    min_lon: Optional[float] = Form(None),
    max_lon: Optional[float] = Form(None)
):
    """
    Generate a demo heatmap without video input.
    
    Creates synthetic traffic data for testing and demonstration purposes.
    """
    geo_bounds = None
    if all([min_lat, max_lat, min_lon, max_lon]):
        geo_bounds = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        }
    
    try:
        result = create_demo_heatmap(
            output_path=str(HEATMAP_OUTPUT_PATH),
            grid_size=grid_size,
            geo_bounds=geo_bounds
        )
        
        return HeatmapGenerationResponse(
            success=True,
            message="Demo heatmap generated successfully",
            heatmap_path=result["heatmap_path"],
            total_frames_processed=0,
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Demo heatmap generation failed: {str(e)}"
        )


@router.get("/get-geojson")
async def get_geojson():
    """
    Retrieve the extracted pavement markings GeoJSON.
    
    Returns the most recently generated GeoJSON file containing
    detected pavement markings from real satellite imagery.
    """
    if not GEOJSON_OUTPUT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No GeoJSON data available. Please run the pipeline first to process satellite imagery."
        )
    
    try:
        with open(GEOJSON_OUTPUT_PATH, 'r') as f:
            geojson_data = json.load(f)
        return JSONResponse(content=geojson_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read GeoJSON: {str(e)}"
        )


@router.get("/get-heatmap")
async def get_heatmap():
    """
    Retrieve the traffic heatmap data.
    
    Returns the heatmap generated from real CCTV footage.
    """
    if not HEATMAP_OUTPUT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No heatmap data available. Please run the pipeline first to process your CCTV footage."
        )
    
    try:
        with open(HEATMAP_OUTPUT_PATH, 'r') as f:
            heatmap_data = json.load(f)
        return JSONResponse(content=heatmap_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read heatmap: {str(e)}"
        )


@router.get("/get-geojson-file")
async def get_geojson_file():
    """
    Download the GeoJSON file directly.
    """
    if not GEOJSON_OUTPUT_PATH.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found")
    
    return FileResponse(
        path=str(GEOJSON_OUTPUT_PATH),
        media_type="application/geo+json",
        filename="pavement_markings.geojson"
    )


@router.get("/get-heatmap-file")
async def get_heatmap_file():
    """
    Download the heatmap JSON file directly.
    """
    if not HEATMAP_OUTPUT_PATH.exists():
        raise HTTPException(status_code=404, detail="Heatmap file not found")
    
    return FileResponse(
        path=str(HEATMAP_OUTPUT_PATH),
        media_type="application/json",
        filename="traffic_heatmap.json"
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mapathon-backend"}


def create_demo_geojson() -> dict:
    """Create demo GeoJSON data for testing."""
    features = []
    
    # Demo lane lines
    base_lat = DEFAULT_BOUNDS["min_lat"]
    base_lon = DEFAULT_BOUNDS["min_lon"]
    lat_range = DEFAULT_BOUNDS["max_lat"] - DEFAULT_BOUNDS["min_lat"]
    lon_range = DEFAULT_BOUNDS["max_lon"] - DEFAULT_BOUNDS["min_lon"]
    
    # Vertical lane lines
    for i in range(3):
        x_offset = 0.3 + i * 0.2
        features.append({
            "type": "Feature",
            "id": len(features),
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [base_lon + x_offset * lon_range, base_lat + 0.1 * lat_range],
                    [base_lon + x_offset * lon_range, base_lat + 0.9 * lat_range]
                ]
            },
            "properties": {
                "marking_type": "lane_line",
                "confidence": 0.85
            }
        })
    
    # Horizontal crosswalk
    for i in range(5):
        y_offset = 0.48 + i * 0.01
        features.append({
            "type": "Feature",
            "id": len(features),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [base_lon + 0.25 * lon_range, base_lat + y_offset * lat_range],
                    [base_lon + 0.75 * lon_range, base_lat + y_offset * lat_range],
                    [base_lon + 0.75 * lon_range, base_lat + (y_offset + 0.008) * lat_range],
                    [base_lon + 0.25 * lon_range, base_lat + (y_offset + 0.008) * lat_range],
                    [base_lon + 0.25 * lon_range, base_lat + y_offset * lat_range]
                ]]
            },
            "properties": {
                "marking_type": "crosswalk",
                "confidence": 0.92
            }
        })
    
    # Stop line
    features.append({
        "type": "Feature",
        "id": len(features),
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [base_lon + 0.25 * lon_range, base_lat + 0.45 * lat_range],
                [base_lon + 0.45 * lon_range, base_lat + 0.45 * lat_range],
                [base_lon + 0.45 * lon_range, base_lat + 0.46 * lat_range],
                [base_lon + 0.25 * lon_range, base_lat + 0.46 * lat_range],
                [base_lon + 0.25 * lon_range, base_lat + 0.45 * lat_range]
            ]]
        },
        "properties": {
            "marking_type": "stop_line",
            "confidence": 0.88
        }
    })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_features": len(features),
            "geo_bounds": DEFAULT_BOUNDS,
            "demo": True
        }
    }
