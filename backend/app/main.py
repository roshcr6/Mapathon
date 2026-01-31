"""
Mapathon Backend - FastAPI Application

A geospatial AI system for pavement marking extraction
and traffic heatmap generation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import CORS_ORIGINS, DATA_DIR, OUTPUT_DIR
from .routes.api import router as api_router

# Create FastAPI app
app = FastAPI(
    title="Mapathon Backend API",
    description="""
    AI-powered geospatial system for:
    - Extracting pavement markings from satellite imagery
    - Generating traffic heatmaps from CCTV footage
    - Serving GeoJSON and heatmap data for visualization
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for data access
if OUTPUT_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(OUTPUT_DIR)), name="data")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["API"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Mapathon Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "extract_pavement": "POST /api/extract-pavement",
            "generate_heatmap": "POST /api/generate-heatmap",
            "generate_demo_heatmap": "POST /api/generate-demo-heatmap",
            "get_geojson": "GET /api/get-geojson",
            "get_heatmap": "GET /api/get-heatmap",
            "health": "GET /api/health"
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Mapathon Backend started successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Mapathon Backend shutting down...")
