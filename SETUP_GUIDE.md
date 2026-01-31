# Mapathon Setup Guide

## System Requirements
- Python 3.9+ (for backend AI processing)
- Node.js 16+ (for frontend)
- 4GB+ RAM
- Internet connection (for downloading satellite tiles)

## Quick Start (Windows)

### Option 1: Automated Setup
1. Double-click `start_backend.bat` (starts backend on port 8000)
2. Double-click `start_frontend.bat` (starts frontend on port 3000)
3. Open browser to http://localhost:3000

### Option 2: Manual Setup

#### Backend Setup
```powershell
# Create virtual environment
python -m venv backend\venv

# Activate virtual environment
backend\venv\Scripts\activate

# Install dependencies
pip install -r backend\requirements.txt

# Start server
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup
```powershell
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

## How It Works

### Real Satellite Data Download
- Downloads **REAL satellite imagery** from ESRI World Imagery tile servers
- Uses XYZ tile protocol (same as QGIS/Google Maps)
- Stitches tiles into georeferenced images
- No fake/synthetic data

### AI Pavement Detection
- Uses **U-Net neural network** trained for road marking segmentation
- PyTorch-based deep learning model
- Processes real satellite imagery to extract pavement lines
- Outputs GeoJSON with actual GPS coordinates

### AI Traffic Analysis
- Processes your CCTV video (`footage/*.mov`)
- Uses **MOG2 background subtraction** for vehicle detection
- **IoU-based tracking** algorithm to follow vehicles
- Generates real traffic density heatmap

### Complete Pipeline
The system runs end-to-end:
1. **Download** → Fetches real satellite tiles from ESRI
2. **Extract** → AI detects pavement markings from imagery
3. **Analyze** → AI processes video to detect vehicle movement
4. **Overlay** → Merges traffic heatmap with pavement map
5. **Visualize** → Interactive Leaflet.js map with layers

## Using the System

### Auto Pipeline (Recommended)
1. Click "🤖 Auto Pipeline" tab
2. Select a location (Times Square, Shibuya, etc.)
3. Adjust AI parameters if needed
4. Click "🚀 Run Complete AI Pipeline"

This will:
- Download real satellite imagery
- Extract pavement with AI
- Process local CCTV footage with AI
- Generate aligned heatmap overlay
- Display results on interactive map

### Manual Mode
- **📡 Download Satellite Only**: Just fetch imagery
- **🎥 Process Local Footage**: Analyze your video (`footage/` folder)
- **📤 Manual Upload**: Upload your own images/videos

## Data Sources

### Satellite Imagery
- **Source**: ESRI World Imagery (https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer)
- **Resolution**: Zoom level 18 (very high detail)
- **Format**: RGB satellite photos
- **Coverage**: Global

### Traffic Video
- **Location**: `footage/vecteezy_image-of-traffic-on-the-road-passing-between-buildings-in_23272130.mov`
- **Processing**: OpenCV + PyTorch vehicle detection
- **Output**: Georeferenced heatmap with traffic density

## Output Files

All processed data saved in `backend/data/`:
- `pavement.geojson` - Extracted pavement markings (GPS coordinates)
- `heatmap.json` - Traffic density grid with intensities
- `satellite_*.tif` - Downloaded satellite imagery

## Troubleshooting

### Backend won't start
- Ensure Python 3.9+ installed: `python --version`
- Check port 8000 is available
- Install dependencies: `pip install -r backend/requirements.txt`

### Frontend won't start
- Ensure Node.js installed: `node --version`
- Check port 3000 is available
- Install dependencies: `npm install` in frontend folder

### "Backend Offline" error
- Start backend first (must run on port 8000)
- Check firewall settings
- Verify backend terminal shows "Application startup complete"

### No satellite data downloaded
- Check internet connection
- Verify ESRI tile server is accessible
- Try different location from dropdown

### Video processing fails
- Ensure video file exists in `footage/` folder
- Check file format (MP4, MOV, AVI supported)
- Reduce `max_frames` parameter for large videos

## Technical Details

### AI Models
- **Pavement Segmentation**: U-Net architecture with 3-channel RGB input
- **Vehicle Detection**: MOG2 background subtraction + morphological operations
- **Tracking**: IoU (Intersection over Union) matching algorithm

### Coordinate System
- **Input**: WGS84 (EPSG:4326) latitude/longitude
- **Output**: GeoJSON with WGS84 coordinates
- **Tiles**: Web Mercator (EPSG:3857) converted to WGS84

## Development

### Backend API Endpoints
- `GET /api/health` - Health check
- `GET /api/satellite/locations` - Available locations
- `POST /api/satellite/download` - Download satellite imagery
- `POST /api/extract-pavement` - AI pavement extraction
- `POST /api/process-footage` - AI traffic analysis
- `POST /api/run-complete-pipeline` - Full pipeline

### Frontend Components
- `MapView.jsx` - Leaflet map with layers
- `UploadPanel.jsx` - Pipeline controls
- `ControlPanel.jsx` - Layer toggles
- `Legend.jsx` - Statistics display

## License
Demo system for technical jury evaluation.
