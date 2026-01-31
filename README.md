# 🗺️ Mapathon - AI-Powered Pavement Marking Extraction & Traffic Heatmap System

A complete demo system that uses AI to detect and extract pavement markings from satellite imagery and generate traffic heatmaps from CCTV video footage.

![System Architecture](https://via.placeholder.com/800x400?text=Mapathon+System+Architecture)

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [QGIS Setup & Data Acquisition](#qgis-setup--data-acquisition)
5. [Backend Setup](#backend-setup)
6. [Frontend Setup](#frontend-setup)
7. [Running the Demo](#running-the-demo)
8. [API Documentation](#api-documentation)
9. [Project Structure](#project-structure)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Mapathon is a prototype geospatial AI system that demonstrates:

1. **Pavement Marking Extraction**: Uses computer vision to detect lane lines, crosswalks, arrows, and other road markings from satellite/aerial imagery exported from QGIS.

2. **Traffic Heatmap Generation**: Processes CCTV video footage to detect motion patterns and generate spatial heatmaps showing traffic density.

3. **Web-Based Visualization**: Displays results on an interactive map with layer controls, overlaying extracted markings and heatmaps on satellite imagery.

---

## ✨ Features

### AI Pipeline
- ✅ Automatic pavement marking detection using OpenCV
- ✅ Contour extraction and classification (lane lines, crosswalks, stop lines, arrows)
- ✅ GeoJSON output with confidence scores
- ✅ Motion detection using MOG2 background subtraction
- ✅ Traffic density heatmap generation
- ✅ Georeferenced output aligned with satellite imagery

### Web Interface
- ✅ Interactive map with Leaflet.js
- ✅ Satellite imagery base layer (ESRI World Imagery)
- ✅ Vector overlay for pavement markings
- ✅ Canvas-based heatmap rendering
- ✅ Layer toggle controls
- ✅ File upload for custom data processing

---

## 💻 System Requirements

### Software Requirements
- **Python**: 3.9 or higher
- **Node.js**: 18.x or higher
- **QGIS**: 3.28 LTS or higher (for data acquisition only)

### Hardware Recommendations
- RAM: 8GB minimum, 16GB recommended
- Storage: 2GB for application and sample data
- GPU: Optional (CPU processing supported)

---

## 🛰️ QGIS Setup & Data Acquisition

QGIS is used **only for preprocessing** to export georeferenced satellite imagery. The system does NOT depend on QGIS at runtime.

### Step 1: Install QGIS

1. Download QGIS LTS from: https://qgis.org/en/site/forusers/download.html
2. Install with default options
3. Launch QGIS Desktop

### Step 2: Add Satellite Imagery Layer

1. Open QGIS and create a new project
2. In the **Browser Panel** (left side), expand **XYZ Tiles**
3. If you don't see satellite options, add them manually:

#### Add ESRI World Imagery:
1. Right-click on **XYZ Tiles** → **New Connection**
2. Enter the following:
   - **Name**: `ESRI World Imagery`
   - **URL**: `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`
3. Click **OK**
4. Double-click **ESRI World Imagery** to add it to the map

#### Alternative - Google Satellite:
1. Right-click on **XYZ Tiles** → **New Connection**
2. Enter:
   - **Name**: `Google Satellite`
   - **URL**: `https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}`
3. Click **OK**

### Step 3: Navigate to Target Location

1. Use the zoom and pan tools to navigate to a road intersection
2. Recommended zoom level: **18-19** for best road marking visibility
3. Suggested test locations:
   - **Times Square, NYC**: 40.758896, -73.985130
   - **Tokyo Shibuya Crossing**: 35.659489, 139.700573
   - **Any major highway interchange**

### Step 4: Export Georeferenced Image

1. Go to **Project** → **Import/Export** → **Export Map to Image**
2. Configure export settings:
   - **Extent**: Choose "Map Canvas Extent" or draw custom extent
   - **Resolution**: 300 DPI recommended
   - **Format**: GeoTIFF (.tif) for georeferenced output, or PNG
3. Click **Save** and choose output location

#### For Georeferenced TIFF:
```
File → Export → Export Map to Image
- Enable "Append georeference information"
- Save as .tif format
```

### Step 5: Note the Geographic Bounds

Record the bounding box coordinates for your export:
- **Min Latitude**: (bottom)
- **Max Latitude**: (top)
- **Min Longitude**: (left)
- **Max Longitude**: (right)

You'll enter these when processing the image in the web interface.

---

## ⚙️ Backend Setup

### Step 1: Create Python Virtual Environment

Open a terminal in the `backend` folder:

```powershell
cd c:\Users\lenovo\OneDrive\Desktop\mapathon\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate

# Or for PowerShell:
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install PyTorch (if not included)

For CPU only:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA GPU support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Run the Backend Server

```powershell
python run.py
```

Or using uvicorn directly:
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

---

## 🌐 Frontend Setup

### Step 1: Install Node.js Dependencies

Open a new terminal in the `frontend` folder:

```powershell
cd c:\Users\lenovo\OneDrive\Desktop\mapathon\frontend

# Install dependencies
npm install
```

### Step 2: Run the Development Server

```powershell
npm run dev
```

The frontend will be available at: **http://localhost:3000**

---

## 🚀 Running the Demo

### Quick Start (Demo Mode)

1. **Start the Backend**:
   ```powershell
   cd backend
   .\venv\Scripts\Activate
   python run.py
   ```

2. **Start the Frontend** (new terminal):
   ```powershell
   cd frontend
   npm run dev
   ```

3. **Open Browser**: Navigate to http://localhost:3000

4. The system will automatically load **demo data** showing:
   - Sample pavement markings (lane lines, crosswalks)
   - Synthetic traffic heatmap

### Using Your Own Data

#### Process Satellite Image:

1. Export an image from QGIS (see [QGIS Setup](#qgis-setup--data-acquisition))
2. In the web interface, go to **Upload Data** → **Satellite Image**
3. Click **Select Image** and choose your exported file
4. Adjust the **Threshold** (higher = less sensitive detection)
5. Click **Extract Pavement Markings**

#### Process CCTV Video:

1. Obtain a traffic video file (MP4, AVI, MOV)
2. In the web interface, go to **Upload Data** → **CCTV Video**
3. Click **Select Video** and choose your file
4. Adjust **Grid Size** (higher = more detail) and **Sample Rate**
5. Click **Generate Heatmap**

---

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/extract-pavement` | Extract pavement markings from image |
| POST | `/api/generate-heatmap` | Generate heatmap from video |
| POST | `/api/generate-demo-heatmap` | Generate demo heatmap (no video needed) |
| GET | `/api/get-geojson` | Get extracted pavement GeoJSON |
| GET | `/api/get-heatmap` | Get traffic heatmap data |
| GET | `/api/health` | Health check |

### Extract Pavement Request

```bash
curl -X POST "http://localhost:8000/api/extract-pavement" \
  -F "file=@satellite_image.png" \
  -F "threshold=200" \
  -F "min_area=100" \
  -F "min_lat=40.7128" \
  -F "max_lat=40.7138" \
  -F "min_lon=-74.0060" \
  -F "max_lon=-74.0050"
```

### Generate Heatmap Request

```bash
curl -X POST "http://localhost:8000/api/generate-heatmap" \
  -F "file=@traffic_video.mp4" \
  -F "grid_size=50" \
  -F "frame_sample_rate=5" \
  -F "min_lat=40.7128" \
  -F "max_lat=40.7138" \
  -F "min_lon=-74.0060" \
  -F "max_lon=-74.0050"
```

### Response Examples

**GeoJSON Response:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[-74.005, 40.713], ...]]
      },
      "properties": {
        "marking_type": "lane_line",
        "confidence": 0.85,
        "area_pixels": 1250.5
      }
    }
  ],
  "metadata": {
    "total_features": 15,
    "geo_bounds": {...}
  }
}
```

**Heatmap Response:**
```json
{
  "bounds": {
    "min_lat": 40.7128,
    "max_lat": 40.7138,
    "min_lon": -74.0060,
    "max_lon": -74.0050
  },
  "grid_size": 50,
  "points": [
    {"lat": 40.713, "lon": -74.005, "intensity": 0.85},
    ...
  ],
  "statistics": {
    "max_intensity": 1.0,
    "min_intensity": 0.1,
    "mean_intensity": 0.45,
    "total_points": 200,
    "frames_processed": 150
  }
}
```

---

## 📁 Project Structure

```
mapathon/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration settings
│   │   ├── main.py                # FastAPI application
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py         # Pydantic models
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── api.py             # API endpoints
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── pavement_extractor.py  # AI pavement detection
│   │       └── heatmap_generator.py   # Traffic heatmap generation
│   ├── data/
│   │   ├── uploads/               # Uploaded files
│   │   └── outputs/               # Generated outputs
│   ├── requirements.txt
│   └── run.py                     # Server entry point
│
├── frontend/
│   ├── public/
│   │   └── vite.svg
│   ├── src/
│   │   ├── components/
│   │   │   ├── MapView.jsx        # Main map component
│   │   │   ├── ControlPanel.jsx   # Layer toggles
│   │   │   ├── UploadPanel.jsx    # File upload UI
│   │   │   └── Legend.jsx         # Map legend
│   │   ├── services/
│   │   │   └── api.js             # API client
│   │   ├── styles/
│   │   │   ├── index.css          # Global styles
│   │   │   └── App.css            # App layout
│   │   ├── App.jsx                # Main app component
│   │   └── main.jsx               # React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

## 🔧 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError: No module named 'cv2'`
```powershell
pip install opencv-python
```

**Problem**: `ModuleNotFoundError: No module named 'rasterio'`
```powershell
# Windows may need pre-built wheel
pip install rasterio --find-links https://girder.github.io/large_image_wheels
```

**Problem**: Port 8000 already in use
```powershell
# Find and kill the process
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port
uvicorn app.main:app --port 8001
```

### Frontend Issues

**Problem**: `npm ERR! ERESOLVE`
```powershell
npm install --legacy-peer-deps
```

**Problem**: Map not loading
- Check that the backend is running on port 8000
- Verify CORS is configured correctly
- Check browser console for errors

### QGIS Issues

**Problem**: XYZ Tiles not loading
- Check internet connection
- Try a different tile server
- Disable any proxy settings

**Problem**: Export resolution too low
- Increase DPI in export settings
- Zoom in more before exporting
- Use "Print Layout" for more control

---

## 📊 Technical Details

### Pavement Detection Algorithm

1. **Preprocessing**:
   - Convert to grayscale
   - CLAHE contrast enhancement
   - Gaussian blur for noise reduction
   - Bilateral filtering for edge preservation

2. **Detection**:
   - Threshold-based bright region detection
   - Canny edge detection
   - Morphological operations (open, close, dilate)

3. **Classification**:
   - Contour analysis
   - Aspect ratio calculation
   - Circularity measurement
   - Shape-based classification

### Heatmap Generation Algorithm

1. **Motion Detection**:
   - MOG2 background subtraction
   - Shadow removal
   - Morphological cleanup

2. **Aggregation**:
   - Grid-based accumulation
   - Frame sampling for efficiency
   - Gaussian smoothing

3. **Normalization**:
   - Min-max scaling
   - Intensity mapping to color gradient

---

## 📄 License

This project is provided as a demo for educational and evaluation purposes.

---

## 🤝 Support

For technical questions or issues, please create an issue in the project repository.

---

Built with ❤️ for the Mapathon Demo
