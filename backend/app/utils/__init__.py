# Utils package
from .pavement_extractor import PavementExtractor, extract_pavement_markings
from .heatmap_generator import HeatmapGenerator, generate_traffic_heatmap, create_demo_heatmap
from .satellite_downloader import SatelliteDataManager, TileDownloader, download_satellite_data
from .traffic_analyzer import TrafficVideoProcessor, VehicleDetector, VehicleTracker, process_traffic_video

# Optional torch-based imports (only load if torch is available)
try:
    from .ai_pavement_model import PavementDetector, PavementSegmentationModel, LightweightPavementModel
except ImportError:
    # torch not installed, skip AI model (using sklearn classifier instead)
    PavementDetector = None
    PavementSegmentationModel = None
    LightweightPavementModel = None
