"""
Satellite Data Downloader Module

Downloads satellite/aerial imagery from various sources:
- ESRI World Imagery
- OpenStreetMap
- Google Satellite (for reference)

This module allows downloading georeferenced satellite tiles
for any location in the world without requiring QGIS at runtime.
"""

import os
import math
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
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
    
    @property
    def center(self) -> Tuple[float, float]:
        """Return center point (lat, lon)."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon
        }


class TileDownloader:
    """
    Downloads map tiles from XYZ tile servers.
    Supports ESRI, OSM, and other tile providers.
    """
    
    TILE_SERVERS = {
        "esri_satellite": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attribution": "ESRI World Imagery",
            "max_zoom": 19
        },
        "esri_streets": {
            "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            "attribution": "ESRI World Street Map",
            "max_zoom": 19
        },
        "osm": {
            "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attribution": "OpenStreetMap",
            "max_zoom": 19
        },
        "carto_light": {
            "url": "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
            "attribution": "CartoDB",
            "max_zoom": 19
        }
    }
    
    def __init__(self, tile_source: str = "esri_satellite"):
        """Initialize tile downloader with specified source."""
        if tile_source not in self.TILE_SERVERS:
            raise ValueError(f"Unknown tile source: {tile_source}")
        
        self.source = self.TILE_SERVERS[tile_source]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mapathon/1.0 (Geospatial AI Demo)'
        })
    
    @staticmethod
    def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    @staticmethod
    def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to latitude/longitude (NW corner)."""
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return (lat, lon)
    
    @staticmethod
    def get_tile_bounds(x: int, y: int, zoom: int) -> GeoBounds:
        """Get geographic bounds of a tile."""
        nw_lat, nw_lon = TileDownloader.tile_to_lat_lon(x, y, zoom)
        se_lat, se_lon = TileDownloader.tile_to_lat_lon(x + 1, y + 1, zoom)
        return GeoBounds(
            min_lat=se_lat,
            max_lat=nw_lat,
            min_lon=nw_lon,
            max_lon=se_lon
        )
    
    def download_tile(self, x: int, y: int, zoom: int) -> Optional[Image.Image]:
        """Download a single tile."""
        url = self.source["url"].format(z=zoom, x=x, y=y)
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.warning(f"Failed to download tile ({x}, {y}, {zoom}): {e}")
            return None
    
    def download_area(
        self,
        bounds: GeoBounds,
        zoom: int = 18,
        output_path: Optional[str] = None
    ) -> Tuple[Image.Image, GeoBounds]:
        """
        Download all tiles covering a geographic area and stitch them together.
        
        Args:
            bounds: Geographic bounds to download
            zoom: Zoom level (18-19 recommended for road details)
            output_path: Optional path to save the image
            
        Returns:
            Tuple of (stitched image, actual bounds covered)
        """
        zoom = min(zoom, self.source["max_zoom"])
        
        # Get tile range
        min_x, max_y = self.lat_lon_to_tile(bounds.min_lat, bounds.min_lon, zoom)
        max_x, min_y = self.lat_lon_to_tile(bounds.max_lat, bounds.max_lon, zoom)
        
        # Ensure correct order
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y
        
        logger.info(f"Downloading tiles: x=[{min_x},{max_x}], y=[{min_y},{max_y}], zoom={zoom}")
        
        # Calculate image dimensions
        tile_size = 256
        width = (max_x - min_x + 1) * tile_size
        height = (max_y - min_y + 1) * tile_size
        
        # Create output image
        result = Image.new('RGB', (width, height))
        
        # Download and stitch tiles
        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        downloaded = 0
        
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                tile = self.download_tile(x, y, zoom)
                downloaded += 1
                
                if tile:
                    # Calculate position in output image
                    px = (x - min_x) * tile_size
                    py = (y - min_y) * tile_size
                    result.paste(tile, (px, py))
                
                # Rate limiting
                if downloaded % 10 == 0:
                    logger.info(f"Downloaded {downloaded}/{total_tiles} tiles...")
                    time.sleep(0.1)
        
        # Calculate actual bounds
        actual_bounds = GeoBounds(
            min_lat=self.tile_to_lat_lon(min_x, max_y + 1, zoom)[0],
            max_lat=self.tile_to_lat_lon(min_x, min_y, zoom)[0],
            min_lon=self.tile_to_lat_lon(min_x, min_y, zoom)[1],
            max_lon=self.tile_to_lat_lon(max_x + 1, min_y, zoom)[1]
        )
        
        # Save if path provided
        if output_path:
            result.save(output_path)
            logger.info(f"Saved satellite image to: {output_path}")
            
            # Save metadata
            meta_path = Path(output_path).with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    "bounds": actual_bounds.to_dict(),
                    "zoom": zoom,
                    "size": {"width": width, "height": height},
                    "source": self.source["attribution"],
                    "tiles": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
                }, f, indent=2)
        
        return result, actual_bounds


class SatelliteDataManager:
    """
    Manages satellite data for specific locations.
    Provides predefined locations and custom area downloads.
    """
    
    # Predefined locations with good road markings for demo
    LOCATIONS = {
        "times_square_nyc": {
            "name": "Times Square, New York",
            "bounds": GeoBounds(40.7565, 40.7595, -73.9890, -73.9840),
            "zoom": 19
        },
        "shibuya_tokyo": {
            "name": "Shibuya Crossing, Tokyo",
            "bounds": GeoBounds(35.6580, 35.6610, 139.6990, 139.7030),
            "zoom": 19
        },
        "champs_elysees_paris": {
            "name": "Champs-Élysées, Paris",
            "bounds": GeoBounds(48.8690, 48.8720, 2.3050, 2.3100),
            "zoom": 19
        },
        "highway_intersection": {
            "name": "Highway Intersection Sample",
            "bounds": GeoBounds(40.7480, 40.7520, -73.9920, -73.9870),
            "zoom": 18
        },
        "downtown_la": {
            "name": "Downtown Los Angeles",
            "bounds": GeoBounds(34.0480, 34.0520, -118.2550, -118.2490),
            "zoom": 19
        }
    }
    
    def __init__(self, data_dir: str):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = TileDownloader("esri_satellite")
    
    def get_location(self, location_key: str) -> Optional[Dict[str, Any]]:
        """Get predefined location info."""
        return self.LOCATIONS.get(location_key)
    
    def list_locations(self) -> list:
        """List all predefined locations as list of dicts."""
        return [{"id": k, "name": v["name"]} for k, v in self.LOCATIONS.items()]
    
    def download_location(self, location_key: str) -> Tuple[str, GeoBounds]:
        """
        Download satellite imagery for a predefined location.
        
        Returns:
            Tuple of (image path, bounds)
        """
        if location_key not in self.LOCATIONS:
            raise ValueError(f"Unknown location: {location_key}")
        
        loc = self.LOCATIONS[location_key]
        output_path = self.data_dir / f"{location_key}_satellite.png"
        
        logger.info(f"Downloading satellite imagery for: {loc['name']}")
        
        _, bounds = self.downloader.download_area(
            loc["bounds"],
            zoom=loc["zoom"],
            output_path=str(output_path)
        )
        
        return str(output_path), bounds
    
    def download_custom_area(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        name: str = "custom",
        zoom: int = 18
    ) -> Tuple[str, GeoBounds]:
        """
        Download satellite imagery for a custom area.
        
        Args:
            min_lat, max_lat, min_lon, max_lon: Geographic bounds
            name: Name for the output file
            zoom: Zoom level (18-19 recommended)
            
        Returns:
            Tuple of (image path, bounds)
        """
        bounds = GeoBounds(min_lat, max_lat, min_lon, max_lon)
        output_path = self.data_dir / f"{name}_satellite.png"
        
        logger.info(f"Downloading satellite imagery for custom area: {bounds}")
        
        _, actual_bounds = self.downloader.download_area(
            bounds,
            zoom=zoom,
            output_path=str(output_path)
        )
        
        return str(output_path), actual_bounds


def download_satellite_data(
    output_dir: str,
    location: str = "times_square_nyc"
) -> Dict[str, Any]:
    """
    Convenience function to download satellite data.
    
    Args:
        output_dir: Directory to save data
        location: Predefined location key or "custom"
        
    Returns:
        Dictionary with image path and bounds
    """
    manager = SatelliteDataManager(output_dir)
    
    try:
        image_path, bounds = manager.download_location(location)
        
        return {
            "success": True,
            "image_path": image_path,
            "bounds": bounds.to_dict(),
            "location_name": manager.LOCATIONS[location]["name"]
        }
    except Exception as e:
        logger.error(f"Failed to download satellite data: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test download
    result = download_satellite_data("./test_data", "times_square_nyc")
    print(json.dumps(result, indent=2))
