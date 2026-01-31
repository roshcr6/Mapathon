"""
Test AI Detection - Shows what the AI sees in the satellite image

This script:
1. Loads a satellite image
2. Runs the pavement detector
3. Shows debug visualization
4. Outputs detection statistics
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from pathlib import Path
import json

def test_detection():
    print("=" * 60)
    print("AI PAVEMENT DETECTION TEST")
    print("=" * 60)
    
    # Check for satellite image
    data_dir = Path(__file__).parent / "backend" / "data" / "satellite"
    image_files = list(data_dir.glob("*.png"))
    
    if not image_files:
        print("\n❌ No satellite image found!")
        print(f"   Looking in: {data_dir}")
        print("\nRun the pipeline first to download satellite imagery.")
        return
    
    image_path = str(image_files[0])
    print(f"\n📷 Found satellite image: {image_path}")
    
    # Load bounds from JSON
    json_files = list(data_dir.glob("*.json"))
    bounds = None
    if json_files:
        with open(json_files[0]) as f:
            bounds = json.load(f)
        print(f"📍 Geo bounds loaded: {bounds}")
    
    # Run detection
    print("\n🔍 Running Enhanced Pavement Detection...")
    
    from app.utils.pavement_detector_v2 import EnhancedPavementDetector, GeoBounds
    
    geo_bounds = None
    if bounds:
        # Handle nested bounds structure
        b = bounds.get('bounds', bounds)
        geo_bounds = GeoBounds(
            min_lat=b['min_lat'],
            max_lat=b['max_lat'],
            min_lon=b['min_lon'],
            max_lon=b['max_lon']
        )
    
    # Create detector with BALANCED settings
    detector = EnhancedPavementDetector(
        white_threshold=185,  # MEDIUM sensitivity
        min_area=50,  # Catch medium-sized markings
        geo_bounds=geo_bounds
    )
    
    # Run detection
    markings = detector.detect(image_path)
    
    print(f"\n✅ Detection Complete!")
    print(f"   Total markings found: {len(markings)}")
    
    # Count by type
    types = {}
    for m in markings:
        types[m.marking_type] = types.get(m.marking_type, 0) + 1
    
    print("\n📊 Breakdown by type:")
    for t, count in types.items():
        print(f"   - {t}: {count}")
    
    # Save visualization
    output_dir = Path(__file__).parent / "backend" / "data" / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    vis_path = str(output_dir / "ai_detection_visualization.png")
    detector.create_visualization(markings, vis_path)
    print(f"\n🖼️  Visualization saved: {vis_path}")
    
    # Save GeoJSON
    geojson_path = str(output_dir / "pavement_markings.geojson")
    geojson = detector.to_geojson(markings)
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"📄 GeoJSON saved: {geojson_path}")
    
    print("\n" + "=" * 60)
    print("HOW THE AI SEES THE IMAGE:")
    print("=" * 60)
    print("""
The AI detects pavement markings by looking for:

1. WHITE MARKINGS (most common):
   - High brightness (pixel value > 160-200)
   - Low color saturation (appears gray/white, not colored)
   - Found on dark road surface (asphalt = dark gray)

2. YELLOW MARKINGS:
   - Specific yellow hue in HSV color space
   - Center line markings, no parking zones

3. SHAPE ANALYSIS:
   - Lane lines: Very elongated (aspect ratio > 5:1)
   - Crosswalks: Rectangular bars (aspect ratio 2-5:1)
   - Stop lines: Wide and short

4. FALSE POSITIVE FILTERING:
   - Removes buildings (too large)
   - Removes noise (too small)
   - Removes non-elongated shapes (not line-like)

Check the visualization image to see exactly what was detected!
""")
    
    return markings

if __name__ == "__main__":
    test_detection()
