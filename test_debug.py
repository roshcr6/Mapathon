"""
Comprehensive test and debug script for the mapathon pipeline.
Runs each component individually to find errors.
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test all imports work."""
    print("=" * 60)
    print("STEP 1: Testing imports...")
    print("=" * 60)
    
    try:
        from app.utils.pavement_extractor import PavementExtractor, extract_pavement_markings
        print("✅ PavementExtractor imported")
    except Exception as e:
        print(f"❌ PavementExtractor import failed: {e}")
        return False
    
    try:
        from app.utils.traffic_analyzer import process_traffic_video
        print("✅ traffic_analyzer imported")
    except Exception as e:
        print(f"❌ traffic_analyzer import failed: {e}")
        return False
    
    try:
        from app.utils.satellite_downloader import SatelliteDataManager
        print("✅ SatelliteDataManager imported")
    except Exception as e:
        print(f"❌ SatelliteDataManager import failed: {e}")
        return False
    
    try:
        from app.utils.ai_trainer import PavementMarkingClassifier
        print("✅ AI Classifier imported")
    except Exception as e:
        print(f"❌ AI Classifier import failed: {e}")
        return False
    
    return True

def test_satellite_download():
    """Test satellite download."""
    print("\n" + "=" * 60)
    print("STEP 2: Testing satellite download...")
    print("=" * 60)
    
    try:
        from app.utils.satellite_downloader import SatelliteDataManager
        
        data_dir = Path(__file__).parent / "backend" / "data" / "satellite"
        manager = SatelliteDataManager(str(data_dir))
        
        print(f"Available locations: {[loc['id'] for loc in manager.list_locations()]}")
        
        image_path, bounds = manager.download_location("times_square_nyc")
        print(f"✅ Downloaded satellite image: {image_path}")
        print(f"   Bounds: {bounds.to_dict()}")
        
        return image_path, bounds
    except Exception as e:
        import traceback
        print(f"❌ Satellite download failed: {e}")
        traceback.print_exc()
        return None, None

def test_pavement_extraction(image_path, bounds):
    """Test pavement extraction."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing pavement extraction...")
    print("=" * 60)
    
    if image_path is None:
        print("⚠️ Skipping - no image available")
        return False
    
    try:
        from app.utils.pavement_extractor import extract_pavement_markings
        
        output_path = Path(__file__).parent / "backend" / "data" / "outputs" / "test_pavement.geojson"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = extract_pavement_markings(
            image_path=image_path,
            output_path=str(output_path),
            threshold=220,
            min_area=100,
            geo_bounds=bounds.to_dict()
        )
        
        print(f"✅ Pavement extraction successful!")
        print(f"   Features found: {result.get('feature_count', 0)}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   Output: {result.get('geojson_path', 'N/A')}")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Pavement extraction failed: {e}")
        traceback.print_exc()
        return False

def test_video_processing(bounds):
    """Test video processing."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing video processing...")
    print("=" * 60)
    
    footage_dir = Path(__file__).parent / "footage"
    video_files = list(footage_dir.glob("*.mov")) + list(footage_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"⚠️ No video files found in {footage_dir}")
        return False
    
    print(f"Found video: {video_files[0].name}")
    
    try:
        from app.utils.traffic_analyzer import process_traffic_video
        
        output_path = Path(__file__).parent / "backend" / "data" / "outputs" / "test_heatmap.json"
        
        result = process_traffic_video(
            video_path=str(video_files[0]),
            output_path=str(output_path),
            geo_bounds=bounds.to_dict() if bounds else {"min_lat": 40.758, "max_lat": 40.760, "min_lon": -73.986, "max_lon": -73.984},
            grid_size=50,
            frame_sample_rate=2,
            max_frames=100  # Quick test
        )
        
        print(f"✅ Video processing successful!")
        print(f"   Frames processed: {result.get('statistics', {}).get('frames_processed', 0)}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Video processing failed: {e}")
        traceback.print_exc()
        return False

def test_ai_model():
    """Test AI model loading and prediction."""
    print("\n" + "=" * 60)
    print("STEP 5: Testing AI model...")
    print("=" * 60)
    
    model_path = Path(__file__).parent / "backend" / "data" / "trained_model.pkl"
    
    if not model_path.exists():
        print(f"⚠️ No trained model found at {model_path}")
        print("   Run: python train_ai.py")
        return False
    
    try:
        from app.utils.ai_trainer import PavementMarkingClassifier
        import numpy as np
        import cv2
        
        classifier = PavementMarkingClassifier(str(model_path))
        print(f"✅ AI model loaded")
        
        # Create a test contour (rectangle)
        test_contour = np.array([
            [[10, 10]], [[100, 10]], [[100, 20]], [[10, 20]]
        ], dtype=np.int32)
        
        prediction = classifier.predict(test_contour)
        print(f"   Test prediction: {prediction}")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ AI model test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("MAPATHON PIPELINE DIAGNOSTIC TEST")
    print("=" * 60 + "\n")
    
    # Run all tests
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ FAILED: Fix import errors first!")
        return
    
    image_path, bounds = test_satellite_download()
    
    pavement_ok = test_pavement_extraction(image_path, bounds)
    
    video_ok = test_video_processing(bounds)
    
    ai_ok = test_ai_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Imports:           {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Satellite:         {'✅ PASS' if image_path else '❌ FAIL'}")
    print(f"Pavement Extract:  {'✅ PASS' if pavement_ok else '❌ FAIL'}")
    print(f"Video Process:     {'✅ PASS' if video_ok else '❌ FAIL'}")
    print(f"AI Model:          {'✅ PASS' if ai_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if all([imports_ok, image_path, pavement_ok, video_ok, ai_ok]):
        print("\n✅ ALL TESTS PASSED! Pipeline should work.")
    else:
        print("\n❌ Some tests failed. Fix the errors above.")


if __name__ == "__main__":
    main()
