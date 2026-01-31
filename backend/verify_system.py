"""
System Verification Script
Checks all components are properly configured for real data processing
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.9"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} - NEED 3.9+")
        return False

def check_dependencies():
    """Check critical dependencies are installed"""
    print("\n✓ Checking dependencies...")
    required = [
        'fastapi',
        'uvicorn',
        'cv2',
        'torch',
        'torchvision',
        'PIL',
        'rasterio',
        'shapely',
        'geojson',
        'numpy',
        'scipy',
        'requests'
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_project_structure():
    """Verify project structure exists"""
    print("\n✓ Checking project structure...")
    base_dir = Path(__file__).parent
    
    required_paths = [
        'app/main.py',
        'app/config.py',
        'app/routes/api.py',
        'app/utils/satellite_downloader.py',
        'app/utils/ai_pavement_model.py',
        'app/utils/traffic_analyzer.py',
        'data/',
        'footage/'
    ]
    
    all_exist = True
    for path in required_paths:
        full_path = base_dir / path
        if full_path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} - MISSING")
            all_exist = False
    
    return all_exist

def check_footage():
    """Check if footage exists"""
    print("\n✓ Checking CCTV footage...")
    base_dir = Path(__file__).parent
    footage_dir = base_dir / 'footage'
    
    if not footage_dir.exists():
        print("  ✗ footage/ directory missing")
        return False
    
    videos = list(footage_dir.glob('*.mov')) + list(footage_dir.glob('*.mp4')) + list(footage_dir.glob('*.avi'))
    
    if videos:
        print(f"  ✓ Found {len(videos)} video file(s):")
        for video in videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"    - {video.name} ({size_mb:.1f} MB)")
        return True
    else:
        print("  ⚠ No video files found in footage/")
        return False

def check_data_directory():
    """Verify data output directory"""
    print("\n✓ Checking data directory...")
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("  ✓ Created data/ directory")
    else:
        print("  ✓ data/ directory exists")
    
    # Check write permissions
    test_file = data_dir / '.test'
    try:
        test_file.write_text('test')
        test_file.unlink()
        print("  ✓ Write permissions OK")
        return True
    except Exception as e:
        print(f"  ✗ Cannot write to data/ - {e}")
        return False

def check_satellite_connection():
    """Test connection to ESRI tile server"""
    print("\n✓ Checking satellite tile server connection...")
    try:
        import requests
        url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/18/77171/98305"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("  ✓ ESRI World Imagery server - ACCESSIBLE")
            print(f"    Downloaded test tile: {len(response.content)} bytes")
            return True
        else:
            print(f"  ✗ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Connection failed - {e}")
        return False

def check_ai_models():
    """Verify AI model components"""
    print("\n✓ Checking AI model components...")
    try:
        import torch
        import cv2
        
        # Check PyTorch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    ✓ CUDA available - GPU acceleration enabled")
        else:
            print(f"    ⚠ CUDA not available - using CPU (slower)")
        
        # Check OpenCV
        print(f"  ✓ OpenCV {cv2.__version__}")
        
        # Test MOG2 background subtractor
        mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        print("  ✓ MOG2 background subtractor initialized")
        
        return True
    except Exception as e:
        print(f"  ✗ AI components failed - {e}")
        return False

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("MAPATHON SYSTEM VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("CCTV Footage", check_footage),
        ("Data Directory", check_data_directory),
        ("Satellite Connection", check_satellite_connection),
        ("AI Models", check_ai_models)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} check crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ System ready for real data processing!")
        print("\nNext steps:")
        print("1. Start backend: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("2. Access API docs: http://localhost:8000/docs")
        print("3. Run complete pipeline via API or frontend")
    else:
        print("\n⚠ Some checks failed. Fix issues before proceeding.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
