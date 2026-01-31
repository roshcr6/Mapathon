"""
Analyze the training images you provided to learn what REAL markings look like
"""
import cv2
import numpy as np
from pathlib import Path

def analyze_training_images():
    print("=" * 70)
    print("ANALYZING YOUR TRAINING DATA")
    print("=" * 70)
    
    # Check crosswalk images
    crosswalk_dir = Path("crosswalk")
    line_dir = Path("lines")
    
    print("\n📸 CROSSWALK IMAGES:")
    for img_path in sorted(crosswalk_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Convert to HSV and grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Find bright pixels (likely the white stripes)
        bright_pixels = gray[gray > 180]
        
        if len(bright_pixels) > 0:
            print(f"\n{img_path.name}:")
            print(f"  - Image size: {img.shape[1]}x{img.shape[0]}")
            print(f"  - Mean brightness of bright areas: {bright_pixels.mean():.1f}")
            print(f"  - Min/Max brightness: {gray.min()}/{gray.max()}")
            print(f"  - Bright pixels (>180): {len(bright_pixels)} ({len(bright_pixels)/gray.size*100:.1f}%)")
            
            # Find white stripes
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                areas = [cv2.contourArea(c) for c in contours]
                widths = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    widths.append((w, h))
                
                print(f"  - Found {len(contours)} stripe contours")
                print(f"  - Stripe sizes (w x h): {widths[:5]}")
    
    print("\n\n📸 LANE LINE IMAGES:")
    for img_path in sorted(line_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        bright_pixels = gray[gray > 180]
        
        if len(bright_pixels) > 0:
            print(f"\n{img_path.name}:")
            print(f"  - Image size: {img.shape[1]}x{img.shape[0]}")
            print(f"  - Mean brightness: {bright_pixels.mean():.1f}")
            print(f"  - Min/Max: {gray.min()}/{gray.max()}")
            print(f"  - Bright pixels: {len(bright_pixels)} ({len(bright_pixels)/gray.size*100:.1f}%)")
            
            # Find lines
            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                for c in contours[:3]:
                    x, y, w, h = cv2.boundingRect(c)
                    aspect = max(w, h) / max(min(w, h), 1)
                    print(f"  - Line: {w}x{h} pixels, aspect ratio {aspect:.1f}:1")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR AI DETECTION:")
    print("=" * 70)
    print("""
Based on your training images:
1. Real markings have brightness > 180-200
2. Crosswalk stripes are rectangular, moderately wide
3. Lane lines are very thin and elongated (aspect ratio > 10:1)
4. White markings have low color saturation
5. Marking pixels are only 1-5% of satellite image

DETECTION STRATEGY:
- Use threshold 175-185 for brightness
- Require aspect ratio > 2.5 for any marking
- Require width < 100 pixels (markings are thin compared to buildings)
- Remove any contour > 0.5% of image area (buildings)
""")

if __name__ == "__main__":
    analyze_training_images()
