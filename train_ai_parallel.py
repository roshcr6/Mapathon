"""
PARALLEL AI TRAINING SCRIPT
Uses all CPU cores for maximum training power.

Run this from the mapathon directory:
    python train_ai_parallel.py

"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from pathlib import Path
from app.utils.ai_trainer_parallel import ParallelPavementClassifier, train_ai_parallel

def main():
    print("=" * 60)
    print("🚀 PARALLEL AI TRAINING - Maximum Power Mode")
    print("=" * 60)
    
    # Training data location
    training_dir = Path(__file__).parent
    output_model = training_dir / "backend" / "data" / "trained_model.pkl"
    
    print(f"Training data: {training_dir}")
    print(f"Output model: {output_model}")
    print()
    
    # Ensure output directory exists
    output_model.parent.mkdir(parents=True, exist_ok=True)
    
    # Train with parallel processing
    classifier = train_ai_parallel(str(training_dir), str(output_model))
    
    print()
    print("=" * 60)
    print("✅ PARALLEL Training Complete!")
    print("=" * 60)
    print(f"Best model: {classifier.best_model_name}")
    print(f"Model saved to: {output_model}")
    print()
    print("The AI can now classify:")
    print("  - lane_line (from lines folder)")
    print("  - crosswalk (from crosswalk folder)")

if __name__ == "__main__":
    main()
