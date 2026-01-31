"""
Train AI Model Script

Run this to train the AI on your labeled data.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.utils.ai_trainer import train_ai_model

def main():
    # Paths - use the lines and crosswalk folders you provided
    base_dir = Path(__file__).parent
    training_data_dir = base_dir  # Parent contains lines/ and crosswalk/
    model_output = base_dir / "backend" / "data" / "trained_model.pkl"
    
    # Create model directory if needed
    model_output.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AI TRAINING - Pavement Marking Classifier")
    print("=" * 60)
    print(f"Training data: {training_data_dir}")
    print(f"Output model: {model_output}")
    print()
    
    # Train model
    classifier = train_ai_model(str(training_data_dir), str(model_output))
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_output}")
    print()
    print("The AI can now classify:")
    print("  - lane_line (from lines folder)")
    print("  - crosswalk (from crosswalk folder)")
    print()

if __name__ == "__main__":
    main()
