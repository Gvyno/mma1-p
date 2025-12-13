"""
Simple script to train the property price prediction model.

This script trains a Random Forest model on the processed property data
and saves it to the models directory for use in predictions.
"""

from pathlib import Path
from model import run_training


def main():
    print("=" * 60)
    print("Property Price Prediction Model - Training Script")
    print("=" * 60)
    print()
    
    # Define paths
    data_path = Path("data") / "properties_processed.csv"
    model_path = Path("models") / "random_forest.pkl"
    
    # Check if data file exists
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure the processed data file exists before training.")
        return
    
    print(f"Data file: {data_path}")
    print(f"Model will be saved to: {model_path}")
    print()
    print("Starting training...")
    print("-" * 60)
    
    # Train the model
    run_training(
        data_path=data_path,
        test_size=0.2,
        random_state=42,
        model_path=model_path
    )
    
    print("-" * 60)
    print()
    print("Training complete! Model saved successfully.")
    print(f"You can now use the model at: {model_path}")
    print()


if __name__ == "__main__":
    main()
