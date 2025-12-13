# Property Price Prediction Model - UI Developer Guide

## Training the Model

Before making predictions, you need to train the model:

```bash
python train_model.py
```

This will:
1. Load the processed property data from `data/properties_processed.csv`
2. Train a Random Forest model
3. Save the trained model to `models/random_forest.pkl`
4. Display training metrics and feature importances

**Expected Output:**
```
============================================================
Property Price Prediction Model - Training Script
============================================================

Data file: data\properties_processed.csv
Model will be saved to: models\random_forest.pkl

Starting training...
------------------------------------------------------------
RandomForestRegressor results
Train rows: 320, Test rows: 80
MAE : 45,231.12
RMSE: 67,890.34
R^2 : 0.8542
Top feature importances:
  indoor_sqm                     0.4521
  bathrooms                      0.1234
  bedrooms                       0.0987
  ...
------------------------------------------------------------

Training complete! Model saved successfully.
```

### Custom Training

For more control over training parameters:

```python
from pathlib import Path
from model import run_training

# Train with custom settings
run_training(
    data_path=Path("data") / "properties_processed.csv",
    test_size=0.2,        # 20% of data for testing
    random_state=42,      # For reproducibility
    model_path=Path("models") / "random_forest.pkl"
)
```

---

## Quick Start

### 1. Loading the Model

```python
from pathlib import Path
from model import RandomForestPriceModel

# Load the pre-trained model
model_path = Path("models") / "random_forest.pkl"
model = RandomForestPriceModel.load(model_path)
```

### 2. Making a Basic Prediction

```python
result = model.predict_from_manual_input(
    bedrooms=3,
    bathrooms=2,
    indoor_sqm=150
)

predicted_price = result['prediction']
print(f"Predicted Price: €{predicted_price:,.2f}")
```

## Required Input Fields

### Core Features (Required)
These are the minimum fields needed for a prediction:

| Field | Type | Description | Typical Range |
|-------|------|-------------|---------------|
| `bedrooms` | float | Number of bedrooms | 0-20 |
| `bathrooms` | float | Number of bathrooms | 0-15 |
| `indoor_sqm` | float | Indoor area in square meters | 0-2000 |

### Optional Features

#### Description Analysis
| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `desc_length` | int | Character count of property description | 0 |
| `desc_has_luxury` | bool | Description contains "luxury" or "luxurious" | False |
| `desc_has_exclusive` | bool | Description contains "exclusive" | False |
| `desc_has_stunning` | bool | Description contains "stunning" | False |

#### Image Analysis
| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `images_dir` | Path or None | Directory containing property images | None |

When `images_dir` is provided, the model automatically:
- Counts total images
- Detects greenery in images (gardens, outdoor spaces)
- Uses these features to improve prediction accuracy

#### Property Features
Pass additional property features using the `feature_flags` dictionary:

```python
feature_flags = {
    "pool": True,
    "garden": True,
    "terrace": True,
    "sea_views": False,
    "parking": True,
    # ... more features
}
```

Common features supported:
- pool
- garden
- terrace
- sea_views
- parking
- air_conditioning
- heating
- fireplace
- balcony
- elevator
- storage
- furnished

## Complete Example

```python
from pathlib import Path
from model import RandomForestPriceModel

# Load model
model = RandomForestPriceModel.load(Path("models") / "random_forest.pkl")

# Prepare input data
result = model.predict_from_manual_input(
    # Required fields
    bedrooms=4,
    bathrooms=3,
    indoor_sqm=250,
    
    # Optional: Description features
    desc_length=350,
    desc_has_luxury=True,
    desc_has_exclusive=False,
    desc_has_stunning=True,
    
    # Optional: Image analysis
    images_dir=Path("path/to/property/images"),
    
    # Optional: Property features
    feature_flags={
        "pool": True,
        "garden": True,
        "terrace": True,
        "sea_views": True,
        "parking": True,
        "air_conditioning": True,
    }
)

# Extract results
predicted_price = result['prediction']
confidence = result['confidence']
warnings = result['warnings']
image_count = result['image_count']
greenery_photos = result['greenery_photo_count']

# Display to user
print(f"Predicted Price: €{predicted_price:,.2f}")
print(f"Confidence: {confidence:.1%}")
print(f"Images analyzed: {image_count} (with {greenery_photos} showing greenery)")

if warnings:
    print("\nWarnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

## Understanding the Response

The `predict_from_manual_input()` method returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `prediction` | float | Predicted property price in euros |
| `confidence` | float | Confidence score (0.0 to 1.0) based on feature completeness |
| `warnings` | list[str] | Data quality warnings (e.g., values outside typical ranges) |
| `image_count` | int | Number of images analyzed (0 if no images_dir provided) |
| `greenery_photo_count` | int | Number of images with detected greenery |
| `input_data` | DataFrame | The prepared feature data used for prediction |

## Error Handling

```python
from pathlib import Path
from model import RandomForestPriceModel

try:
    model = RandomForestPriceModel.load(Path("models") / "random_forest.pkl")
except FileNotFoundError:
    print("Error: Model file not found. Please train the model first.")
    exit(1)

try:
    result = model.predict_from_manual_input(
        bedrooms=bedrooms_input,
        bathrooms=bathrooms_input,
        indoor_sqm=sqm_input,
        # ... other parameters
    )
except ValueError as e:
    print(f"Error: {e}")
    # Handle invalid input
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Validation Rules

Implement client-side validation:

```python
def validate_input(bedrooms, bathrooms, indoor_sqm):
    errors = []
    
    if bedrooms < 0 or bedrooms > 20:
        errors.append("Bedrooms must be between 0 and 20")
    
    if bathrooms < 0 or bathrooms > 15:
        errors.append("Bathrooms must be between 0 and 15")
    
    if indoor_sqm < 0 or indoor_sqm > 2000:
        errors.append("Indoor area must be between 0 and 2000 m²")
    
    return errors
```

## Training New Models

If you need to retrain the model with new data:

```python
from model import run_training
from pathlib import Path

# Train with default settings
run_training()

# Or customize
run_training(
    data_path=Path("data") / "properties_processed.csv",
    test_size=0.2,
    random_state=42,
    model_path=Path("models") / "random_forest.pkl"
)
```

## Feature Importance

To show users which features matter most:

```python
top_features = model.top_importances(k=10)

print("Most Important Factors:")
for i, (feature_name, importance) in enumerate(top_features, 1):
    print(f"{i}. {feature_name}: {importance:.1%}")
```

This can help users understand why certain features increase or decrease the predicted price.

## Support Files

- **model.py**: Core model implementation
- **test_prediction.py**: Example usage and test cases
- **prep.py**: Data preprocessing (reference for feature engineering)
- **requirements.txt**: Python dependencies

**Last Updated**: December 2025  
**Model Version**: RandomForest v1.0  
**Required Dependencies**: scikit-learn, pandas, numpy, joblib, Pillow
