# Flask Web Interface - Property Price Predictor

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already done)

```bash
python train_model.py
```

This will create `models/random_forest.pkl` which is required for predictions.

### 3. Run the Flask App

```bash
python app.py
```

### 4. Open in Browser

Navigate to: **http://localhost:5000**

## Notes

- The app automatically creates the `uploads/` directory on startup
- Temporary image files are automatically deleted after prediction
- Maximum file upload size: 16MB per file
- Supported image formats: JPG, JPEG, PNG

## Troubleshooting

**Error: "Model not loaded"**
- Make sure you've run `python train_model.py` first
- Check that `models/random_forest.pkl` exists

**Port already in use**
- Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

