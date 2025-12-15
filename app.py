from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import os
import shutil
import uuid
from model import RandomForestPriceModel

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model once at startup
MODEL_PATH = Path("models") / "random_forest.pkl"
try:
    model = RandomForestPriceModel.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please train the model first by running: python train_model.py")
    model = None

# Available features from the dataset
AVAILABLE_FEATURES = [
    "Private_Pool", "Private_Garden", "U_F_Heating_Climate_Control",
    "Games_Room", "Luxury", "Communal_Pool", "Alarm_System_Security",
    "Jacuzzi", "Communal_Garden", "Domotics", "Gym", "Barbeque", "Sauna",
    "New_Construction_Condition", "Guest_Apartment", "Staff_Accommodation",
    "Heated_pool", "Fireplace_Climate_Control", "Pool_Views", "More_Than_One_Parking"
]

@app.route('/')
def index():
    return render_template('index.html', features=AVAILABLE_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded. Please train the model first.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get form data
        bedrooms = float(request.form.get('bedrooms', 0))
        bathrooms = float(request.form.get('bathrooms', 0))
        indoor_sqm = float(request.form.get('indoor_sqm', 0))
        description = request.form.get('description', '')
        
        # Get selected features
        selected_features = request.form.getlist('features')
        feature_flags = {feature: True for feature in selected_features}
        
        # Handle image uploads
        images_dir = None
        if 'images' in request.files:
            files = request.files.getlist('images')
            if files and files[0].filename:
                # Create temporary directory for this prediction
                temp_id = str(uuid.uuid4())
                temp_dir = Path(app.config['UPLOAD_FOLDER']) / temp_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save uploaded images
                for file in files:
                    if file.filename:
                        filename = file.filename
                        # Only allow image files
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file.save(temp_dir / filename)
                
                images_dir = temp_dir
        
        # Extract description features
        desc_length = len(description)
        desc_lower = description.lower()
        desc_has_luxury = 'luxury' in desc_lower or 'luxurious' in desc_lower
        desc_has_exclusive = 'exclusive' in desc_lower
        desc_has_stunning = 'stunning' in desc_lower
        
        # Make prediction
        result = model.predict_from_manual_input(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            indoor_sqm=indoor_sqm,
            images_dir=images_dir,
            desc_length=desc_length,
            desc_has_luxury=desc_has_luxury,
            desc_has_exclusive=desc_has_exclusive,
            desc_has_stunning=desc_has_stunning,
            feature_flags=feature_flags if feature_flags else None
        )
        
        # Clean up temporary images directory
        if images_dir and images_dir.exists():
            shutil.rmtree(images_dir)
        
        return render_template('result.html', result=result, 
                             bedrooms=bedrooms, bathrooms=bathrooms, 
                             indoor_sqm=indoor_sqm, description=description,
                             selected_features=selected_features)
    
    except ValueError as e:
        flash(f'Invalid input: {str(e)}', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

