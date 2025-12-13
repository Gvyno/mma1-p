from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
from PIL import Image


DEFAULT_DATA_PATH = Path("data") / "properties_processed.csv"


@dataclass
class ModelMetrics:
	mae: float
	rmse: float
	r2: float


class RandomForestPriceModel:
	def __init__(self, n_estimators: int = 300, max_depth: int | None = None, random_state: int = 42):
		self.model = RandomForestRegressor(
			n_estimators=n_estimators,
			max_depth=max_depth,
			random_state=random_state,
			n_jobs=-1,
			oob_score=False,
		)
		self.feature_names: List[str] = []

	@staticmethod
	def _load_dataframe(csv_path: Path) -> pd.DataFrame:
		if not csv_path.exists(): raise FileNotFoundError(f"Dataset not found at {csv_path}")
		df = pd.read_csv(csv_path)
		return df

	@staticmethod
	def _split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
		drop_cols = {"price", "reference", "location", "property_type"}
		feature_cols = [col for col in df.columns if col not in drop_cols]
		X = df[feature_cols].copy()

		bool_cols = X.select_dtypes(include=[bool]).columns.tolist()
		if bool_cols: X[bool_cols] = X[bool_cols].astype(int)

		y = df["price"].astype(float)
		return X, y

	def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
		self.feature_names = X_train.columns.tolist()
		self.model.fit(X_train, y_train)

	def save(self, path: Path) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"model": self.model,
			"feature_names": self.feature_names,
		}
		joblib.dump(payload, path)

	@classmethod
	def load(cls, path: Path) -> "RandomForestPriceModel":
		path = Path(path)
		payload = joblib.load(path)
		model_obj = cls()
		model_obj.model = payload["model"]
		model_obj.feature_names = payload.get("feature_names", [])
		return model_obj

	def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelMetrics:
		preds = self.model.predict(X_test)
		mae = mean_absolute_error(y_test, preds)
		rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
		r2 = r2_score(y_test, preds)
		return ModelMetrics(mae=mae, rmse=rmse, r2=r2)

	def top_importances(self, k: int = 10) -> List[Tuple[str, float]]:
		if not self.feature_names: return []
		
		importances = self.model.feature_importances_
		pairs = list(zip(self.feature_names, importances))
		pairs.sort(key=lambda p: p[1], reverse=True)
		return pairs[:k]

	@staticmethod
	def _detect_image_greenery(image_path: Path) -> bool:
		"""
		Detect if an image has greenery based on green channel dominance.
		Adapted from prep.py detect_image_content method.
		"""
		try:
			img = Image.open(image_path)
			img = img.convert('RGB')
			pixels = np.array(img)
			
			# Extract color channels
			red_channel = pixels[:, :, 0]
			green_channel = pixels[:, :, 1]
			blue_channel = pixels[:, :, 2]
			
			# Green detection for gardens/outdoor spaces
			green_dominant = (green_channel > red_channel) & (green_channel > blue_channel)
			green_ratio = green_dominant.sum() / green_channel.size
			
			return green_ratio > 0.2
		except Exception as e:
			print(f"Warning: Could not process image {image_path}: {e}")
			return False

	@staticmethod
	def _extract_image_features(images_dir: Path) -> Tuple[int, int]:
		"""
		Extract image features from a directory of images.
		
		Args: images_dir: Path to directory containing property images
		
		Returns: Tuple of (total_image_count, greenery_photo_count)
		"""
		if not isinstance(images_dir, Path):
			images_dir = Path(images_dir)
		
		if not images_dir.exists() or not images_dir.is_dir():
			return 0, 0
		
		# Find all image files
		image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
		image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix in image_extensions]
		
		image_count = len(image_files)
		greenery_photo_count = sum(RandomForestPriceModel._detect_image_greenery(image_file) for image_file in image_files)

		return image_count, greenery_photo_count

	def predict_from_manual_input(
		self,
		bedrooms: float,
		bathrooms: float,
		indoor_sqm: float,
		images_dir: Path | None = None,
		desc_length: int = 0,
		desc_has_luxury: bool = False,
		desc_has_exclusive: bool = False,
		desc_has_stunning: bool = False,
		feature_flags: Dict[str, bool] | None = None,
	) -> Dict[str, Any]:
		"""
		Predict property price from manually entered data.
		
		Args:
			bedrooms: Number of bedrooms (float)
			bathrooms: Number of bathrooms (float)
			indoor_sqm: Indoor area in square meters (float)
			images_dir: Optional path to directory containing property images for analysis
			desc_length: Length of property description (int)
			desc_has_luxury: Whether description contains "luxury" or "luxurious" (bool)
			desc_has_exclusive: Whether description contains "exclusive" (bool)
			desc_has_stunning: Whether description contains "stunning" (bool)
			feature_flags: Dict of feature flags (e.g., {"has_pool": True, "has_garden": False})
		
		Returns:
			Dict with keys:
				- 'prediction': predicted price (float)
				- 'input_data': the prepared feature DataFrame
				- 'confidence': relative confidence based on feature completeness
				- 'warnings': list of any data quality warnings
				- 'image_count': number of images found in directory (if provided)
				- 'greenery_photo_count': number of images with greenery detected (if provided)
		"""
		if not self.feature_names: raise ValueError("Model has no feature names. Ensure model is trained or loaded before prediction.")
		
		warnings = []
		image_count = 0
		greenery_photo_count = 0
		
		# Validate numeric inputs
		if bedrooms < 0 or bedrooms > 20: warnings.append(f"Bedrooms value {bedrooms} is outside typical range (0-20)")
		if bathrooms < 0 or bathrooms > 15: warnings.append(f"Bathrooms value {bathrooms} is outside typical range (0-15)")
		if indoor_sqm < 0 or indoor_sqm > 2000: warnings.append(f"Indoor sqm {indoor_sqm} is outside typical range (0-2000)")
		
		# If images directory provided, extract image features
		if images_dir is not None:
			images_dir = Path(images_dir)
			if images_dir.exists() and images_dir.is_dir():
				image_count, greenery_photo_count = self._extract_image_features(images_dir)
				print(f"Images processed: {image_count} total, {greenery_photo_count} with greenery detected")
			else: warnings.append(f"Images directory not found or is not a directory: {images_dir}")
		
		# Create input data dictionary
		input_data = {
			'bedrooms': bedrooms,
			'bathrooms': bathrooms,
			'indoor_sqm': indoor_sqm,
			'desc_length': desc_length,
			'desc_has_luxury': int(desc_has_luxury),
			'desc_has_exclusive': int(desc_has_exclusive),
			'desc_has_stunning': int(desc_has_stunning),
			'image_count': image_count,
			'greenery_photo_count': greenery_photo_count,
		}
		
		# Add feature flags if provided
		if feature_flags is not None:
			for feature_name, flag_value in feature_flags.items():
				col_name = f"has_{feature_name.replace(' ', '_').replace('/', '_').replace('-', '_')}"
				input_data[col_name] = int(flag_value)
		
		# Create DataFrame with all required features
		X_input = pd.DataFrame([input_data])
		
		# Ensure all model features are present, filling missing ones with 0
		for feature in self.feature_names:
			if feature not in X_input.columns:
				X_input[feature] = 0
		
		# Reorder columns to match training data
		X_input = X_input[self.feature_names]
		
		# Make prediction
		prediction = float(self.model.predict(X_input)[0])
		
		# Calculate confidence score based on data completeness
		total_features = len(self.feature_names)
		provided_features = sum(1 for col in self.feature_names if input_data.get(col, 0) != 0)
		confidence = provided_features / total_features if total_features > 0 else 0.5
		
		return {
			'prediction': prediction,
			'input_data': X_input,
			'confidence': confidence,
			'warnings': warnings,
			'image_count': image_count,
			'greenery_photo_count': greenery_photo_count,
		}


def run_training(
	data_path: Path = DEFAULT_DATA_PATH,
	test_size: float = 0.2,
	random_state: int = 42,
	model_path: Path | None = Path("models") / "random_forest.pkl", 
) -> None:
	df = RandomForestPriceModel._load_dataframe(data_path)
	X, y = RandomForestPriceModel._split_features_target(df)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state
	)

	model = RandomForestPriceModel(random_state=random_state)
	model.train(X_train, y_train)
	metrics = model.evaluate(X_test, y_test)

	print("RandomForestRegressor results")
	print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")
	print(f"MAE : {metrics.mae:,.2f}")
	print(f"RMSE: {metrics.rmse:,.2f}")
	print(f"R^2 : {metrics.r2:.4f}")

	top_feats = model.top_importances()
	if top_feats:
		print("Top feature importances:")
		for name, score in top_feats: print(f"  {name:30s} {score:.4f}")

	if model_path is not None:
		print(f"\nSaving model to: {model_path}")
		model.save(model_path)
