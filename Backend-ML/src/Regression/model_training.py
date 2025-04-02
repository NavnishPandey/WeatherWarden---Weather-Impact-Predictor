import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from Rainfall_forecast.model import RainfallPredictor
from Temperature_forecast.model import TemperaturePredictor
from src.utils.regression_utils import (
    load_and_preprocess,
    temporal_train_test_split,
    evaluate_model
)


# Load data
data_path = r'C:\Users\Alka\Documents\WeatherWarden---Weather-Impact-Predictor\Backend-ML\src\Data gathering\weather_data.csv'
df, features, temp_target, rain_target = load_and_preprocess(data_path)

# Temperature Prediction
print("\n=== Temperature Prediction ===")
X_train, X_test, y_train, y_test = temporal_train_test_split(features, temp_target)
temp_model = TemperaturePredictor().train(X_train, y_train)
temp_metrics = evaluate_model(temp_model, X_test, y_test, "Temperature")
print("Temperature Model Metrics:", temp_metrics)

# Rainfall Prediction
print("\n=== Rainfall Prediction ===")
X_train, X_test, y_train, y_test = temporal_train_test_split(features, rain_target)
rain_model = RainfallPredictor().train(X_train, y_train)
rain_metrics = evaluate_model(rain_model, X_test, y_test, "Rainfall")
print("Rainfall Model Metrics:", rain_metrics)