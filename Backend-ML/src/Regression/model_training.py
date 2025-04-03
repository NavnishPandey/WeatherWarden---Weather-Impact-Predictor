import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from Rainfall_forecast.model import RainfallPredictor
from Temperature_forecast.model import TemperaturePredictor
from Rainfall_forecast.tuner import tune_rainfall_model 
from Temperature_forecast.tuner import tune_temperature_model
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
print("\nTuning temperature model...")
best_temp_params = tune_temperature_model(X_train, y_train, n_trials=10)
temp_model = TemperaturePredictor(best_temp_params).train(X_train, y_train)
temp_metrics = evaluate_model(temp_model, X_test, y_test, "Temperature")
print("Temperature Model Metrics:", temp_metrics)

# Rainfall Prediction
print("\n=== Rainfall Prediction ===")
print("\nTuning rainfall model...")
X_train_rain, X_test_rain, y_train_rain, y_test_rain = temporal_train_test_split(features, rain_target)
best_rain_params = tune_rainfall_model(X_train_rain, y_train_rain, n_trials=10)
rain_model = RainfallPredictor().train(X_train_rain, y_train_rain)
rain_metrics = evaluate_model(rain_model, X_test_rain, y_test_rain, "Rainfall")
print("Rainfall Model Metrics:", rain_metrics)