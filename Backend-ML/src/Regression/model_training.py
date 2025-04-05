import sys
import os
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))

from Temperature_forecast.model import TemperaturePredictor
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
temp_model, selected_features = TemperaturePredictor(best_temp_params).train(X_train, y_train)

temp_metrics = evaluate_model(temp_model, X_test, y_test, "Temperature")
print("Temperature Model Metrics:", temp_metrics)
print("slected features:", selected_features)


