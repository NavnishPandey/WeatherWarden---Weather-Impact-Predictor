import sys
import os
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))

from model import WeatherSeverityClassifier
from utils.classification_utils import load_and_preprocess_data, evaluate_classifier, train_val_test_split
from tuner import optimize_hyperparameters

data_path = r'C:\Users\Alka\Documents\WeatherWarden---Weather-Impact-Predictor\Backend-ML\src\Data gathering\weather_data.csv'

print("\n=== Weather condition Classification ===")
features, target = load_and_preprocess_data(data_path)
best_params = optimize_hyperparameters(features, target, 10)

    # Train final model
X_train, X_test, y_train, y_test = train_val_test_split(features, target)
model = WeatherSeverityClassifier(params=best_params)
model.train(X_train, y_train)
    
evaluate_classifier(model, X_test, y_test)
print(model.selected_features)