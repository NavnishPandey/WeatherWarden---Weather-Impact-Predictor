from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.utils.regression_utils import build_preprocessor
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class RainfallPredictor:
    def __init__(self, params: dict = None):
        self.numeric_features = ['humidity', 'wind_speed', 'pressure',
            'visibility', 'traffic_density',
            'month_sin', 'month_cos',
            'day_sin', 'day_cos',
            'is_weekend']
        self.categorical_features = ['weather_condition', 'road_condition']
        
        self.preprocessor = build_preprocessor(self.numeric_features, self.categorical_features)
        self.model = self._build_model(params)
    
    def _build_model(self, params):
        """Build model pipeline with default or custom parameters"""
        default_params = {
            'n_estimators': 150,
            'max_depth': 8,
            'min_samples_split': 5
        }
        effective_params = {**default_params, **(params or {})}
        
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(**effective_params))
        ])
    
    def train(self, X, y):
        """Train the rainfall prediction model"""
        self.model.fit(X, y)
        return self.model