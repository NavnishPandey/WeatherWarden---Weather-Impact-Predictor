from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.utils.regression_utils import build_preprocessor
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline

class RainfallPredictor:
    def __init__(self, params: dict = None, n_features: int = 4):
        self.all_numeric_features = [
            'humidity', 'wind_speed', 'pressure',
            'visibility', 'traffic_density',
            'month_sin', 'month_cos',
            'day_sin', 'day_cos',
            'is_weekend'
        ]
        self.categorical_features = ['weather_condition', 'road_condition']
        self.n_features = n_features
        self.feature_selector = SelectKBest(mutual_info_regression, k=n_features)
        self.model = self._build_model(params)
    
    def _build_model(self, params):
        # Preprocessor for all possible features
        full_preprocessor = build_preprocessor(
            self.all_numeric_features,
            self.categorical_features
        )
        
        # Final model pipeline
        return Pipeline([
            ('preprocessor', full_preprocessor),
            ('feature_selector', self.feature_selector),
            ('regressor', RandomForestRegressor(
                **(params or {}),
                random_state=42
            ))
        ])
    
    def train(self, X, y):
        # Auto-select best features during training
        self.model.fit(X, y)
        self.selected_features = self._get_selected_features(X)
        return self.model
    
    def _get_selected_features(self, X):
        # Get mask of selected features
        mask = self.model.named_steps['feature_selector'].get_support()
        
        # Combine numeric and categorical features
        all_features = (
            self.all_numeric_features + 
            list(self.model.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names_out(self.categorical_features))
        )
        
        return [f for f, m in zip(all_features, mask) if m]

    def get_feature_importance(self):
        """Get importance scores for selected features"""
        importances = self.model.named_steps['regressor'].feature_importances_
        return dict(zip(self.selected_features, importances))