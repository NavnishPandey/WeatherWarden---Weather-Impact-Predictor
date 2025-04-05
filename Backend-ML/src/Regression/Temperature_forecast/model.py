from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from utils.regression_utils import build_preprocessor

class TemperaturePredictor:
    def __init__(self, params: dict = None, n_features: int = 4):
        # All possible features we might consider
        self.all_numeric_features = [
            'rainfall', 'humidity', 'wind_speed', 'pressure',
            'visibility', 'traffic_density',
            'month_sin', 'month_cos',
            'day_sin', 'day_cos',
            'is_weekend'
        ]
        self.categorical_features = ['weather_condition', 'road_condition']
        self.n_features = n_features  # Number of top features to select
        
        # Feature selector and model pipeline
        self.model = self._build_model(params)
        self.selected_features = None  # Will be set during training
    
    def _build_model(self, params):
        """Build the complete pipeline with feature selection"""
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        effective_params = {**default_params, **(params or {})}
        
        return Pipeline([
            ('preprocessor', build_preprocessor(
                self.all_numeric_features,
                self.categorical_features
            )),
            ('feature_selector', SelectKBest(
                mutual_info_regression,
                k=self.n_features
            )),
            ('regressor', GradientBoostingRegressor(**effective_params))
        ])
    
    def train(self, X, y):
        """Train the model and store selected features"""
        self.model.fit(X, y)
        self.selected_features = self._get_selected_features(X)
        return self.model , self.selected_features
    
    
    def _get_selected_features(self, X):
        # Get the selection mask
        mask = self.model.named_steps['feature_selector'].get_support()
        
        # Categorical columns are now single columns due to OrdinalEncoder
        all_features = self.all_numeric_features + self.categorical_features
        
        # Return only selected features
        return [f for f, m in zip(all_features, mask) if m]

    
    def get_feature_importance(self):
        """Get importance scores for selected features"""
        importances = self.model.named_steps['regressor'].feature_importances_
        return dict(zip(self.selected_features, importances))