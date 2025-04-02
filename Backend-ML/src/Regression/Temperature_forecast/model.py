from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.utils.regression_utils import build_preprocessor

class TemperaturePredictor:
    def __init__(self, params: dict = None):
        self.numeric_features = ['humidity', 'wind_speed', 'pressure', 'visibility', 
                               'traffic_density', 'day_of_year', 'month']
        self.categorical_features = ['weather_condition', 'road_condition']
        
        self.preprocessor = build_preprocessor(self.numeric_features, self.categorical_features)
        self.model = self._build_model(params)
    
    def _build_model(self, params):
        """Build model pipeline with default or custom parameters"""
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5
        }
        effective_params = {**default_params, **(params or {})}
        
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', GradientBoostingRegressor(**effective_params))
        ])
    
    def train(self, X, y):
        """Train the temperature prediction model"""
        self.model.fit(X, y)
        return self.model