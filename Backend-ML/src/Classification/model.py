from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from typing import Dict, Optional
import numpy as np
import pandas as pd

class WeatherSeverityClassifier:
    def __init__(self, df: pd.DataFrame=None, params: Optional[Dict] = None, n_features: int = 4):
        self.df = df  # Save full DataFrame to extract column names
        self.n_features = n_features
        self.model = self._build_model(params or {})
        self.selected_features = None
        self.feature_importances_ = None

    def _build_model(self, params: Dict) -> Pipeline:
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        effective_params = {**default_params, **params}

        return Pipeline([
            ('feature_selector', SelectKBest(
                mutual_info_classif,
                k=self.n_features
            )),
            ('classifier', RandomForestClassifier(**effective_params))
        ])

    def train(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Train the model and store selected features and importances"""
        self.model.fit(X, y)

        # Get feature mask from SelectKBest
        selector = self.model.named_steps['feature_selector']
        mask = selector.get_support()

        # Use the column names from the original DataFrame (df)
        self.selected_features = X.columns[mask].tolist()

        

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Dict:
        return self.feature_importances_

    def get_selected_features(self) -> list:
        return self.selected_features
