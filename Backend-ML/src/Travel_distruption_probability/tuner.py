from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform
import numpy as np

class ModelTuner:
    """Handles hyperparameter tuning for weather impact models"""
    
    def __init__(self):
        self.best_params = None
        self.best_score = None
    
    def _get_param_grids(self):
        """Define hyperparameter search spaces for different models"""
        return {
            'XGBoost': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(2, 8),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5),
                'reg_alpha': uniform(0, 2),
                'reg_lambda': uniform(0, 2),
                'gamma': uniform(0, 0.5),
                'min_child_weight': randint(1, 10)
            },
            'LightGBM': {
                'n_estimators': randint(50, 300),
                'num_leaves': randint(10, 50),
                'max_depth': randint(2, 8),
                'learning_rate': uniform(0.01, 0.3),
                'feature_fraction': uniform(0.5, 0.5),
                'bagging_fraction': uniform(0.5, 0.5),
                'reg_alpha': uniform(0, 2),
                'reg_lambda': uniform(0, 2),
                'min_data_in_leaf': randint(5, 30)
            },
            'RandomForest': {
                'n_estimators': randint(50, 300),
                'max_depth': [None] + list(range(2, 10)),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'ccp_alpha': uniform(0, 0.02),
                'max_samples': uniform(0.5, 0.5)
            },
            'GradientBoosting': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(2, 8),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.5, 0.5),
                'max_features': ['sqrt', 'log2', None]
            }
        }
    
    def tune_model(self, estimator, X, y, model_type, n_iter=30, cv=5):
        """
        Perform randomized search for hyperparameter tuning
        
        Args:
            estimator: The model to tune
            X: Training features
            y: Training labels
            model_type: Type of model ('XGBoost', 'LightGBM', etc.)
            n_iter: Number of parameter settings to sample
            cv: Number of cross-validation folds
            
        Returns:
            best_estimator: Model with optimized parameters
            best_params: Dictionary of best parameters found
        """
        param_grid = self._get_param_grids().get(model_type, {})
        
        if not param_grid:
            raise ValueError(f"No parameter grid defined for model type: {model_type}")
        
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_log_loss',
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y)
        self.best_params = search.best_params_
        self.best_score = -search.best_score_
        
        print(f"\nBest parameters for {model_type}:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        print(f"Best log loss: {self.best_score:.4f}")
        
        return search.best_estimator_, search.best_params_