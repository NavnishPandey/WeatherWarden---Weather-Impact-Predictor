import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from Temperature_forecast.model import TemperaturePredictor
import numpy as np


def tune_temperature_model(X, y, n_trials: int = 50) -> dict:
    """Optimize temperature prediction hyperparameters"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0)
        }
        
        model = TemperaturePredictor(params).model
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(
            model, X, y, 
            cv=tscv, 
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=-1
        )
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params