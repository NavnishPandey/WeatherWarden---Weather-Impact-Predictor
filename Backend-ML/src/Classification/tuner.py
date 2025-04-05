import optuna
from model import WeatherSeverityClassifier
from utils.classification_utils import load_data, preprocess_data
from sklearn.model_selection import cross_val_score

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
    }
    
    model = WeatherSeverityClassifier(params=params)
    scores = cross_val_score(model.model, X, y, cv=3, scoring='accuracy')
    return scores.mean()

def optimize_hyperparameters(X, y, n_trials):
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    return study.best_params