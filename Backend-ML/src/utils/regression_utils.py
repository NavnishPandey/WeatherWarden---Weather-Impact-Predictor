import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess weather data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Common feature engineering
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    temp_target = df['temperature']
    rain_target = df['rainfall']
    
    # Prepare features and targets - CORRECTED VERSION
    features = df.drop([
        'date', 
        'severity', 
        'travel_disruption',
        'temperature',  # Now explicitly dropping targets
    ], axis=1)
    
    return df, features, temp_target, rain_target



def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Build preprocessor pipeline with StandardScaler for numeric and Label Encoding (OrdinalEncoder) for categorical"""
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def temporal_train_test_split(features, target, test_size=0.2):
    """Split data temporally (no shuffling)"""
    split_idx = int(len(features) * (1 - test_size))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, target_name=""):
    """Evaluate model performance and plot results"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'R2': r2_score(y_test, predictions)
    }
    
    
    return metrics