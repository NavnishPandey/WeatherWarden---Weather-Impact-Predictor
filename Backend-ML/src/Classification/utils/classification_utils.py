from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np


def evaluate_classifier(model, X_test, y_test):
    """Generate comprehensive evaluation metrics"""
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw weather data"""
    df = pd.read_csv(filepath)
    return df


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple

def preprocess_data(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:

    # --- 2. Create a copy to avoid modifying the original DataFrame ---
    df = df.copy()
        
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # --- 4. Identify numeric and categorical columns ---
    numeric_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # --- 5. Label encode categorical features ---
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # --- 6. Standard scale numeric features ---
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # --- 7. Label encode target if categorical ---
    if y.dtype == 'object' or y.dtype.name == 'category':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
    else:
        y = y.values  # If already numeric, just extract values

    X_processed = X
    y_processed = y

    return X_processed, y_processed



def load_and_preprocess_data(filepath: str) -> Tuple:
    """
    Combined load and preprocess operation
    For training: pass target_col
    For inference: omit target_col
    """
    df = load_data(filepath)
    target = 'weather_condition'
    return preprocess_data(df, target)


def train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
   
    return X_train, X_test, y_train, y_test