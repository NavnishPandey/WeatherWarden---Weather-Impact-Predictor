# model_training.py
from utils import DataProcessor
from model import WeatherImpactModel
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class ModelTrainer:
    """Enhanced training pipeline with data validation, feature engineering, and model saving"""
    
    def __init__(self, data_path: str, k_features: int = 4):
        self.data_path = data_path
        self.k_features = k_features
        self.data_processor = DataProcessor(data_path, k_features)
        self.model = WeatherImpactModel()
    
    def _validate_data(self, df):
        """Perform data validation checks"""
        required_columns = [
            'temperature', 'humidity', 'rainfall', 'wind_speed', 
            'pressure', 'visibility', 'weather_condition', 
            'severity', 'road_condition', 'traffic_density',
            'travel_disruption', 'date'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.isnull().sum().sum() > 0:
            raise ValueError("Data contains missing values")
            
        if len(df['travel_disruption'].unique()) < 2:
            raise ValueError("Target variable needs at least 2 classes")
    
    def run_pipeline(self):
        """Enhanced training pipeline with data validation, model training, evaluation and saving"""
        try:
            # Load and validate data
            df = self.data_processor.load_data()
            self._validate_data(df)
            
            # Preprocess data
            print("\nPreprocessing data and engineering features...")
            X_train, X_test, y_train, y_test = self.data_processor.preprocess_data(df, balance_method='smote_tomek')
            
            # Train model with feature selection
            print("\nTraining model with feature selection...")
            self.model.train(X_train, y_train, k_features=self.k_features)
            y_train_pred = self.model.predict(X_train)
            
            print("\nModel performance on training data:")
            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"Training Accuracy: {train_accuracy:.4f}")
            
            if self.model.tuner.best_params:
                print("Best parameters:", self.model.tuner.best_params)
            
            # Evaluate model
            print("\nEvaluating model performance on test data...")
            metrics = self.model.evaluate(X_test, y_test)
            
            # Feature importance
            self._analyze_feature_importance(X_train, y_train)
            
            # Show prediction
            sample = X_test[:1]
            self._show_prediction_example(sample)
            
            # Save trained model
            
            model_path = os.path.join("weather_disruption_model.pkl")
            joblib.dump(self.model.best_model, model_path)
            print(f"\nTrained model saved")
            
            return self.model, metrics
            
        except Exception as e:
            print(f"\nError in training pipeline: {str(e)}")
            raise
    
    def _analyze_feature_importance(self, X_train, y_train):
        """Analyze and display feature importance"""
        if hasattr(self.model.best_model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.best_model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nFeature Importance Rankings:")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {self.model.selected_features[idx]}: {importances[idx]:.4f}")
    
    def _show_prediction_example(self, sample):
        """Display example prediction with probabilities"""
        probas = self.model.predict_proba(sample)
        class_names = self.model.classes_
        
        print("\nExample Probability Prediction:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {probas[0][i]:.4f}")
        
        predicted_class = class_names[np.argmax(probas)]
        print(f"\nPredicted Class: {predicted_class}")


if __name__ == "__main__":
    data_path = r'C:\Users\Alka\Documents\WeatherWarden---Weather-Impact-Predictor\Backend-ML\src\Data_gathering\weather_data1.csv'
    
    try:
        print("Starting enhanced training pipeline...")
        trainer = ModelTrainer(data_path, k_features=5)
        trained_model, metrics = trainer.run_pipeline()
        
        print("\nTraining completed successfully!")
        print(f"Best model: {trained_model.model_type}")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
