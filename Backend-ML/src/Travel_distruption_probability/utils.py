import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.utils import class_weight

class DataProcessor:
    """Handles data loading, preprocessing, feature engineering, and class imbalance"""
    
    def __init__(self, file_path: str, k_features: int = 4):
        self.file_path = file_path
        self.label_encoders = {}
        self.target_encoder = None
        self.k_features = k_features
        self.feature_selector = None
        self.selected_features = None
        self.class_weights = None
        self.class_weight_dict = None
        self.balance_method = None
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate the raw data"""
        df = pd.read_csv(self.file_path)
        
        # Check for required base columns
        required_base_columns = [
            'date', 'temperature', 'humidity', 'wind_speed', 
            'pressure', 'rainfall', 'visibility', 'weather_condition',
            'severity', 'road_condition', 'traffic_density', 'travel_disruption'
        ]
        
        missing_cols = [col for col in required_base_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")
            
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all derived features"""
        # Temporal features
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Interaction features
        df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
        df['wind_pressure_ratio'] = df['wind_speed'] / df['pressure']
        
        # Transformations
        df['rainfall'] = np.log1p(df['rainfall'])
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame, balance_method: str = 'class_weight') -> tuple:
        """
        Full preprocessing pipeline with feature engineering and class balancing
        Args:
            balance_method: One of ['class_weight', 'oversample', 'undersample', 'smote', 'smote_tomek']
        Returns:
            Tuple of (X_train_selected, X_test_selected, y_train, y_test)
        """
        # Step 1: Feature engineering
        df = self._engineer_features(df)
        
        # Step 2: Encode categorical variables
        categorical_cols = ['weather_condition', 'severity', 'road_condition']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Step 3: Encode target variable
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(df['travel_disruption'])
        
        # Step 4: Define final feature set
        features = [
            'temperature', 'temp_humidity_index', 'rainfall', 'humidity',
            'wind_speed', 'pressure', 'wind_pressure_ratio', 'visibility',
            'weather_condition', 'severity', 'road_condition', 'traffic_density',
            'day_of_year', 'day_of_week'
        ]
        
        # Verify all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features after engineering: {missing_features}")
        
        X = df[features]
        self.feature_names = features
        
        # Step 5: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Step 6: Handle class imbalance
        self.balance_method = balance_method
        if balance_method == 'class_weight':
            self.class_weights = class_weight.compute_sample_weight('balanced', y_train)
            self.class_weight_dict = dict(zip(
                np.unique(y_train),
                class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            ))
        elif balance_method == 'oversample':
            ros = RandomOverSampler(random_state=42)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        elif balance_method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        elif balance_method == 'smote':
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        elif balance_method == 'smote_tomek':
            smt = SMOTETomek(random_state=42)
            X_train, y_train = smt.fit_resample(X_train, y_train)
        else:
            raise ValueError(f"Unknown balance method: {balance_method}")
        
        # Step 7: Feature selection
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=self.k_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Store selected features
        mask = self.feature_selector.get_support()
        self.selected_features = [self.feature_names[i] for i in range(len(mask)) if mask[i]]
        
        print(f"\nPreprocessing complete. Used {balance_method} for class imbalance.")
        print(f"Selected features: {self.selected_features}")
        
        return X_train_selected, X_test_selected, y_train, y_test
    
    def get_selected_features(self):
        """Return names of selected features"""
        return self.selected_features

    def get_target_classes(self):
        """Return target class names"""
        return list(self.target_encoder.classes_)
    
    def get_class_weights(self):
        """Get computed class weights if using class_weight method"""
        return {
            'sample_weights': self.class_weights,
            'class_weight_dict': self.class_weight_dict
        } if hasattr(self, 'class_weights') else None

    def _print_class_distribution(self, y_train, y_test):
        """Helper to print class distribution information"""
        train_counts = pd.Series(y_train).value_counts()
        test_counts = pd.Series(y_test).value_counts()
        
        print("\nClass Distribution After Balancing:")
        print(f"Training set: {dict(train_counts)}")
        print(f"Test set: {dict(test_counts)}")
        
        if self.balance_method != 'class_weight':
            print("\nSample counts after balancing:")
            print(f"Training set: {len(y_train)} samples")
            print(f"Test set: {len(y_test)} samples")

    def get_class_weights(self):
        """Get computed class weights if using class_weight method"""
        if hasattr(self, 'class_weights'):
            return {
                'sample_weights': self.class_weights,
                'class_weight_dict': self.class_weight_dict
            }
        return None

    def get_selected_features(self):
        """Return names of selected features"""
        return self.selected_features

    def get_target_classes(self):
        """Return target class names"""
        return list(self.target_encoder.classes_)
    
    def get_label_encoder(self, column: str) -> LabelEncoder:
        """Get label encoder for a specific column"""
        return self.label_encoders.get(column)