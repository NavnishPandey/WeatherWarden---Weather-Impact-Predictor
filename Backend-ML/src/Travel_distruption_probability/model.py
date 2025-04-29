# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.metrics import (log_loss, accuracy_score, 
#                            precision_score, recall_score, 
#                            f1_score, confusion_matrix,
#                            classification_report)
# from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV, SequentialFeatureSelector
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from scikeras.wrappers import KerasClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l1_l2
# from tensorflow.keras.callbacks import EarlyStopping
# from imblearn.over_sampling import ADASYN
# from imblearn.pipeline import make_pipeline
# import numpy as np
# import pickle

# class WeatherImpactModel:
#     """Enhanced model with automatic algorithm selection and robust regularization"""
    
#     def __init__(self):
#         self.best_model = None
#         self.trained = False
#         self.classes_ = None
#         self.selected_features = None
#         self.model_type = None
#         self.model_scores = {}
#         self.input_shape = None
#         self.feature_names = None
#         self.class_weights = None
        
#     def _create_nn(self):
#         """Create neural network with enhanced regularization"""
#         model = Sequential([
#             Dense(128, activation='relu', 
#                   input_shape=self.input_shape,
#                   kernel_regularizer=l1_l2(l1=0.02, l2=0.02)),  # Stronger L1/L2
#             Dropout(0.6),  # Increased dropout
#             BatchNormalization(),
#             Dense(64, activation='relu',
#                   kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
#             Dropout(0.5),  # Increased dropout
#             Dense(32, activation='relu'),
#             Dense(len(self.classes_) if self.classes_ else 1, 
#                   activation='softmax')
#         ])
#         model.compile(
#             optimizer=Adam(learning_rate=0.0005),  # Smaller learning rate
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )
#         return model
    
#     def _initialize_models(self):
#         """Initialize models with strict regularization"""
#         return {
#             'XGBoost_Strict': Pipeline([
#                 ('feature_selector', SequentialFeatureSelector(
#                     XGBClassifier(max_depth=3),
#                     n_features_to_select='auto',
#                     direction='backward',
#                     cv=5
#                 )),
#                 ('classifier', XGBClassifier(
#                     objective='multi:softprob',
#                     eval_metric='mlogloss',
#                     n_estimators=200,
#                     max_depth=3,          # Reduced further
#                     learning_rate=0.01,   # Smaller
#                     subsample=0.6,        # More aggressive
#                     colsample_bytree=0.5,
#                     reg_alpha=1.0,        # Stronger L1
#                     reg_lambda=2.0,       # Stronger L2
#                     min_child_weight=3,
#                     gamma=0.2,
#                     early_stopping_rounds=25,
#                     random_state=42
#                 ))
#             ]),
#             'LightGBM_Strict': Pipeline([
#                 ('feature_selector', SelectKBest(mutual_info_classif)),
#                 ('classifier', LGBMClassifier(
#                     objective='multiclass',
#                     n_estimators=150,
#                     max_depth=3,
#                     learning_rate=0.03,
#                     num_leaves=15,        # Reduced
#                     min_data_in_leaf=20,  # Increased
#                     reg_alpha=1.0,
#                     reg_lambda=2.0,
#                     feature_fraction=0.5,
#                     bagging_fraction=0.6,
#                     random_state=42
#                 ))
#             ]),
#             'RandomForest_Strict': Pipeline([
#                 ('feature_selector', RFECV(
#                     estimator=RandomForestClassifier(max_depth=4),
#                     step=1,
#                     cv=5,
#                     scoring='accuracy'
#                 )),
#                 ('classifier', RandomForestClassifier(
#                     n_estimators=150,
#                     max_depth=5,        # Reduced
#                     min_samples_split=12, # Increased
#                     min_samples_leaf=6,  # Increased
#                     max_features=0.5,    # More feature subsampling
#                     ccp_alpha=0.02,      # Increased pruning
#                     max_samples=0.7,     # Instance subsampling
#                     random_state=42,
#                     n_jobs=-1
#                 ))
#             ]),
#             'Stochastic_GBM': Pipeline([
#                 ('feature_selector', SelectKBest(mutual_info_classif)),
#                 ('classifier', GradientBoostingClassifier(
#                     n_estimators=150,
#                     learning_rate=0.05,
#                     max_depth=3,
#                     subsample=0.5,       # Stochastic sampling
#                     max_features=0.7,
#                     validation_fraction=0.25,
#                     n_iter_no_change=10, # Early stopping
#                     random_state=42
#                 ))
#             ]),
#             'ANN_Strict': Pipeline([
#                 ('feature_selector', SelectKBest(mutual_info_classif)),
#                 ('scaler', StandardScaler()),
#                 ('nn', KerasClassifier(
#                     model=self._create_nn,
#                     epochs=150,
#                     batch_size=32,       # Smaller batches
#                     verbose=0,
#                     validation_split=0.25, # Larger validation
#                     callbacks=[EarlyStopping(patience=15)]
#                 ))
#             ])
#         }
    
#     def train(self, X_train, y_train, k_features=4):
#         """Training with strict regularization and enhanced validation"""
#         # Set input shape for NN
#         self.input_shape = (X_train.shape[1],)
        
#         # Store feature names
#         if hasattr(X_train, 'columns'):
#             self.feature_names = X_train.columns.tolist()
#         else:
#             self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
#         # Compute class weights
#         self.classes_ = np.unique(y_train)
#         self.class_weights = compute_class_weight(
#             'balanced', 
#             classes=self.classes_, 
#             y=y_train
#         )
        
#         # Initialize models with strict regularization
#         models = self._initialize_models()
        
#         # Create larger validation set (30%)
#         X_train_fit, X_val, y_train_fit, y_val = train_test_split(
#             X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
#         )
        
#         # Model evaluation with strict early stopping
#         best_score = -np.inf
#         for name, model in models.items():
#             try:
#                 # Configure early stopping
#                 if 'XGB' in name or 'GBM' in name:
#                     model.set_params(
#                         classifier__early_stopping_rounds=15,
#                         classifier__eval_set=[(X_val, y_val)],
#                         classifier__verbose=1
#                     )
#                 elif 'LGBM' in name:
#                     model.set_params(
#                         classifier__early_stopping_rounds=15,
#                         classifier__eval_set=[(X_val, y_val)],
#                         classifier__eval_metric='multi_logloss',
#                         classifier__verbose=-1
#                     )
                
#                 model.fit(X_train_fit, y_train_fit)
#                 score = model.score(X_val, y_val)
                
#                 if score > best_score:
#                     best_score = score
#                     self.best_model = model
#                     self.model_type = name
#             except Exception as e:
#                 print(f"Error training {name}: {str(e)}")
#                 continue
        
#         # Store selected features
#         if hasattr(self.best_model, 'named_steps'):
#             if 'feature_selector' in self.best_model.named_steps:
#                 selector = self.best_model.named_steps['feature_selector']
#                 if hasattr(selector, 'get_support'):
#                     mask = selector.get_support()
#                     self.selected_features = [self.feature_names[i] for i in range(len(mask)) if mask[i]]
#                 elif hasattr(selector, 'support_'):
#                     self.selected_features = [f for f, s in zip(self.feature_names, selector.support_) if s]
        
#         self.trained = True
        
#         # Print strict regularization summary
#         print("\nStrict Regularization Summary:")
#         print(f"Best Model: {self.model_type}")
        
#         return self


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle
from tuner import ModelTuner 
from sklearn.metrics import (log_loss, accuracy_score, 
                           precision_score, recall_score, 
                           f1_score, confusion_matrix,
                           classification_report)

class WeatherImpactModel:
    """Main model class with integrated hyperparameter tuning"""
    
    def __init__(self):
        self.best_model = None
        self.trained = False
        self.classes_ = None
        self.selected_features = None
        self.model_type = None
        self.model_scores = {}
        self.input_shape = None
        self.feature_names = None
        self.class_weights = None
        self.tuner = ModelTuner()  # Initialize tuner
    
    def _create_nn(self):
        """Create neural network architecture"""
        model = Sequential([
            Dense(128, activation='relu', 
                  input_shape=self.input_shape,
                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.5),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dense(len(self.classes_) if self.classes_ else 1, 
                  activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _initialize_models(self):
        """Initialize model pipelines"""
        return {
            'XGBoost': Pipeline([
                ('feature_selector', SelectKBest(mutual_info_classif)),
                ('classifier', XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    random_state=42,
                    use_label_encoder=False
                ))
            ]),
            'LightGBM': Pipeline([
                ('feature_selector', SelectKBest(mutual_info_classif)),
                ('classifier', LGBMClassifier(
                    objective='multiclass',
                    random_state=42
                ))
            ]),
            'RandomForest': Pipeline([
                ('feature_selector', SelectKBest(mutual_info_classif)),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            'GradientBoosting': Pipeline([
                ('feature_selector', SelectKBest(mutual_info_classif)),
                ('classifier', GradientBoostingClassifier(
                    random_state=42
                ))
            ]),
            'ANN': Pipeline([
                ('feature_selector', SelectKBest(mutual_info_classif)),
                ('scaler', StandardScaler()),
                ('nn', KerasClassifier(
                    model=self._create_nn,
                    epochs=100,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(patience=10)]
                ))
            ])
        }
    
    def train(self, X_train, y_train, k_features=4, tune_hyperparams=True):
        """Train model with optional hyperparameter tuning"""
        self.input_shape = (X_train.shape[1],)
        
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        self.classes_ = np.unique(y_train)
        self.class_weights = compute_class_weight(
            'balanced', classes=self.classes_, y=y_train
        )
        
        models = self._initialize_models()
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        best_score = -np.inf
        for name, model in models.items():
            try:
                model.set_params(feature_selector__k=k_features)
                
                # Skip tuning for ANN and if tuning is disabled
                if tune_hyperparams and name != 'ANN':
                    print(f"\nTuning {name} hyperparameters...")
                    model, _ = self.tuner.tune_model(
                        model.named_steps['classifier'],
                        X_train_fit,
                        y_train_fit,
                        name
                    )
                    models[name].set_params(classifier=model)
                
                # Fit the model
                models[name].fit(X_train_fit, y_train_fit)
                score = models[name].score(X_val, y_val)
                
                if score > best_score:
                    best_score = score
                    self.best_model = models[name]
                    self.model_type = name
                    
            except Exception as e:
                print(f"Error with {name}: {str(e)}")
                continue
        
        # Store selected features
        if hasattr(self.best_model, 'named_steps'):
            selector = self.best_model.named_steps['feature_selector']
            mask = selector.get_support()
            self.selected_features = [self.feature_names[i] for i in range(len(mask)) if mask[i]]
        
        self.trained = True
        return self

    # [Keep all existing predict, evaluate, save/load methods]    
        
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities"""
        if not self.trained or self.best_model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.best_model.predict_proba(X)
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels"""
        if not self.trained or self.best_model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self.best_model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        if not self.trained or self.best_model is None:
            raise RuntimeError("Model must be trained before evaluation")
            
        # Probability predictions
        probas = self.predict_proba(X_test)
        
        # Class predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_type': self.model_type,
            'log_loss': log_loss(y_test, probas),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'selected_features': list(self.selected_features),
            'model_scores': self.model_scores
        }
        
        # Print results
        print("\nModel Evaluation Metrics:")
        print(f"Model Type: {metrics['model_type']}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nSelected Features:")
        print(metrics['selected_features'])
        
        return metrics
    
    def save_model(self, file_path: str):
        """Save trained model to file"""
        if not self.trained:
            raise RuntimeError("Model must be trained before saving")
            
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'classes': self.classes_,
                'model_type': self.model_type,
                'selected_features': self.selected_features
            }, f)
    
    def load_model(self, file_path: str):
        """Load trained model from file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.best_model = data['model']
        self.classes_ = data['classes']
        self.model_type = data.get('model_type', 'Unknown')
        self.selected_features = data.get('selected_features', None)
        self.trained = True   
  
        