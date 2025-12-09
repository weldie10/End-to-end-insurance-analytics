"""
Model Trainer Module

This module provides the ModelTrainer class for training machine learning
models to predict optimal premium values.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Optional, List, Tuple
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class for training machine learning models to predict optimal premium values.
    
    Attributes:
        model: Trained model
        scaler: Feature scaler
        encoders: Dictionary of label encoders for categorical features
        feature_importance: Feature importance scores
        results: Model evaluation results
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the ModelTrainer.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.encoders: Dict = {}
        self.feature_importance: Optional[Dict] = None
        self.results: Dict = {}
        
        logger.info(f"ModelTrainer initialized with model_type={model_type}")
    
    def prepare_features(
        self,
        data: pd.DataFrame,
        target_column: str = "TotalPremium"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple: (X, y) features and target
        """
        # Car features
        car_features = [
            "VehicleType", "RegistrationYear", "make", "Model",
            "Cylinders", "cubiccapacity", "kilowatts", "bodytype",
            "NumberOfDoors", "CustomValueEstimate", "AlarmImmobiliser",
            "TrackingDevice", "NewVehicle", "WrittenOff", "Rebuilt"
        ]
        
        # Owner features
        owner_features = [
            "Gender", "MaritalStatus", "Title", "Language", "Bank",
            "AccountType", "IsVATRegistered", "Citizenship", "LegalType"
        ]
        
        # Location features
        location_features = [
            "Province", "PostalCode", "MainCrestaZone", "SubCrestaZone", "Country"
        ]
        
        # Plan features
        plan_features = [
            "SumInsured", "TermFrequency", "CalculatedPremiumPerTerm",
            "ExcessSelected", "CoverCategory", "CoverType", "CoverGroup",
            "Section", "Product", "StatutoryClass", "StatutoryRiskType"
        ]
        
        # Combine all features
        all_features = car_features + owner_features + location_features + plan_features
        
        # Filter to existing columns
        available_features = [f for f in all_features if f in data.columns]
        
        X = data[available_features].copy()
        y = data[target_column].copy()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                # Handle unseen categories
                known_classes = set(self.encoders[col].classes_)
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in known_classes else 'Unknown'
                )
                # Add 'Unknown' if not in classes
                if 'Unknown' not in self.encoders[col].classes_:
                    self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'Unknown')
                X[col] = self.encoders[col].transform(X[col])
        
        # Fill missing numeric values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        return X, y
    
    def train(
        self,
        data: pd.DataFrame,
        target_column: str = "TotalPremium",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train the model.
        
        Args:
            data: Training data
            target_column: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed
            
        Returns:
            Dict: Training results
        """
        logger.info("Starting model training...")
        
        # Prepare features
        X, y = self.prepare_features(data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        elif self.model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        self.results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': self.feature_importance
        }
        
        logger.info(f"Training completed. Test R² = {test_r2:.4f}, Test RMSE = {test_rmse:.4f}")
        return self.results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance_df = pd.DataFrame(
            list(self.feature_importance.items())[:top_n],
            columns=['Feature', 'Importance']
        )
        return importance_df
    
    def save_model(self, model_path: Path):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'model_type': self.model_type,
            'results': self.results
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.encoders = model_data['encoders']
        self.model_type = model_data['model_type']
        self.results = model_data.get('results', {})
        
        logger.info(f"Model loaded from {model_path}")

