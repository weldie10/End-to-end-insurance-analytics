"""
Claim Severity Prediction Model

This module provides the ClaimSeverityPredictor class for predicting
the financial liability (TotalClaims) for policies that have a claim.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ClaimSeverityPredictor:
    """
    A class for predicting claim severity (TotalClaims amount) for policies with claims.
    
    This model estimates the financial liability associated with a policy
    when a claim occurs. Only trained on data where TotalClaims > 0.
    
    Models supported:
    - Linear Regression
    - Random Forest
    - XGBoost
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize the ClaimSeverityPredictor.
        
        Args:
            model_type: Type of model ('linear', 'random_forest', 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names: Optional[list] = None
        self.training_results: Dict = {}
        
        logger.info(f"ClaimSeverityPredictor initialized with model_type={model_type}")
    
    def _create_model(self, random_state: int = 42):
        """Create the specified model instance."""
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == "xgboost":
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        random_state: int = 42
    ) -> Dict:
        """
        Train the claim severity prediction model.
        
        Args:
            X_train: Training features
            y_train: Training target (TotalClaims)
            X_test: Test features (optional)
            y_test: Test target (optional)
            random_state: Random seed
            
        Returns:
            Dict: Training results with metrics
        """
        logger.info(f"Training {self.model_type} model for claim severity prediction...")
        
        self.feature_names = list(X_train.columns)
        self._create_model(random_state)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        results = {
            'train_predictions': y_train_pred,
            'train_actual': y_train.values
        }
        
        if X_test is not None and y_test is not None:
            y_test_pred = self.model.predict(X_test)
            results['test_predictions'] = y_test_pred
            results['test_actual'] = y_test.values
        
        self.training_results = results
        logger.info(f"Model training completed")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict claim severity for new data.
        
        Args:
            X: Feature data
            
        Returns:
            np.ndarray: Predicted claim amounts
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"{self.model_type} model does not support feature importance.")
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        importance_df = pd.DataFrame(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n],
            columns=['Feature', 'Importance']
        )
        
        return importance_df

