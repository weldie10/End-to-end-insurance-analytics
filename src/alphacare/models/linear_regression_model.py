"""
Linear Regression Model Module

This module provides the LinearRegressionModel class for fitting
linear regression models by zipcode.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class LinearRegressionModel:
    """
    A class for fitting linear regression models to predict total claims by zipcode.
    
    Attributes:
        models (Dict): Dictionary storing models for each zipcode
        results (Dict): Dictionary storing model results and metrics
    """
    
    def __init__(self):
        """Initialize the LinearRegressionModel."""
        self.models: Dict = {}
        self.results: Dict = {}
        logger.info("LinearRegressionModel initialized")
    
    def fit_by_zipcode(
        self,
        data: pd.DataFrame,
        target_column: str = "TotalClaims",
        feature_columns: Optional[List[str]] = None,
        min_samples: int = 10
    ) -> Dict:
        """
        Fit linear regression models for each zipcode.
        
        Args:
            data: DataFrame containing the data
            target_column: Name of the target column
            feature_columns: List of feature columns. If None, uses numeric columns.
            min_samples: Minimum number of samples required per zipcode
            
        Returns:
            Dict: Dictionary containing models and results for each zipcode
        """
        if "PostalCode" not in data.columns:
            raise ValueError("PostalCode column is required")
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        if feature_columns is None:
            # Default feature columns
            feature_columns = [
                "TotalPremium", "SumInsured", "CalculatedPremiumPerTerm",
                "CustomValueEstimate", "RegistrationYear", "Cylinders",
                "cubiccapacity", "kilowatts", "NumberOfDoors"
            ]
            # Filter to only existing columns
            feature_columns = [col for col in feature_columns if col in data.columns]
        
        zipcodes = data["PostalCode"].value_counts()
        valid_zipcodes = zipcodes[zipcodes >= min_samples].index
        
        logger.info(f"Fitting models for {len(valid_zipcodes)} zipcodes")
        
        for zipcode in valid_zipcodes:
            zipcode_data = data[data["PostalCode"] == zipcode].copy()
            
            # Prepare features and target
            X = zipcode_data[feature_columns].select_dtypes(include=[np.number])
            y = zipcode_data[target_column]
            
            # Remove rows with missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) < min_samples:
                continue
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            y_pred = model.predict(X)
            
            # Metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            self.models[zipcode] = {
                'model': model,
                'features': list(X.columns),
                'n_samples': len(X),
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'coefficients': dict(zip(X.columns, model.coef_)),
                'intercept': model.intercept_
            }
            
            logger.debug(f"Zipcode {zipcode}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        logger.info(f"Successfully fitted {len(self.models)} models")
        self.results['models_by_zipcode'] = self.models
        return self.models
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all models.
        
        Returns:
            pd.DataFrame: Summary of models
        """
        summary_data = []
        
        for zipcode, model_info in self.models.items():
            summary_data.append({
                'Zipcode': zipcode,
                'R² Score': model_info['r2_score'],
                'RMSE': model_info['rmse'],
                'MAE': model_info['mae'],
                'N Samples': model_info['n_samples'],
                'N Features': len(model_info['features'])
            })
        
        return pd.DataFrame(summary_data).sort_values('R² Score', ascending=False)
    
    def predict(self, zipcode: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for a given zipcode.
        
        Args:
            zipcode: Zipcode to use for prediction
            X: Feature data
            
        Returns:
            np.ndarray: Predictions
        """
        if zipcode not in self.models:
            raise ValueError(f"No model found for zipcode {zipcode}")
        
        model = self.models[zipcode]['model']
        return model.predict(X)

