"""
Insurance Data Preprocessing Module

This module provides the InsuranceDataPreprocessor class for preparing
insurance data for machine learning models, including feature engineering,
encoding, and missing value handling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class InsuranceDataPreprocessor:
    """
    A class for preprocessing insurance data for machine learning models.
    
    Handles:
    - Missing value imputation
    - Feature engineering
    - Categorical encoding (one-hot and label encoding)
    - Feature scaling
    - Train-test splitting
    """
    
    def __init__(self, encoding_method: str = "one_hot"):
        """
        Initialize the InsuranceDataPreprocessor.
        
        Args:
            encoding_method: Method for encoding categorical variables
                            ('one_hot' or 'label')
        """
        self.encoding_method = encoding_method
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
        logger.info(f"InsuranceDataPreprocessor initialized with encoding_method={encoding_method}")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features that might be relevant to predictions.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        df = data.copy()
        
        # Age-based features
        if "RegistrationYear" in df.columns:
            current_year = 2015  # Based on data period
            df["VehicleAge"] = current_year - df["RegistrationYear"]
            df["VehicleAge"] = df["VehicleAge"].clip(lower=0, upper=50)  # Cap at 50 years
        
        # Premium-to-value ratio
        if "TotalPremium" in df.columns and "SumInsured" in df.columns:
            df["PremiumToInsuredRatio"] = df["TotalPremium"] / (df["SumInsured"] + 1)
        
        # Claim frequency indicator (if we have historical data)
        if "TotalClaims" in df.columns:
            df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
            df["ClaimAmount"] = df["TotalClaims"].clip(lower=0)
        
        # Margin calculation
        if "TotalPremium" in df.columns and "TotalClaims" in df.columns:
            df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
            df["LossRatio"] = df["TotalClaims"] / (df["TotalPremium"] + 1)
        
        # Vehicle value categories
        if "CustomValueEstimate" in df.columns:
            df["ValueCategory"] = pd.cut(
                df["CustomValueEstimate"],
                bins=[0, 50000, 150000, 300000, float('inf')],
                labels=["Low", "Medium", "High", "VeryHigh"]
            )
        
        # Premium categories
        if "TotalPremium" in df.columns:
            df["PremiumCategory"] = pd.cut(
                df["TotalPremium"],
                bins=[-float('inf'), 0, 100, 500, 2000, float('inf')],
                labels=["Negative", "Low", "Medium", "High", "VeryHigh"]
            )
        
        logger.info(f"Engineered features. New shape: {df.shape}")
        return df
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            strategy: Strategy for imputation ('median', 'mean', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        df = data.copy()
        
        if strategy == "drop":
            df = df.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
            return df
        
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0, inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col].fillna(mode_value, inplace=True)
        
        logger.info(f"Handled missing values using {strategy} strategy")
        return df
    
    def encode_categorical_features(
        self,
        data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Encode categorical features into numeric format.
        
        Args:
            data: Input DataFrame
            categorical_columns: List of categorical columns to encode.
                               If None, auto-detect categorical columns.
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        df = data.copy()
        
        if categorical_columns is None:
            categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
        
        if self.encoding_method == "one_hot":
            # One-hot encoding
            for col in categorical_columns:
                if col in df.columns:
                    if not self.is_fitted:
                        # Fit encoder
                        self.onehot_encoders[col] = OneHotEncoder(
                            sparse_output=False,
                            handle_unknown='ignore',
                            drop='first'
                        )
                        self.onehot_encoders[col].fit(df[[col]])
                    
                    # Transform
                    encoded = self.onehot_encoders[col].transform(df[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in self.onehot_encoders[col].categories_[0][1:]],
                        index=df.index
                    )
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        
        elif self.encoding_method == "label":
            # Label encoding
            for col in categorical_columns:
                if col in df.columns:
                    if not self.is_fitted:
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(df[col].astype(str))
                    
                    # Transform
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features using {self.encoding_method} encoding")
        return df
    
    def prepare_features_for_training(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            feature_columns: Specific columns to use as features.
                           If None, use all except target and excluded columns.
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple: (X, y) features and target
        """
        df = data.copy()
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_column])
        
        # Get feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
            if exclude_columns:
                feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        # Select features
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove non-numeric columns that weren't encoded
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Fill any remaining missing values
        X = X.fillna(X.median())
        
        self.feature_names = list(X.columns)
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple: Scaled training and test features
        """
        if not self.is_fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.is_fitted = True
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names after preprocessing."""
        if self.feature_names is None:
            raise ValueError("Features not prepared yet. Call prepare_features_for_training() first.")
        return self.feature_names

