"""
Data Loader Module

This module provides the DataLoader class for loading and preprocessing
insurance claim data from various sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to load and preprocess insurance claim data.
    
    This class handles data loading from various formats, data cleaning,
    type conversion, and basic preprocessing operations.
    
    Attributes:
        data_path (Path): Path to the data directory
        raw_data (pd.DataFrame): Raw loaded data
        processed_data (pd.DataFrame): Processed/cleaned data
        data_info (Dict): Dictionary containing data metadata
    """
    
    def __init__(self, data_path: Union[str, Path] = "Data"):
        """
        Initialize the DataLoader.
        
        Args:
            data_path: Path to the data directory. Defaults to "Data"
        """
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.data_info: Dict = {}
        
        logger.info(f"DataLoader initialized with data path: {self.data_path}")
    
    def load_data(
        self, 
        filename: str = "MachineLearningRating_v3.txt",
        delimiter: str = "|",
        encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            filename: Name of the file to load
            delimiter: Delimiter used in the file (default: "|")
            encoding: File encoding (default: "utf-8")
            
        Returns:
            pd.DataFrame: Loaded data
        """
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            self.raw_data = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                low_memory=False
            )
            logger.info(f"Successfully loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self) -> Dict:
        """
        Get information about the loaded data.
        
        Returns:
            Dict: Dictionary containing data information
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.data_info = {
            "shape": self.raw_data.shape,
            "columns": list(self.raw_data.columns),
            "dtypes": self.raw_data.dtypes.to_dict(),
            "missing_values": self.raw_data.isnull().sum().to_dict(),
            "missing_percentage": (self.raw_data.isnull().sum() / len(self.raw_data) * 100).to_dict(),
            "memory_usage": self.raw_data.memory_usage(deep=True).sum() / 1024**2,  # MB
            "numeric_columns": list(self.raw_data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.raw_data.select_dtypes(include=['object']).columns),
            "date_columns": []
        }
        
        return self.data_info
    
    def preprocess_data(
        self,
        convert_dates: bool = True,
        handle_missing: bool = True,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            convert_dates: Whether to convert date columns
            handle_missing: Whether to handle missing values
            numeric_columns: List of columns to convert to numeric
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting data preprocessing...")
        self.processed_data = self.raw_data.copy()
        
        # Convert TransactionMonth to datetime
        if convert_dates and "TransactionMonth" in self.processed_data.columns:
            self.processed_data["TransactionMonth"] = pd.to_datetime(
                self.processed_data["TransactionMonth"],
                errors='coerce'
            )
            logger.info("Converted TransactionMonth to datetime")
        
        # Convert numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in self.processed_data.columns:
                    self.processed_data[col] = pd.to_numeric(
                        self.processed_data[col],
                        errors='coerce'
                    )
        
        # Auto-detect and convert numeric columns
        numeric_cols = [
            "TotalPremium", "TotalClaims", "SumInsured", 
            "CalculatedPremiumPerTerm", "CustomValueEstimate",
            "RegistrationYear", "Cylinders", "cubiccapacity", 
            "kilowatts", "NumberOfDoors", "CapitalOutstanding",
            "NumberOfVehiclesInFleet", "PostalCode"
        ]
        
        for col in numeric_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(
                    self.processed_data[col],
                    errors='coerce'
                )
        
        # Handle missing values
        if handle_missing:
            # Fill numeric columns with 0 or median
            for col in self.processed_data.select_dtypes(include=[np.number]).columns:
                if self.processed_data[col].isnull().sum() > 0:
                    if col in ["TotalClaims"]:
                        self.processed_data[col].fillna(0, inplace=True)
                    else:
                        self.processed_data[col].fillna(
                            self.processed_data[col].median(), 
                            inplace=True
                        )
            
            # Fill categorical columns with "Unknown"
            for col in self.processed_data.select_dtypes(include=['object']).columns:
                self.processed_data[col].fillna("Unknown", inplace=True)
        
        logger.info("Data preprocessing completed")
        return self.processed_data
    
    def get_data(self, processed: bool = True) -> pd.DataFrame:
        """
        Get the loaded data.
        
        Args:
            processed: If True, return processed data; otherwise return raw data
            
        Returns:
            pd.DataFrame: Requested data
        """
        if processed:
            if self.processed_data is None:
                logger.warning("Processed data not available. Returning raw data.")
                return self.raw_data
            return self.processed_data
        else:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            return self.raw_data
    
    def save_processed_data(self, output_path: Union[str, Path], format: str = "csv"):
        """
        Save processed data to a file.
        
        Args:
            output_path: Path where to save the data
            format: File format ('csv', 'parquet', 'pkl')
        """
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            self.processed_data.to_csv(output_path, index=False)
        elif format == "parquet":
            self.processed_data.to_parquet(output_path, index=False)
        elif format == "pkl":
            self.processed_data.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Processed data saved to {output_path}")

