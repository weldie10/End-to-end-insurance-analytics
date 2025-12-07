"""
Exploratory Data Analysis Module

This module provides the EDAAnalyzer class for performing comprehensive
exploratory data analysis on insurance data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDAAnalyzer:
    """
    A class for performing Exploratory Data Analysis on insurance data.
    
    This class provides methods for data summarization, visualization,
    outlier detection, and trend analysis.
    
    Attributes:
        data (pd.DataFrame): The dataset to analyze
        results (Dict): Dictionary to store analysis results
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDAAnalyzer.
        
        Args:
            data: DataFrame containing the insurance data
        """
        self.data = data.copy()
        self.results: Dict = {}
        logger.info(f"EDAAnalyzer initialized with {len(self.data)} rows")
    
    def calculate_descriptive_stats(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate descriptive statistics for numerical columns.
        
        Args:
            columns: List of columns to analyze. If None, analyzes all numeric columns.
            
        Returns:
            pd.DataFrame: Descriptive statistics
        """
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.data.columns]
        
        stats = self.data[numeric_cols].describe()
        stats.loc['variance'] = self.data[numeric_cols].var()
        stats.loc['skewness'] = self.data[numeric_cols].skew()
        stats.loc['kurtosis'] = self.data[numeric_cols].kurtosis()
        
        self.results['descriptive_stats'] = stats
        logger.info("Descriptive statistics calculated")
        return stats
    
    def calculate_loss_ratio(
        self, 
        group_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate Loss Ratio (TotalClaims / TotalPremium) by groups.
        
        Args:
            group_by: List of columns to group by (e.g., ['Province', 'VehicleType', 'Gender'])
            
        Returns:
            pd.DataFrame: Loss ratio by groups
        """
        if "TotalClaims" not in self.data.columns or "TotalPremium" not in self.data.columns:
            raise ValueError("TotalClaims and TotalPremium columns are required")
        
        # Calculate overall loss ratio
        overall_loss_ratio = (
            self.data["TotalClaims"].sum() / self.data["TotalPremium"].sum()
            if self.data["TotalPremium"].sum() > 0 else 0
        )
        
        self.results['overall_loss_ratio'] = overall_loss_ratio
        
        if group_by:
            # Filter valid group columns
            valid_groups = [col for col in group_by if col in self.data.columns]
            
            if valid_groups:
                loss_ratio_by_group = (
                    self.data.groupby(valid_groups)
                    .agg({
                        'TotalClaims': 'sum',
                        'TotalPremium': 'sum'
                    })
                    .reset_index()
                )
                loss_ratio_by_group['LossRatio'] = (
                    loss_ratio_by_group['TotalClaims'] / 
                    loss_ratio_by_group['TotalPremium'].replace(0, np.nan)
                )
                loss_ratio_by_group = loss_ratio_by_group.sort_values('LossRatio', ascending=False)
                
                self.results[f'loss_ratio_by_{"_".join(valid_groups)}'] = loss_ratio_by_group
                logger.info(f"Loss ratio calculated by {valid_groups}")
                return loss_ratio_by_group
        
        logger.info(f"Overall loss ratio: {overall_loss_ratio:.4f}")
        return pd.DataFrame({'LossRatio': [overall_loss_ratio]})
    
    def detect_outliers(
        self, 
        columns: Optional[List[str]] = None,
        method: str = "iqr"
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect outliers in numerical columns.
        
        Args:
            columns: List of columns to check. If None, checks all numeric columns.
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Dict: Dictionary with outlier information for each column
        """
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.data.columns]
        
        outliers = {}
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outlier_data = self.data[outlier_mask][[col]]
                
            elif method == "zscore":
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outlier_mask = z_scores > 3
                outlier_data = self.data[outlier_mask][[col]]
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if len(outlier_data) > 0:
                outliers[col] = {
                    'count': len(outlier_data),
                    'percentage': len(outlier_data) / len(self.data) * 100,
                    'data': outlier_data
                }
        
        self.results['outliers'] = outliers
        logger.info(f"Outlier detection completed for {len(outliers)} columns")
        return outliers
    
    def analyze_temporal_trends(
        self,
        date_column: str = "TransactionMonth",
        value_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Analyze temporal trends in the data.
        
        Args:
            date_column: Name of the date column
            value_columns: List of columns to analyze over time
            
        Returns:
            pd.DataFrame: Temporal trends
        """
        if date_column not in self.data.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        
        if value_columns is None:
            value_columns = ["TotalPremium", "TotalClaims"]
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column], errors='coerce')
        
        # Group by month
        monthly_trends = self.data.groupby(
            self.data[date_column].dt.to_period('M')
        )[value_columns].agg(['sum', 'mean', 'count']).reset_index()
        
        monthly_trends.columns = ['Month'] + [
            f'{col}_{agg}' for col in value_columns for agg in ['sum', 'mean', 'count']
        ]
        
        self.results['temporal_trends'] = monthly_trends
        logger.info("Temporal trends analyzed")
        return monthly_trends
    
    def analyze_vehicle_make_model_claims(self) -> pd.DataFrame:
        """
        Analyze claims by vehicle make and model.
        
        Returns:
            pd.DataFrame: Claims analysis by make/model
        """
        required_cols = ["make", "Model", "TotalClaims"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        vehicle_analysis = (
            self.data.groupby(["make", "Model"])
            .agg({
                'TotalClaims': ['sum', 'mean', 'count'],
                'TotalPremium': ['sum', 'mean'],
                'CustomValueEstimate': 'mean'
            })
            .reset_index()
        )
        
        vehicle_analysis.columns = [
            'Make', 'Model', 'TotalClaims_Sum', 'TotalClaims_Mean', 'TotalClaims_Count',
            'TotalPremium_Sum', 'TotalPremium_Mean', 'AvgCustomValue'
        ]
        
        vehicle_analysis['LossRatio'] = (
            vehicle_analysis['TotalClaims_Sum'] / 
            vehicle_analysis['TotalPremium_Sum'].replace(0, np.nan)
        )
        
        vehicle_analysis = vehicle_analysis.sort_values('TotalClaims_Sum', ascending=False)
        
        self.results['vehicle_analysis'] = vehicle_analysis
        logger.info("Vehicle make/model analysis completed")
        return vehicle_analysis
    
    def create_correlation_matrix(
        self,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create correlation matrix for numerical columns.
        
        Args:
            columns: List of columns to include. If None, uses all numeric columns.
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.data.columns]
        
        corr_matrix = self.data[numeric_cols].corr()
        self.results['correlation_matrix'] = corr_matrix
        logger.info("Correlation matrix created")
        return corr_matrix
    
    def plot_distribution(
        self,
        column: str,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot distribution of a column.
        
        Args:
            column: Column name to plot
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            matplotlib.Figure: The figure object
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        self.data[column].hist(bins=50, ax=axes[0], edgecolor='black')
        axes[0].set_title(f'Distribution of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')
        
        # Box plot
        self.data.boxplot(column=column, ax=axes[1])
        axes[1].set_title(f'Box Plot of {column}')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_loss_ratio_by_group(
        self,
        group_column: str,
        top_n: int = 10,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot loss ratio by a grouping column.
        
        Args:
            group_column: Column to group by
            top_n: Number of top groups to show
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            matplotlib.Figure: The figure object
        """
        if group_column not in self.data.columns:
            raise ValueError(f"Column '{group_column}' not found")
        
        loss_ratio = self.calculate_loss_ratio(group_by=[group_column])
        top_groups = loss_ratio.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(top_groups)), top_groups['LossRatio'], color='steelblue')
        ax.set_yticks(range(len(top_groups)))
        ax.set_yticklabels(top_groups[group_column].values)
        ax.set_xlabel('Loss Ratio')
        ax.set_title(f'Loss Ratio by {group_column} (Top {top_n})')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def generate_summary_report(self) -> Dict:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Dict: Summary report dictionary
        """
        report = {
            'data_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'results': self.results
        }
        
        if 'overall_loss_ratio' in self.results:
            report['overall_loss_ratio'] = self.results['overall_loss_ratio']
        
        logger.info("Summary report generated")
        return report

