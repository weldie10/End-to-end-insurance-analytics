"""
Model Evaluation Module

This module provides the ModelEvaluator class for comprehensive
model evaluation using regression and classification metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    A class for evaluating machine learning models with comprehensive metrics.
    
    Supports:
    - Regression metrics: RMSE, R², MAE
    - Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.evaluation_results: Dict = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_regression_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> Dict:
        """
        Evaluate a regression model.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dict: Evaluation metrics (RMSE, R², MAE)
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        mean_actual = np.mean(y_true)
        mean_predicted = np.mean(y_pred)
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'mean_actual': mean_actual,
            'mean_predicted': mean_predicted,
            'mean_error': mean_actual - mean_predicted
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        return results
    
    def evaluate_classification_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict:
        """
        Evaluate a classification model.
        
        Args:
            y_true: True target labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)
            model_name: Name of the model
            
        Returns:
            Dict: Evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        # ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_proba)
                results['roc_auc'] = roc_auc
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Classification report
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            results['classification_report'] = class_report
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare multiple models' performance.
        
        Returns:
            pd.DataFrame: Comparison of all evaluated models
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet.")
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            row = {'Model': model_name}
            row.update(results)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Remove non-scalar columns for display
        display_cols = [col for col in df.columns if col not in ['confusion_matrix', 'classification_report']]
        return df[display_cols]
    
    def get_best_model(self, metric: str = "r2_score") -> str:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            str: Name of the best model
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet.")
        
        best_model = None
        best_score = float('-inf') if metric in ['r2_score', 'accuracy', 'f1_score', 'roc_auc'] else float('inf')
        
        for model_name, results in self.evaluation_results.items():
            if metric in results:
                score = results[metric]
                if metric in ['r2_score', 'accuracy', 'f1_score', 'roc_auc']:
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                else:  # For RMSE, MAE (lower is better)
                    if score < best_score:
                        best_score = score
                        best_model = model_name
        
        return best_model

