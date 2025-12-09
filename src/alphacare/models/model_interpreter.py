"""
Model Interpretability Module

This module provides the ModelInterpreter class for explaining model predictions
using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import SHAP and LIME (optional dependencies)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class ModelInterpreter:
    """
    A class for interpreting machine learning model predictions.
    
    Provides:
    - SHAP values for global and local interpretability
    - LIME explanations for local interpretability
    - Feature importance analysis
    """
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Initialize the ModelInterpreter.
        
        Args:
            model: Trained model to interpret
            X_train: Training data used for model training
            feature_names: List of feature names (optional)
        """
        self.model = model
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.feature_names = feature_names if feature_names else list(range(self.X_train.shape[1]))
        self.shap_explainer = None
        self.lime_explainer = None
        
        logger.info("ModelInterpreter initialized")
    
    def create_shap_explainer(self, explainer_type: str = "tree"):
        """
        Create a SHAP explainer for the model.
        
        Args:
            explainer_type: Type of explainer ('tree', 'linear', 'kernel')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        try:
            if explainer_type == "tree" and hasattr(self.model, 'predict'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif explainer_type == "linear":
                self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
            else:
                # Use KernelExplainer as fallback
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict,
                    shap.sample(self.X_train, min(100, len(self.X_train)))
                )
            
            logger.info(f"SHAP {explainer_type} explainer created")
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            raise
    
    def calculate_shap_values(self, X: pd.DataFrame, max_samples: int = 1000) -> np.ndarray:
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Data to explain
            max_samples: Maximum number of samples to explain (for performance)
            
        Returns:
            np.ndarray: SHAP values
        """
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        # Limit samples for performance
        if len(X_values) > max_samples:
            indices = np.random.choice(len(X_values), max_samples, replace=False)
            X_values = X_values[indices]
        
        shap_values = self.shap_explainer.shap_values(X_values)
        
        # Handle multi-output models
        if isinstance(shap_values, list):
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        logger.info(f"Calculated SHAP values for {len(X_values)} samples")
        return shap_values
    
    def get_top_shap_features(
        self,
        X: pd.DataFrame,
        top_n: int = 10,
        max_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Get top features by mean absolute SHAP value.
        
        Args:
            X: Data to explain
            top_n: Number of top features to return
            max_samples: Maximum number of samples to explain
            
        Returns:
            pd.DataFrame: Top features with their mean SHAP values
        """
        shap_values = self.calculate_shap_values(X, max_samples)
        
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names[:len(mean_abs_shap)],
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False).head(top_n)
        
        return feature_importance
    
    def explain_prediction_shap(
        self,
        X_instance: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Explain a single prediction using SHAP.
        
        Args:
            X_instance: Single instance to explain
            feature_names: Feature names (optional)
            
        Returns:
            Dict: Explanation with feature contributions
        """
        if self.shap_explainer is None:
            self.create_shap_explainer()
        
        X_values = X_instance.values if isinstance(X_instance, pd.DataFrame) else X_instance
        shap_values = self.shap_explainer.shap_values(X_values)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        feature_names = feature_names or self.feature_names
        
        explanation = {
            'feature_names': feature_names[:len(shap_values[0])],
            'shap_values': shap_values[0].tolist(),
            'base_value': float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else None
        }
        
        return explanation
    
    def create_lime_explainer(self, mode: str = "regression"):
        """
        Create a LIME explainer for the model.
        
        Args:
            mode: Type of problem ('regression' or 'classification')
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                mode=mode,
                discretize_continuous=True
            )
            logger.info(f"LIME explainer created for {mode} mode")
        except Exception as e:
            logger.error(f"Error creating LIME explainer: {e}")
            raise
    
    def explain_prediction_lime(
        self,
        X_instance: pd.DataFrame,
        num_features: int = 10
    ) -> Dict:
        """
        Explain a single prediction using LIME.
        
        Args:
            X_instance: Single instance to explain
            num_features: Number of top features to show
            
        Returns:
            Dict: Explanation with feature contributions
        """
        if self.lime_explainer is None:
            mode = "regression" if hasattr(self.model, 'predict') else "classification"
            self.create_lime_explainer(mode)
        
        X_values = X_instance.values if isinstance(X_instance, pd.DataFrame) else X_instance
        
        explanation = self.lime_explainer.explain_instance(
            X_values[0],
            self.model.predict,
            num_features=num_features
        )
        
        # Extract feature contributions
        feature_contributions = explanation.as_list()
        
        result = {
            'prediction': float(self.model.predict(X_values)[0]),
            'feature_contributions': [
                {'feature': feat, 'contribution': contrib}
                for feat, contrib in feature_contributions
            ]
        }
        
        return result
    
    def generate_interpretation_report(
        self,
        X_sample: pd.DataFrame,
        top_n: int = 10,
        use_shap: bool = True,
        use_lime: bool = False
    ) -> Dict:
        """
        Generate a comprehensive interpretation report.
        
        Args:
            X_sample: Sample data to explain
            top_n: Number of top features to report
            use_shap: Whether to use SHAP
            use_lime: Whether to use LIME
            
        Returns:
            Dict: Comprehensive interpretation report
        """
        report = {}
        
        if use_shap and SHAP_AVAILABLE:
            try:
                top_shap_features = self.get_top_shap_features(X_sample, top_n)
                report['shap_top_features'] = top_shap_features.to_dict('records')
                report['shap_summary'] = {
                    'total_features': len(self.feature_names),
                    'top_features_count': top_n
                }
            except Exception as e:
                logger.error(f"Error generating SHAP report: {e}")
                report['shap_error'] = str(e)
        
        if use_lime and LIME_AVAILABLE:
            try:
                # Explain a few sample instances
                sample_indices = np.random.choice(len(X_sample), min(3, len(X_sample)), replace=False)
                lime_explanations = []
                for idx in sample_indices:
                    explanation = self.explain_prediction_lime(X_sample.iloc[[idx]])
                    lime_explanations.append(explanation)
                report['lime_sample_explanations'] = lime_explanations
            except Exception as e:
                logger.error(f"Error generating LIME report: {e}")
                report['lime_error'] = str(e)
        
        return report

