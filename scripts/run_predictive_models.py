"""
Comprehensive Predictive Modeling Script for Task 4

This script builds and evaluates predictive models for:
1. Claim Severity Prediction (Risk Model) - Predict TotalClaims for policies with claims
2. Premium Optimization (Pricing Framework) - Predict appropriate premium values
3. Claim Probability Prediction (Binary Classification) - Predict probability of claim occurrence

Models implemented:
- Linear Regression
- Random Forest
- XGBoost

Evaluation includes:
- Regression metrics: RMSE, R², MAE
- Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- SHAP/LIME interpretability analysis
"""

import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphacare.data import DataLoader
from alphacare.models import (
    InsuranceDataPreprocessor,
    ClaimSeverityPredictor,
    PremiumOptimizer,
    ClaimProbabilityPredictor,
    ModelEvaluator,
    ModelInterpreter
)
from alphacare.utils import setup_logging
from sklearn.model_selection import train_test_split

# Setup logging
logger = setup_logging(log_level="INFO")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def format_metric(value: float, metric_name: str = "") -> str:
    """Format metric value for display."""
    if metric_name in ['r2_score', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"


def train_and_evaluate_claim_severity_models(
    data: pd.DataFrame,
    preprocessor: InsuranceDataPreprocessor,
    evaluator: ModelEvaluator,
    output_dir: Path
) -> Dict:
    """Train and evaluate claim severity prediction models."""
    print_subsection("Claim Severity Prediction Models")
    
    # Filter to policies with claims
    claims_data = data[data['TotalClaims'] > 0].copy()
    print(f"Training on {len(claims_data):,} policies with claims")
    
    # Prepare data
    claims_data_processed = preprocessor.engineer_features(claims_data)
    claims_data_processed = preprocessor.handle_missing_values(claims_data_processed)
    claims_data_processed = preprocessor.encode_categorical_features(claims_data_processed)
    
    X, y = preprocessor.prepare_features_for_training(
        claims_data_processed,
        target_column='TotalClaims',
        exclude_columns=['TotalPremium', 'Margin', 'LossRatio', 'HasClaim']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    models_to_train = ['linear', 'random_forest', 'xgboost']
    results = {}
    
    for model_type in models_to_train:
        print(f"\n  Training {model_type} model...")
        predictor = ClaimSeverityPredictor(model_type=model_type)
        
        # Train
        predictor.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate
        test_results = evaluator.evaluate_regression_model(
            y_test.values,
            predictor.training_results['test_predictions'],
            model_name=f"ClaimSeverity_{model_type}"
        )
        
        results[model_type] = {
            'predictor': predictor,
            'evaluation': test_results,
            'feature_importance': predictor.get_feature_importance(top_n=10) if hasattr(predictor.model, 'feature_importances_') else None
        }
        
        print(f"    RMSE: {test_results['rmse']:.2f}, R²: {test_results['r2_score']:.4f}")
    
    return results


def train_and_evaluate_premium_models(
    data: pd.DataFrame,
    preprocessor: InsuranceDataPreprocessor,
    evaluator: ModelEvaluator,
    output_dir: Path
) -> Dict:
    """Train and evaluate premium optimization models."""
    print_subsection("Premium Optimization Models")
    
    # Prepare data
    data_processed = preprocessor.engineer_features(data)
    data_processed = preprocessor.handle_missing_values(data_processed)
    data_processed = preprocessor.encode_categorical_features(data_processed)
    
    X, y = preprocessor.prepare_features_for_training(
        data_processed,
        target_column='TotalPremium',
        exclude_columns=['TotalClaims', 'Margin', 'LossRatio', 'HasClaim', 'ClaimAmount']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    models_to_train = ['linear', 'random_forest', 'xgboost']
    results = {}
    
    for model_type in models_to_train:
        print(f"\n  Training {model_type} model...")
        optimizer = PremiumOptimizer(model_type=model_type)
        
        # Train
        optimizer.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate
        test_results = evaluator.evaluate_regression_model(
            y_test.values,
            optimizer.training_results['test_predictions'],
            model_name=f"Premium_{model_type}"
        )
        
        results[model_type] = {
            'optimizer': optimizer,
            'evaluation': test_results,
            'feature_importance': optimizer.get_feature_importance(top_n=10) if hasattr(optimizer.model, 'feature_importances_') else None
        }
        
        print(f"    RMSE: {test_results['rmse']:.2f}, R²: {test_results['r2_score']:.4f}")
    
    return results


def train_and_evaluate_claim_probability_models(
    data: pd.DataFrame,
    preprocessor: InsuranceDataPreprocessor,
    evaluator: ModelEvaluator,
    output_dir: Path
) -> Dict:
    """Train and evaluate claim probability prediction models."""
    print_subsection("Claim Probability Prediction Models")
    
    # Create binary target
    data_with_target = data.copy()
    data_with_target['HasClaim'] = (data_with_target['TotalClaims'] > 0).astype(int)
    
    # Prepare data
    data_processed = preprocessor.engineer_features(data_with_target)
    data_processed = preprocessor.handle_missing_values(data_processed)
    data_processed = preprocessor.encode_categorical_features(data_processed)
    
    X, y = preprocessor.prepare_features_for_training(
        data_processed,
        target_column='HasClaim',
        exclude_columns=['TotalClaims', 'TotalPremium', 'Margin', 'LossRatio', 'ClaimAmount']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    models_to_train = ['logistic', 'random_forest', 'xgboost']
    results = {}
    
    for model_type in models_to_train:
        print(f"\n  Training {model_type} model...")
        predictor = ClaimProbabilityPredictor(model_type=model_type)
        
        # Train
        predictor.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Evaluate
        test_results = evaluator.evaluate_classification_model(
            y_test.values,
            predictor.training_results['test_predictions'],
            predictor.training_results.get('test_probabilities'),
            model_name=f"ClaimProbability_{model_type}"
        )
        
        results[model_type] = {
            'predictor': predictor,
            'evaluation': test_results,
            'feature_importance': predictor.get_feature_importance(top_n=10) if hasattr(predictor.model, 'feature_importances_') else None
        }
        
        print(f"    Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1_score']:.4f}")
    
    return results


def generate_shap_analysis(
    best_model,
    X_sample: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> Dict:
    """Generate SHAP analysis for model interpretability."""
    print_subsection(f"SHAP Analysis for {model_name}")
    
    try:
        interpreter = ModelInterpreter(best_model.model, X_sample, list(X_sample.columns))
        interpreter.create_shap_explainer("tree")
        
        # Get top features
        top_features = interpreter.get_top_shap_features(X_sample, top_n=10)
        
        # Generate report
        report = interpreter.generate_interpretation_report(
            X_sample.head(100),
            top_n=10,
            use_shap=True
        )
        
        print(f"  Top 5 Features by SHAP Value:")
        for idx, row in top_features.head(5).iterrows():
            print(f"    {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")
        
        return {
            'top_features': top_features.to_dict('records'),
            'report': report
        }
    except Exception as e:
        logger.warning(f"SHAP analysis failed for {model_name}: {e}")
        return {'error': str(e)}


def generate_model_comparison_visualization(evaluator: ModelEvaluator, output_dir: Path):
    """Generate visualization comparing all models."""
    comparison = evaluator.compare_models()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Regression models comparison
    reg_models = comparison[comparison['Model'].str.contains('ClaimSeverity|Premium')]
    if len(reg_models) > 0:
        ax = axes[0, 0]
        x_pos = np.arange(len(reg_models))
        ax.bar(x_pos, reg_models['r2_score'], color='steelblue')
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison (Regression Models)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(reg_models['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[0, 1]
        ax.bar(x_pos, reg_models['rmse'], color='coral')
        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison (Regression Models)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(reg_models['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    # Classification models comparison
    clf_models = comparison[comparison['Model'].str.contains('ClaimProbability')]
    if len(clf_models) > 0:
        ax = axes[1, 0]
        x_pos = np.arange(len(clf_models))
        ax.bar(x_pos, clf_models['accuracy'], color='green')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison (Classification Models)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(clf_models['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[1, 1]
        ax.bar(x_pos, clf_models['f1_score'], color='purple')
        ax.set_xlabel('Model')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score Comparison (Classification Models)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(clf_models['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_comparison.png")


def main():
    """Main function to run all predictive modeling tasks."""
    print_section("Task 4: Predictive Modeling for Risk-Based Pricing")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "reports" / "predictive_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print_section("Step 1: Data Loading and Preprocessing")
    loader = DataLoader(data_path=project_root / "Data")
    data = loader.load_data("MachineLearningRating_v3.txt")
    processed_data = loader.preprocess_data()
    print(f"✓ Loaded {len(processed_data):,} records")
    
    # Initialize preprocessor
    preprocessor = InsuranceDataPreprocessor(encoding_method="one_hot")
    evaluator = ModelEvaluator()
    
    # Step 2: Train Claim Severity Models
    print_section("Step 2: Claim Severity Prediction Models")
    severity_results = train_and_evaluate_claim_severity_models(
        processed_data, preprocessor, evaluator, output_dir
    )
    
    # Step 3: Train Premium Optimization Models
    print_section("Step 3: Premium Optimization Models")
    premium_results = train_and_evaluate_premium_models(
        processed_data, preprocessor, evaluator, output_dir
    )
    
    # Step 4: Train Claim Probability Models
    print_section("Step 4: Claim Probability Prediction Models")
    probability_results = train_and_evaluate_claim_probability_models(
        processed_data, preprocessor, evaluator, output_dir
    )
    
    # Step 5: Model Comparison
    print_section("Step 5: Model Comparison and Evaluation")
    comparison = evaluator.compare_models()
    print("\nModel Performance Comparison:")
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\n✓ Comparison saved to: {output_dir / 'model_comparison.csv'}")
    
    # Generate visualization
    generate_model_comparison_visualization(evaluator, output_dir)
    
    # Step 6: Model Interpretability (SHAP)
    print_section("Step 6: Model Interpretability Analysis")
    
    # Find best models
    best_severity = evaluator.get_best_model("r2_score") if any('ClaimSeverity' in m for m in evaluator.evaluation_results.keys()) else None
    best_premium = evaluator.get_best_model("r2_score") if any('Premium' in m for m in evaluator.evaluation_results.keys()) else None
    best_probability = evaluator.get_best_model("accuracy") if any('ClaimProbability' in m for m in evaluator.evaluation_results.keys()) else None
    
    shap_results = {}
    
    if best_severity:
        model_type = best_severity.split('_')[-1]
        sample_data = processed_data[processed_data['TotalClaims'] > 0].head(100)
        sample_processed = preprocessor.engineer_features(sample_data)
        sample_processed = preprocessor.handle_missing_values(sample_processed)
        sample_processed = preprocessor.encode_categorical_features(sample_processed)
        X_sample, _ = preprocessor.prepare_features_for_training(
            sample_processed,
            target_column='TotalClaims',
            exclude_columns=['TotalPremium', 'Margin', 'LossRatio', 'HasClaim']
        )
        X_sample_scaled, _ = preprocessor.scale_features(X_sample)
        X_sample_scaled = pd.DataFrame(X_sample_scaled, columns=X_sample.columns)
        
        shap_results['claim_severity'] = generate_shap_analysis(
            severity_results[model_type]['predictor'],
            X_sample_scaled,
            best_severity,
            output_dir
        )
    
    # Step 7: Save Results
    print_section("Step 7: Saving Results")
    
    # Compile all results
    all_results = {
        'claim_severity_models': {
            k: {
                'evaluation': v['evaluation'],
                'top_features': v['feature_importance'].to_dict('records') if v['feature_importance'] is not None else None
            }
            for k, v in severity_results.items()
        },
        'premium_models': {
            k: {
                'evaluation': v['evaluation'],
                'top_features': v['feature_importance'].to_dict('records') if v['feature_importance'] is not None else None
            }
            for k, v in premium_results.items()
        },
        'claim_probability_models': {
            k: {
                'evaluation': v['evaluation'],
                'top_features': v['feature_importance'].to_dict('records') if v['feature_importance'] is not None else None
            }
            for k, v in probability_results.items()
        },
        'shap_analysis': shap_results,
        'best_models': {
            'claim_severity': best_severity,
            'premium': best_premium,
            'claim_probability': best_probability
        }
    }
    
    # Save to JSON
    results_path = output_dir / "modeling_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"✓ Results saved to: {results_path}")
    
    print_section("Task 4 Complete!")
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext Steps:")
    print("  1. Review model comparison results")
    print("  2. Examine SHAP interpretability analysis")
    print("  3. Update REPORT.md with Task 4 findings")


if __name__ == "__main__":
    main()

