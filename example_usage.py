"""
Example usage script demonstrating the OOP structure of AlphaCare Insurance Analytics.

This script shows how to use the various classes for data loading, EDA, 
hypothesis testing, and model training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alphacare.data import DataLoader
from alphacare.eda import EDAAnalyzer
from alphacare.statistics import HypothesisTester
from alphacare.models import LinearRegressionModel, ModelTrainer
from alphacare.utils import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")


def main():
    """Main function demonstrating the OOP usage."""
    
    print("=" * 80)
    print("AlphaCare Insurance Analytics - Example Usage")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    loader = DataLoader(data_path="Data")
    data = loader.load_data("MachineLearningRating_v3.txt")
    processed_data = loader.preprocess_data()
    print(f"✓ Loaded {len(processed_data)} rows")
    
    # Step 2: Exploratory Data Analysis
    print("\n[Step 2] Performing Exploratory Data Analysis...")
    eda = EDAAnalyzer(processed_data)
    
    # Calculate descriptive statistics
    stats = eda.calculate_descriptive_stats()
    print(f"✓ Calculated descriptive statistics for {len(stats.columns)} columns")
    
    # Calculate loss ratio
    loss_ratio = eda.calculate_loss_ratio()
    print(f"✓ Overall Loss Ratio: {eda.results['overall_loss_ratio']:.4f}")
    
    # Loss ratio by groups
    loss_ratio_by_group = eda.calculate_loss_ratio(group_by=["Province", "VehicleType"])
    print(f"✓ Calculated loss ratio by Province and VehicleType")
    
    # Step 3: Hypothesis Testing
    print("\n[Step 3] Performing Hypothesis Tests...")
    tester = HypothesisTester(processed_data, alpha=0.05)
    test_results = tester.run_all_tests()
    
    # Print test summaries
    summary = tester.get_results_summary()
    print("\nHypothesis Test Results:")
    print(summary.to_string())
    
    # Step 4: Linear Regression by Zipcode
    print("\n[Step 4] Fitting Linear Regression Models by Zipcode...")
    lr_model = LinearRegressionModel()
    models = lr_model.fit_by_zipcode(
        data=processed_data,
        target_column="TotalClaims",
        min_samples=10
    )
    print(f"✓ Fitted {len(models)} models")
    
    # Get model summary
    model_summary = lr_model.get_model_summary()
    print("\nTop 5 Models by R² Score:")
    print(model_summary.head(5).to_string())
    
    # Step 5: Premium Prediction Model
    print("\n[Step 5] Training Premium Prediction Model...")
    trainer = ModelTrainer(model_type="random_forest")
    training_results = trainer.train(
        data=processed_data,
        target_column="TotalPremium",
        test_size=0.2
    )
    
    print(f"✓ Model trained")
    print(f"  Test R² Score: {training_results['test_r2']:.4f}")
    print(f"  Test RMSE: {training_results['test_rmse']:.4f}")
    
    # Feature importance
    feature_importance = trainer.get_feature_importance(top_n=10)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.to_string())
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

