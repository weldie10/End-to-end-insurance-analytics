"""
Generate descriptive statistics and data quality assessment for the report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphacare.data import DataLoader
from alphacare.eda import EDAAnalyzer
from alphacare.utils import setup_logging

# Setup
setup_logging(log_level="INFO")

def generate_statistics():
    """Generate descriptive statistics and data quality metrics."""
    print("=" * 60)
    print("Generating Descriptive Statistics and Data Quality Assessment")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    loader = DataLoader(data_path="Data")
    data = loader.load_data("MachineLearningRating_v3.txt")
    processed_data = loader.preprocess_data()
    print(f"   ✓ Loaded {len(processed_data):,} records")
    
    # Initialize EDA
    print("\n2. Calculating descriptive statistics...")
    eda = EDAAnalyzer(processed_data)
    
    # Calculate descriptive statistics
    stats = eda.calculate_descriptive_stats(columns=["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"])
    
    # Calculate loss ratio
    loss_ratio = eda.calculate_loss_ratio()
    overall_loss_ratio = eda.results.get('overall_loss_ratio', 0)
    
    # Data quality assessment
    print("\n3. Assessing data quality...")
    info = loader.get_data_info()
    
    # Missing values analysis
    missing_values = processed_data[["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"]].isnull().sum()
    missing_pct = (missing_values / len(processed_data) * 100).round(2)
    
    # Outlier detection
    print("\n4. Detecting outliers...")
    outliers = eda.detect_outliers(columns=["TotalPremium", "TotalClaims", "CustomValueEstimate"])
    
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print("\nTotalPremium Statistics:")
    print(stats["TotalPremium"].to_string())
    print("\nTotalClaims Statistics:")
    print(stats["TotalClaims"].to_string())
    
    print("\n" + "=" * 60)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 60)
    print("\nMissing Values:")
    for col in ["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"]:
        print(f"  {col}: {missing_values[col]:,} ({missing_pct[col]}%)")
    
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    for col, outlier_info in outliers.items():
        print(f"\n{col}:")
        print(f"  Outliers: {outlier_info['count']:,} ({outlier_info['percentage']:.2f}%)")
    
    print("\n" + "=" * 60)
    print("OVERALL LOSS RATIO")
    print("=" * 60)
    print(f"Overall Loss Ratio: {overall_loss_ratio:.4f} ({overall_loss_ratio*100:.2f}%)")
    
    # Save statistics to file
    output_file = Path(__file__).parent.parent / "reports" / "eda_statistics.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write("TotalPremium Statistics:\n")
        f.write(stats["TotalPremium"].to_string() + "\n\n")
        f.write("TotalClaims Statistics:\n")
        f.write(stats["TotalClaims"].to_string() + "\n\n")
        f.write("=" * 60 + "\n")
        f.write("DATA QUALITY ASSESSMENT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Missing Values:\n")
        for col in ["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"]:
            f.write(f"  {col}: {missing_values[col]:,} ({missing_pct[col]}%)\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("OVERALL LOSS RATIO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall Loss Ratio: {overall_loss_ratio:.4f} ({overall_loss_ratio*100:.2f}%)\n")
    
    print(f"\n✓ Statistics saved to: {output_file}")
    print("=" * 60)
    
    return {
        'stats': stats,
        'missing_values': missing_values,
        'missing_pct': missing_pct,
        'outliers': outliers,
        'overall_loss_ratio': overall_loss_ratio
    }

if __name__ == "__main__":
    generate_statistics()

