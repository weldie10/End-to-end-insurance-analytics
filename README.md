# Insurance Risk Analytics & Predictive Modeling

Risk analytics and predictive modeling for **AlphaCare Insurance Solutions** car insurance in South Africa. Analyze historical claim data (Feb 2014 - Aug 2015) to discover low-risk segments and optimize premium pricing.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run example
python example_usage.py
```

## Project Structure

```
src/alphacare/          # OOP classes (DataLoader, EDAAnalyzer, HypothesisTester, ModelTrainer)
data/                   # DVC-tracked data (raw/processed)
notebooks/              # Analysis notebooks
Data/                   # Source data file
```

## Key Classes

- **`DataLoader`**: Load and preprocess data from `Data/MachineLearningRating_v3.txt`
- **`EDAAnalyzer`**: EDA, loss ratio analysis, outlier detection, visualizations
- **`HypothesisTester`**: A/B testing for risk differences (provinces, zipcodes, gender)
- **`LinearRegressionModel`**: Predict claims by zipcode
- **`ModelTrainer`**: ML models for premium prediction (Random Forest, Gradient Boosting)

## Usage

```python
from alphacare.data import DataLoader
from alphacare.eda import EDAAnalyzer
from alphacare.statistics import HypothesisTester

# Load data
loader = DataLoader(data_path="Data")
data = loader.load_data("MachineLearningRating_v3.txt")
processed = loader.preprocess_data()

# EDA
eda = EDAAnalyzer(processed)
loss_ratio = eda.calculate_loss_ratio(group_by=["Province", "VehicleType", "Gender"])

# Hypothesis testing
tester = HypothesisTester(processed, alpha=0.05)
results = tester.run_all_tests()
```

## DVC Setup

```bash
dvc init
dvc remote add -d localstorage $(pwd)/data_storage
dvc add data/raw/MachineLearningRating_v3.txt
dvc push
```

## Analysis Questions

1. Overall Loss Ratio and variation by Province, VehicleType, Gender
2. Distributions and outliers in financial variables
3. Temporal trends in claim frequency/severity
4. Vehicle make/model with highest/lowest claims
5. Hypothesis tests: Risk differences by province, zipcode, gender; margin differences by zipcode

## Technologies

Python 3.8+, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, SciPy, DVC, Jupyter

## Key Dates

- Interim Submission: 8:00 PM UTC, Sunday, 07 Dec 2025
- Final Submission: 8:00 PM UTC, Tuesday, 09 Dec 2025

---
**Team**: Kerod, Mahbubah, Filimon | **License**: Proprietary - AlphaCare Insurance Solutions
