# Insurance Risk Analytics & Predictive Modeling

A comprehensive risk analytics and predictive modeling platform for AlphaCare Insurance Solutions. Analyzes historical car insurance claim data (Feb 2014 - Aug 2015) to identify low-risk segments, optimize premium pricing, and support data-driven business decisions.

## Features

- **Data Pipeline**: Automated data loading, preprocessing, and quality assessment
- **Exploratory Data Analysis**: Loss ratio analysis, outlier detection, temporal trends
- **Statistical Testing**: A/B hypothesis testing for risk differences across geographic and demographic dimensions
- **Machine Learning**: Predictive models for claims and premium optimization
- **Data Version Control**: DVC integration for reproducible data pipelines

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from alphacare.data import DataLoader
from alphacare.eda import EDAAnalyzer
from alphacare.statistics import ABRiskHypothesisTester
from alphacare.models import (
    InsuranceDataPreprocessor,
    ClaimSeverityPredictor,
    PremiumOptimizer,
    ClaimProbabilityPredictor,
    ModelEvaluator
)

# Load and preprocess data
loader = DataLoader(data_path="Data")
data = loader.load_data("MachineLearningRating_v3.txt")
processed = loader.preprocess_data()

# Perform EDA
eda = EDAAnalyzer(processed)
loss_ratio = eda.calculate_loss_ratio(group_by=["Province", "VehicleType"])

# Run A/B hypothesis tests
tester = ABRiskHypothesisTester(processed, alpha=0.05)
results = tester.run_all_tests()

# Build predictive models
preprocessor = InsuranceDataPreprocessor()
data_processed = preprocessor.engineer_features(processed)
data_processed = preprocessor.handle_missing_values(data_processed)
data_processed = preprocessor.encode_categorical_features(data_processed)

# Train claim severity model
severity_predictor = ClaimSeverityPredictor(model_type="xgboost")
# ... train and evaluate models
```

## Project Structure

```
src/alphacare/          # Core OOP classes
├── data/               # Data loading and preprocessing
├── eda/                # Exploratory data analysis
├── statistics/         # A/B hypothesis testing
├── models/             # Machine learning models
└── utils/              # Utility functions

data/                   # DVC-tracked data (raw/processed)
notebooks/              # Analysis notebooks
scripts/                 # Analysis scripts
reports/                 # Generated reports and visualizations
```

## Core Classes

### Data & Analysis
- **`DataLoader`**: Data loading and preprocessing
- **`EDAAnalyzer`**: Exploratory data analysis and visualizations
- **`ABRiskHypothesisTester`**: A/B statistical hypothesis testing for risk drivers

### Predictive Modeling
- **`InsuranceDataPreprocessor`**: Feature engineering, encoding, and data preparation
- **`ClaimSeverityPredictor`**: Predicts claim amounts for policies with claims (regression)
- **`PremiumOptimizer`**: Predicts optimal premium values (regression)
- **`ClaimProbabilityPredictor`**: Predicts probability of claim occurrence (classification)
- **`ModelEvaluator`**: Comprehensive model evaluation (RMSE, R², Accuracy, F1, etc.)
- **`ModelInterpreter`**: SHAP/LIME interpretability analysis

### Legacy Models
- **`LinearRegressionModel`**: Zipcode-based claim prediction
- **`ModelTrainer`**: Premium prediction models

## Key Findings

- **Portfolio Loss Ratio**: 104.77% (indicating unprofitability)
- **Risk Segmentation**: Significant variations identified across provinces and zipcodes
- **Geographic Pricing**: Province and zipcode-based premium adjustments recommended
- **Gender Analysis**: No significant risk differences detected (gender-neutral pricing supported)
- **Predictive Models**: XGBoost and Random Forest models show strong performance for claim severity and premium prediction

## Model Types

The platform supports three main predictive modeling tasks:

1. **Claim Severity Prediction**: Predicts the financial liability (TotalClaims) for policies that have claims
   - Models: Linear Regression, Random Forest, XGBoost
   - Metrics: RMSE, R², MAE

2. **Premium Optimization**: Predicts appropriate premium values for insurance policies
   - Models: Linear Regression, Random Forest, XGBoost
   - Metrics: RMSE, R², MAE
   - Supports risk-based premium calculation

3. **Claim Probability Prediction**: Predicts the probability of a claim occurring (binary classification)
   - Models: Logistic Regression, Random Forest, XGBoost
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

## Model Interpretability

- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-level explanations
- Feature importance analysis for all tree-based models

## Technologies

Python 3.8+, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, LIME, Matplotlib, Seaborn, SciPy, DVC, Jupyter

## License

Proprietary - AlphaCare Insurance Solutions
