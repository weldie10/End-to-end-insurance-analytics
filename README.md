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

- **`DataLoader`**: Data loading and preprocessing
- **`EDAAnalyzer`**: Exploratory data analysis and visualizations
- **`ABRiskHypothesisTester`**: A/B statistical hypothesis testing for risk drivers
- **`LinearRegressionModel`**: Zipcode-based claim prediction
- **`ModelTrainer`**: Premium prediction models

## Key Findings

- **Portfolio Loss Ratio**: 104.77% (indicating unprofitability)
- **Risk Segmentation**: Significant variations identified across provinces and zipcodes
- **Geographic Pricing**: Province and zipcode-based premium adjustments recommended
- **Gender Analysis**: No significant risk differences detected (gender-neutral pricing supported)

## Technologies

Python 3.8+, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy, DVC, Jupyter

## License

Proprietary - AlphaCare Insurance Solutions
