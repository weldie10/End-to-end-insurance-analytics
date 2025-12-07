# Insurance Risk Analytics & Predictive Modeling

Risk analytics and predictive modeling platform for AlphaCare Insurance Solutions. Analyzes historical car insurance claim data (Feb 2014 - Aug 2015) to identify low-risk segments and optimize premium pricing strategies.

## Features

- **Data Loading & Preprocessing**: Automated data pipeline with quality assessment
- **Exploratory Data Analysis**: Loss ratio analysis, outlier detection, temporal trends
- **Statistical Testing**: A/B hypothesis testing for risk differences across dimensions
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
from alphacare.statistics import HypothesisTester

# Load and preprocess data
loader = DataLoader(data_path="Data")
data = loader.load_data("MachineLearningRating_v3.txt")
processed = loader.preprocess_data()

# Perform EDA
eda = EDAAnalyzer(processed)
loss_ratio = eda.calculate_loss_ratio(group_by=["Province", "VehicleType", "Gender"])

# Run hypothesis tests
tester = HypothesisTester(processed, alpha=0.05)
results = tester.run_all_tests()
```

## Project Structure

```
src/alphacare/     # Core OOP classes
data/              # DVC-tracked data (raw/processed)
notebooks/         # Analysis notebooks
Data/              # Source data files
```

## Core Classes

- **`DataLoader`**: Data loading and preprocessing
- **`EDAAnalyzer`**: Exploratory data analysis and visualizations
- **`HypothesisTester`**: Statistical hypothesis testing
- **`LinearRegressionModel`**: Zipcode-based claim prediction
- **`ModelTrainer`**: Premium prediction models

## Data Version Control

Data files are tracked using DVC:

```bash
dvc init
dvc remote add -d localstorage $(pwd)/data_storage
dvc add data/raw/MachineLearningRating_v3.txt
dvc push
```

## Technologies

Python 3.8+, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy, DVC, Jupyter

---

**License**: Proprietary - AlphaCare Insurance Solutions
