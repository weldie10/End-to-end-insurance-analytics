# End-to-End Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project provides comprehensive risk analytics and predictive modeling capabilities for **AlphaCare Insurance Solutions (ACIS)**, focusing on car insurance planning and marketing in South Africa. The objective is to analyze historical insurance claim data to optimize marketing strategies and discover "low-risk" segments for premium reduction, thereby attracting new clients.

## Business Objectives

- Analyze historical insurance claim data (February 2014 - August 2015)
- Discover low-risk segments for premium optimization
- Build predictive models for optimal premium pricing
- Perform A/B hypothesis testing to validate risk differences
- Provide actionable insights for marketing strategy optimization

## Project Structure

```
End-to-end-insurance-analytics/
├── Data/                          # Raw data files
│   └── MachineLearningRating_v3.txt
├── data/                          # Processed data (tracked by DVC)
│   ├── raw/                       # Raw data versions
│   └── processed/                 # Processed data
├── src/                           # Source code
│   └── alphacare/                 # Main package
│       ├── data/                  # Data loading and preprocessing
│       │   ├── __init__.py
│       │   └── data_loader.py
│       ├── eda/                   # Exploratory Data Analysis
│       │   ├── __init__.py
│       │   └── eda_analyzer.py
│       ├── statistics/            # Statistical testing
│       │   ├── __init__.py
│       │   └── hypothesis_tester.py
│       ├── models/                # Machine learning models
│       │   ├── __init__.py
│       │   ├── linear_regression_model.py
│       │   └── model_trainer.py
│       └── utils/                 # Utilities
│           ├── __init__.py
│           ├── logger_config.py
│           └── dvc_manager.py
├── notebooks/                     # Jupyter notebooks for analysis
├── reports/                       # Analysis reports and visualizations
├── tests/                         # Unit tests
├── .github/                       # GitHub Actions workflows
│   └── workflows/
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Features

### 1. Data Loading & Preprocessing (`DataLoader`)
- Load insurance data from various formats
- Automatic data type conversion
- Missing value handling
- Data quality assessment

### 2. Exploratory Data Analysis (`EDAAnalyzer`)
- Descriptive statistics
- Loss ratio calculations by various dimensions
- Outlier detection
- Temporal trend analysis
- Vehicle make/model analysis
- Correlation analysis
- Beautiful visualizations

### 3. Statistical Hypothesis Testing (`HypothesisTester`)
- Test risk differences across provinces
- Test risk differences between zipcodes
- Test margin (profit) differences between zipcodes
- Test risk differences between genders
- Comprehensive test results reporting

### 4. Machine Learning Models
- **Linear Regression by Zipcode** (`LinearRegressionModel`): Predicts total claims for each zipcode
- **Premium Prediction Model** (`ModelTrainer`): Predicts optimal premium values using:
  - Car features (type, make, model, specifications)
  - Owner features (demographics, account info)
  - Location features (province, zipcode, zones)
  - Plan features (coverage, terms, etc.)

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- DVC (Data Version Control)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd End-to-end-insurance-analytics
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## Usage

### Basic Data Loading

```python
from alphacare.data import DataLoader

# Initialize data loader
loader = DataLoader(data_path="Data")

# Load data
data = loader.load_data("MachineLearningRating_v3.txt")

# Get data information
info = loader.get_data_info()

# Preprocess data
processed_data = loader.preprocess_data()
```

### Exploratory Data Analysis

```python
from alphacare.eda import EDAAnalyzer

# Initialize EDA analyzer
eda = EDAAnalyzer(processed_data)

# Calculate descriptive statistics
stats = eda.calculate_descriptive_stats()

# Calculate loss ratio by province, vehicle type, and gender
loss_ratio = eda.calculate_loss_ratio(group_by=["Province", "VehicleType", "Gender"])

# Detect outliers
outliers = eda.detect_outliers(columns=["TotalClaims", "TotalPremium"])

# Analyze temporal trends
trends = eda.analyze_temporal_trends()

# Analyze vehicle make/model claims
vehicle_analysis = eda.analyze_vehicle_make_model_claims()
```

### Hypothesis Testing

```python
from alphacare.statistics import HypothesisTester

# Initialize hypothesis tester
tester = HypothesisTester(processed_data, alpha=0.05)

# Run all tests
results = tester.run_all_tests()

# Get summary
summary = tester.get_results_summary()
print(summary)
```

### Machine Learning Models

#### Linear Regression by Zipcode

```python
from alphacare.models import LinearRegressionModel

# Initialize model
lr_model = LinearRegressionModel()

# Fit models for each zipcode
models = lr_model.fit_by_zipcode(
    data=processed_data,
    target_column="TotalClaims",
    min_samples=10
)

# Get model summary
summary = lr_model.get_model_summary()
print(summary)
```

#### Premium Prediction Model

```python
from alphacare.models import ModelTrainer

# Initialize model trainer
trainer = ModelTrainer(model_type="random_forest")

# Train model
results = trainer.train(
    data=processed_data,
    target_column="TotalPremium",
    test_size=0.2
)

# Get feature importance
feature_importance = trainer.get_feature_importance(top_n=20)
print(feature_importance)

# Save model
trainer.save_model("models/premium_predictor.pkl")
```

## Data Version Control (DVC)

### Initialize DVC

```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d localstorage /path/to/your/local/storage

# Add data files
dvc add Data/MachineLearningRating_v3.txt

# Commit DVC files
git add Data/MachineLearningRating_v3.txt.dvc .dvc
git commit -m "Add data files to DVC"

# Push to remote
dvc push
```

Or use the DVCManager class:

```python
from alphacare.utils import DVCManager

dvc = DVCManager()
dvc.init_dvc()
dvc.add_remote("localstorage", "/path/to/storage", default=True)
dvc.add_data("Data/MachineLearningRating_v3.txt")
dvc.push()
```

## Key Analysis Questions

1. **What is the overall Loss Ratio (TotalClaims / TotalPremium) for the portfolio?**
   - How does it vary by Province, VehicleType, and Gender?

2. **What are the distributions of key financial variables?**
   - Are there outliers in TotalClaims or CustomValueEstimate?

3. **Are there temporal trends?**
   - Did claim frequency or severity change over the 18-month period?

4. **Which vehicle makes/models are associated with highest and lowest claim amounts?**

5. **Hypothesis Testing:**
   - Are there risk differences across provinces?
   - Are there risk differences between zipcodes?
   - Is there margin difference between zipcodes?
   - Are there risk differences between women and men?

## Deliverables

- [x] Git repository with proper version control
- [x] Comprehensive README
- [x] Modular OOP code structure
- [x] Data Version Control (DVC) setup
- [ ] Exploratory Data Analysis (EDA) notebooks
- [ ] Statistical hypothesis testing results
- [ ] Machine learning models
- [ ] Final analysis report with recommendations

## Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning
- **Matplotlib/Seaborn/Plotly**: Data visualization
- **SciPy**: Statistical testing
- **DVC**: Data version control
- **Jupyter**: Interactive analysis

## Contributing

This is a project for AlphaCare Insurance Solutions. For questions or contributions, please contact the data analytics team.

## Key Dates

- Challenge Introduction: 10:30 AM UTC, Wednesday, 03 Dec 2025
- Interim Submission: 8:00 PM UTC, Sunday, 07 Dec 2025
- Final Submission: 8:00 PM UTC, Tuesday, 09 Dec 2025

## License

Proprietary - AlphaCare Insurance Solutions

## Team

- Facilitator: Kerod, Mahbubah, Filimon

---

**Note**: This project follows best practices in data engineering, predictive analytics, and machine learning engineering, with a focus on reproducibility, modularity, and maintainability.

