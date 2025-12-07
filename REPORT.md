# End-to-End Insurance Risk Analytics & Predictive Modeling
## Project Report

**AlphaCare Insurance Solutions (ACIS)**  
**Date**: December 2025  
**Data Period**: February 2014 - August 2015  
**Status**: Task 1 & 2 Completed | Task 3 In Progress

---

## Executive Summary

This project develops a comprehensive risk analytics and predictive modeling platform for car insurance in South Africa. Through analysis of ~1 million historical claim records, we identify low-risk segments, optimize premium pricing, and enable data-driven marketing strategies to attract new clients while maintaining profitability.

**Key Findings**:
- **Critical Finding**: Overall portfolio loss ratio of **104.77%** indicates unprofitability requiring immediate action
- Descriptive statistics calculated for TotalPremium (mean=61.91, median=2.18) and TotalClaims (mean=64.86, median=0.00)
- Data quality assessment: 0% missing values after preprocessing for key financial variables
- Outlier detection: 20.90% outliers in TotalPremium, 0.28% in TotalClaims using IQR method
- Significant risk variations identified across provinces and vehicle types
- Temporal trends reveal patterns in claim frequency and severity over 18-month period
- Vehicle make analysis highlights high and low-risk segments
- Foundation established for predictive modeling and premium optimization

---

## Project Objective

**Primary Objective**: Analyze historical insurance claim data (February 2014 - August 2015) to identify low-risk customer segments and develop predictive models that enable AlphaCare Insurance Solutions to optimize premium pricing, reduce risk exposure, and attract new clients through targeted marketing strategies.

**Specific Objectives**:
1. **Risk Segmentation**: Identify low-risk customer segments (by province, zipcode, vehicle type, gender) for premium reduction opportunities
2. **Premium Optimization**: Develop machine learning models to predict optimal premium values based on car features, owner characteristics, and location
3. **Marketing Strategy**: Enable data-driven marketing to low-risk segments with competitive premium pricing
4. **Profitability Analysis**: Understand risk and margin differences across geographic and demographic dimensions through statistical hypothesis testing
5. **Predictive Modeling**: Build models that predict total claims by zipcode and optimal premium values for new policies

**Success Criteria**:
- Identify at least 3 low-risk segments with loss ratio < 40%
- Develop predictive models with R² > 0.7 for premium prediction
- Complete hypothesis testing for all specified dimensions (provinces, zipcodes, gender)
- Generate comprehensive visualizations supporting business decisions

---

## 1. Project Goals

### Primary Objectives

1. **Risk Segmentation**: Identify low-risk customer segments for premium reduction opportunities
2. **Premium Optimization**: Develop predictive models to determine optimal premium values
3. **Marketing Strategy**: Enable targeted marketing to low-risk segments
4. **Profitability Analysis**: Understand risk differences across geographic and demographic dimensions
5. **Predictive Modeling**: Build machine learning models for claims prediction and premium optimization

### Business Impact

- **Attract New Clients**: Reduced premiums for low-risk segments increase market competitiveness
- **Risk Management**: Better understanding of risk factors enables informed pricing decisions
- **Profitability**: Optimize margins while expanding customer base
- **Data-Driven Decisions**: Replace intuition with statistical evidence

---

## 2. Methodology

### 2.1 Data Overview

**Data Source**: Historical insurance claim data  
**Format**: Pipe-delimited text file (`MachineLearningRating_v3.txt`)  
**Records**: 1,000,098 transactions  
**Period**: February 2014 - August 2015 (18 months)  
**Key Variables**:
- Policy information (PolicyID, TransactionMonth)
- Client demographics (Gender, MaritalStatus, Province, PostalCode)
- Vehicle characteristics (Make, Model, VehicleType, RegistrationYear)
- Financial metrics (TotalPremium, TotalClaims, SumInsured)
- Coverage details (CoverType, CoverCategory)

### 2.2 Data Quality Assessment

**Missing Values Analysis**:
- **TotalPremium**: 0 missing values (0.00%) - Complete after preprocessing
- **TotalClaims**: 0 missing values (0.00%) - Complete after preprocessing
- **SumInsured**: 0 missing values (0.00%) - Complete after preprocessing
- **CustomValueEstimate**: 0 missing values (0.00%) - Complete after preprocessing

**Data Quality Actions Taken**:
- Missing numeric values filled with median for continuous variables
- Missing categorical values filled with "Unknown"
- Date columns converted to datetime format
- Invalid values handled through preprocessing pipeline

### 2.3 Descriptive Statistics

**TotalPremium Statistics** (n=1,000,098):
| Statistic | Value |
|----------|-------|
| Mean | 61.91 |
| Median | 2.18 |
| Standard Deviation | 230.28 |
| Minimum | -782.58 |
| Maximum | 65,282.60 |
| 25th Percentile | 0.00 |
| 75th Percentile | 21.93 |
| Skewness | 138.60 (highly right-skewed) |
| Kurtosis | 37,176.19 (heavy-tailed distribution) |

**TotalClaims Statistics** (n=1,000,098):
| Statistic | Value |
|----------|-------|
| Mean | 64.86 |
| Median | 0.00 |
| Standard Deviation | 2,384.08 |
| Minimum | -12,002.41 |
| Maximum | 393,092.10 |
| 25th Percentile | 0.00 |
| 75th Percentile | 0.00 |
| Skewness | 69.93 (highly right-skewed) |
| Kurtosis | 6,791.93 (heavy-tailed distribution) |

**Key Observations**:
- Both variables are highly right-skewed with heavy tails
- Median values much lower than means, indicating many zero/low values
- Large standard deviations relative to means suggest high variability
- Negative values present (likely data entry errors or adjustments)

### 2.4 Outlier Detection & Treatment

**Outlier Detection Method**: Interquartile Range (IQR) Method
- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR
- Outliers identified as values outside these bounds

**Outlier Statistics**:
- **TotalPremium**: 209,042 outliers (20.90% of records)
- **TotalClaims**: 2,793 outliers (0.28% of records)
- **CustomValueEstimate**: 217,880 outliers (21.79% of records)

**Outlier Treatment Strategy**:
1. **For Analysis**: Outliers retained for comprehensive risk assessment
2. **For Modeling**: Outliers will be capped at 99th percentile to prevent model skew
3. **Documentation**: All outliers documented for business review
4. **Investigation**: Negative values flagged for data quality review

**Rationale**: Insurance data naturally contains extreme values (large claims, high-value vehicles). Retaining outliers provides complete risk picture while capping prevents model instability.

### 2.5 Overall Loss Ratio

**Portfolio Loss Ratio**: **104.77%**

**Calculation**: TotalClaims / TotalPremium = 64,860,000 / 61,905,000 = 1.0477

**Interpretation**: 
- Loss ratio > 100% indicates portfolio is unprofitable overall
- Premiums collected are less than claims paid
- Urgent need for premium optimization and risk segmentation
- Identifies critical business need for this analysis

### 2.6 Analysis Framework

#### Phase 1: Exploratory Data Analysis (EDA) ✅ Completed
- ✅ Descriptive statistics calculated (TotalPremium, TotalClaims, SumInsured, CustomValueEstimate)
- ✅ Data quality assessment completed (missing values: 0% after preprocessing)
- ✅ Outlier detection completed (IQR method, 20.90% for TotalPremium, 0.28% for TotalClaims)
- ✅ Loss ratio analysis by key dimensions (Province, VehicleType, Gender)
- ✅ Temporal trend analysis (18-month period)
- ✅ Vehicle make/model risk profiling

#### Phase 2: Statistical Hypothesis Testing (In Progress)
- Test risk differences across provinces
- Test risk differences between zipcodes
- Test margin (profit) differences between zipcodes
- Test risk differences between genders

#### Phase 3: Predictive Modeling (Planned)
- Linear regression models by zipcode for claims prediction
- Machine learning models for premium optimization
- Feature importance analysis
- Model evaluation and validation

### 2.3 Technical Implementation

- **Architecture**: Object-Oriented Programming (OOP) with modular design
- **Data Processing**: Pandas, NumPy
- **Statistical Analysis**: SciPy, Statsmodels
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Version Control**: DVC
- **Development**: Python 3.8+, Jupyter Notebooks

---

## 3. Generated Visualizations

### 3.1 Portfolio Overview Dashboard
**File**: `reports/visualizations/01_portfolio_overview.png`

**Components**:
- **Overall Loss Ratio KPI**: Shows portfolio-wide loss ratio of **104.77%** (critical unprofitability indicator)
- **Premium vs Claims Over Time**: Dual-axis line chart showing monthly trends over 18 months
- **Claim Frequency Distribution**: Histogram of claim amounts (log scale) showing highly right-skewed distribution
- **Premium Distribution**: Box plot showing premium distribution (log scale) with median=2.18, mean=61.91

**Insights**:
- **Critical Finding**: Loss ratio > 100% indicates urgent need for premium optimization
- Provides high-level view of portfolio health showing unprofitability
- Identifies temporal patterns in premium and claims
- Highlights distribution characteristics: highly right-skewed with heavy tails
- Median values much lower than means indicate many low/zero value transactions

---

### 3.2 Loss Ratio by Province
**File**: `reports/visualizations/03_loss_ratio_province.png`

**Visualization**: Horizontal bar chart with color coding:
- **Red**: High risk (Loss Ratio > 60%)
- **Orange**: Medium risk (Loss Ratio 40-60%)
- **Green**: Low risk (Loss Ratio < 40%)

**Insights**:
- Identifies provinces with highest and lowest risk profiles
- Enables targeted marketing to low-risk provinces
- Supports premium adjustment strategies by geography
- Risk threshold lines (40% and 60%) provide clear segmentation

**Business Action**: Consider premium reductions in green provinces, premium increases in red provinces

---

### 3.3 Loss Ratio by Vehicle Type
**File**: `reports/visualizations/04_loss_ratio_vehicle_type.png`

**Components**:
- **Top 10 Vehicle Types by Loss Ratio**: Horizontal bar chart showing highest risk vehicle types
- **Premium vs Claims Scatter**: Relationship between average premium and claims by vehicle type

**Insights**:
- Identifies vehicle types with highest risk profiles
- Reveals pricing efficiency (premium vs actual claims)
- Supports vehicle-specific premium adjustments
- Highlights opportunities for product segmentation

**Business Action**: Adjust premiums for high-risk vehicle types, develop specialized products for low-risk segments

---

### 3.4 Temporal Trends Analysis
**File**: `reports/visualizations/07_temporal_trends.png`

**Components**:
- **Monthly Premium and Claims Trends**: Dual-axis line chart showing 18-month patterns
- **Monthly Loss Ratio Trend**: Line chart with risk threshold markers and filled area

**Insights**:
- Reveals seasonal patterns in claims and premiums
- Identifies periods of increased risk
- Shows portfolio performance trends over time
- Risk threshold visualization (40% and 60%) highlights critical periods

**Business Action**: Implement seasonal pricing adjustments, enhance risk management during high-risk periods

---

### 3.5 Vehicle Make Analysis
**File**: `reports/visualizations/09_vehicle_make_analysis.png`

**Components**:
- **Top 10 Vehicle Makes by Total Claims**: Bar chart showing makes with highest claim volumes
- **Top 10 Vehicle Makes by Loss Ratio**: Bar chart showing makes with highest risk (worst performers)

**Insights**:
- Identifies vehicle makes with highest claim volumes
- Highlights makes with worst loss ratios (highest risk)
- Enables make-specific risk assessment
- Supports targeted marketing and pricing strategies

**Business Action**: 
- Develop partnerships with low-risk vehicle manufacturers
- Adjust premiums based on vehicle make risk profiles
- Create make-specific insurance products

---

## 4. Key Findings

### 4.1 Portfolio-Level Insights

1. **Overall Loss Ratio**: Calculated and monitored as primary KPI
2. **Temporal Patterns**: Identified seasonal and monthly trends in claims and premiums
3. **Distribution Characteristics**: Understood claim frequency and premium distributions

### 4.2 Geographic Insights

1. **Province Risk Variation**: Significant differences in loss ratios across provinces
2. **Low-Risk Provinces**: Identified opportunities for premium reduction and market expansion
3. **High-Risk Provinces**: Require premium adjustments and enhanced risk management

### 4.3 Vehicle Insights

1. **Vehicle Type Risk**: Clear differentiation in risk profiles across vehicle types
2. **Make-Specific Risk**: Significant variation in risk by vehicle manufacturer
3. **Pricing Efficiency**: Opportunities to optimize premium-to-risk ratios

### 4.4 Temporal Insights

1. **Seasonal Patterns**: Identified periods of increased risk
2. **Trend Analysis**: Portfolio performance trends over 18-month period
3. **Risk Threshold Monitoring**: Visual tracking of portfolio health against benchmarks

---

## 5. Implementation Status

### Task 1: Git and GitHub Setup ✅ COMPLETED
- [x] Git repository initialized with proper version control
- [x] Comprehensive README with project overview
- [x] Modular OOP code structure implemented
- [x] GitHub Actions CI/CD workflow configured
- [x] Task-specific branches created (main, task-1, task-2)
- [x] Pull request workflow established

#### Task 1 Priority Tables

**Table 1: Git Repository Setup Priorities**

| Priority | Component | Description | Status |
|----------|-----------|-------------|--------|
| **P0 (Critical)** | Repository Initialization | Initialize git repository with proper structure | ✅ Completed |
| **P0 (Critical)** | .gitignore Configuration | Exclude venv, cache, data files, and sensitive information | ✅ Completed |
| **P1 (High)** | Branch Strategy | Create main, task-1, task-2 branches | ✅ Completed |
| **P1 (High)** | Initial Commit | Commit project structure and core files | ✅ Completed |
| **P2 (Medium)** | Commit History | Establish meaningful commit message conventions | ✅ Completed |
| **P2 (Medium)** | Remote Repository | Connect to GitHub remote repository | ✅ Completed |

**Table 2: Documentation & Code Quality Priorities**

| Priority | Component | Description | Status |
|----------|-----------|-------------|--------|
| **P0 (Critical)** | README.md | Comprehensive project documentation with setup instructions | ✅ Completed |
| **P0 (Critical)** | Project Structure | Modular OOP architecture with clear directory structure | ✅ Completed |
| **P1 (High)** | Code Organization | Separate modules for data, EDA, statistics, models, utils | ✅ Completed |
| **P1 (High)** | Requirements.txt | Complete dependency list with versions | ✅ Completed |
| **P2 (Medium)** | Setup.py | Package installation configuration | ✅ Completed |
| **P2 (Medium)** | Example Usage | Demonstration script showing OOP usage | ✅ Completed |

**Table 3: CI/CD & Workflow Priorities**

| Priority | Component | Description | Status |
|----------|-----------|-------------|--------|
| **P0 (Critical)** | GitHub Actions Workflow | Automated linting and testing pipeline | ✅ Completed |
| **P1 (High)** | Pull Request Template | Standardized PR template for code review | ✅ Completed |
| **P1 (High)** | Branch Protection | Establish branch workflow (main, task branches) | ✅ Completed |
| **P2 (Medium)** | Code Linting | Flake8 and Black formatting checks | ✅ Configured |
| **P2 (Medium)** | Automated Testing | Pytest integration for unit tests | ✅ Configured |
| **P3 (Low)** | Code Coverage | Coverage reporting in CI pipeline | ⏳ Planned |

### Task 2: Data Version Control (DVC) ✅ COMPLETED
- [x] DVC initialized in repository
- [x] Local remote storage configured
- [x] Data file tracked with DVC (`MachineLearningRating_v3.txt`)
- [x] DVC artifacts committed with proper .gitignore rules
- [x] Data versioning workflow established

### Task 3: Exploratory Data Analysis & Statistics 🔄 NEXT
- [ ] Complete comprehensive EDA on all key dimensions
- [ ] Statistical hypothesis testing:
  - [ ] Risk differences across provinces
  - [ ] Risk differences between zipcodes
  - [ ] Margin (profit) differences between zipcodes
  - [ ] Risk differences between genders
- [ ] Generate remaining visualizations (12 additional)
- [ ] Document statistical findings and interpretations

### Task 4: Machine Learning & Predictive Modeling 📋 PLANNED
- [ ] Linear regression models by zipcode for total claims prediction
- [ ] Premium prediction machine learning model:
  - [ ] Feature engineering (car, owner, location, plan features)
  - [ ] Model training (Random Forest, Gradient Boosting)
  - [ ] Feature importance analysis
  - [ ] Model evaluation and validation
- [ ] Model performance visualizations
- [ ] Model interpretation and explainability

---

## 6. Visualization Path (Remaining)

### Priority Visualizations

1. **Loss Ratio by Gender** - Gender-based risk analysis
2. **Multi-Dimensional Loss Ratio** - Combined Province × VehicleType analysis
3. **Zipcode Risk Analysis** - Geographic risk clustering
4. **Margin Analysis** - Profitability by location
5. **Hypothesis Test Results** - Statistical test visualizations
6. **Model Performance** - Predictive model evaluation charts
7. **Feature Importance** - ML model driver analysis

### Implementation

All visualizations can be generated using the `scripts/generate_visualizations.py` script, which leverages the OOP classes:
- `DataLoader`: Data loading and preprocessing
- `EDAAnalyzer`: EDA and visualization generation
- `HypothesisTester`: Statistical testing (for future visualizations)

---

## 7. Technical Architecture

### Object-Oriented Design

```
src/alphacare/
├── data/              # DataLoader: Data loading and preprocessing
├── eda/               # EDAAnalyzer: Exploratory data analysis
├── statistics/        # HypothesisTester: Statistical testing
├── models/            # ModelTrainer, LinearRegressionModel: ML models
└── utils/             # Logging, DVC management utilities
```

### Key Features

- **Modular Design**: Each component is a separate, reusable class
- **Data Version Control**: DVC integration for reproducible pipelines
- **Comprehensive Logging**: Full logging throughout the pipeline
- **Error Handling**: Robust error handling and validation
- **Extensible**: Easy to add new analysis methods and visualizations

---

## 8. Next Steps

### Task 3: EDA & Statistical Analysis (Current Focus)
1. **Complete Hypothesis Testing**:
   - Test risk differences across provinces (ANOVA/Kruskal-Wallis)
   - Test risk differences between zipcodes
   - Test margin differences between zipcodes
   - Test risk differences between genders (Mann-Whitney U)
   - Document all test results with p-values and interpretations

2. **Generate Additional Visualizations**:
   - Loss ratio by gender
   - Multi-dimensional loss ratio (Province × VehicleType)
   - Zipcode risk analysis
   - Margin analysis by location
   - Hypothesis test results visualization

3. **Statistical Reporting**:
   - Summarize all hypothesis test results
   - Provide actionable insights from statistical analysis
   - Update report with findings

### Task 4: Machine Learning & Predictive Modeling (Next Phase)
1. **Linear Regression by Zipcode**:
   - Fit regression models for each zipcode
   - Evaluate model performance (R², RMSE, MAE)
   - Identify best and worst performing zipcode models

2. **Premium Prediction Model**:
   - Feature engineering from all available dimensions
   - Train Random Forest and Gradient Boosting models
   - Evaluate and compare model performance
   - Analyze feature importance

3. **Model Deployment**:
   - Create model performance visualizations
   - Document model interpretability
   - Prepare model for production use

### Future Tasks
- Build interactive executive dashboard
- Generate comprehensive recommendations
- Present findings to stakeholders
- Implement pricing strategy adjustments

---

## 9. Expected Deliverables

1. ✅ **Data Pipeline**: Automated data loading and preprocessing
2. ✅ **EDA Framework**: Comprehensive exploratory analysis classes
3. ✅ **Initial Visualizations**: 5 key visualizations generated
4. ⏳ **Statistical Analysis**: Hypothesis test results (in progress)
5. ⏳ **Predictive Models**: Trained models for claims and premium prediction
6. ⏳ **Complete Visualization Suite**: 17+ visualizations
7. ⏳ **Executive Dashboard**: Interactive dashboard for stakeholders
8. ⏳ **Final Report**: Complete analysis with recommendations

---

## 10. Key Metrics & KPIs

### Business Metrics
- **Overall Loss Ratio**: Calculated and monitored
- **Low-Risk Segment Size**: Identified through province and vehicle analysis
- **Risk Variation**: Quantified across geographic and vehicle dimensions
- **Temporal Trends**: Tracked over 18-month period

### Technical Metrics
- **Data Quality**: 1M+ records processed successfully
- **Visualization Coverage**: 5 of 17 planned visualizations completed
- **Code Quality**: OOP architecture with modular design
- **Reproducibility**: DVC integration for data version control

---

## 11. Risk Factors & Considerations

1. **Data Quality**: Missing values handled, outliers identified
2. **Temporal Changes**: Market conditions may have changed since 2015 data period
3. **Model Generalization**: Models will need validation on current data
4. **Regulatory Compliance**: Ensure pricing strategies comply with South African regulations
5. **Ethical Considerations**: Avoid discriminatory pricing practices

---

## 12. Conclusion

This project establishes a solid foundation for data-driven insurance risk analytics. Through comprehensive EDA and initial visualizations, we have:

- **Identified Key Risk Segments**: Provinces and vehicle types with varying risk profiles
- **Established Analysis Framework**: OOP architecture enabling scalable analysis
- **Generated Actionable Insights**: Visualizations supporting business decision-making
- **Created Reproducible Pipeline**: DVC integration ensures data version control

The visualization path provides clear communication of insights to both technical and business stakeholders. As we progress with hypothesis testing and predictive modeling, we will build upon these foundations to deliver comprehensive recommendations for premium optimization and marketing strategy.

---

## Appendix: Generated Visualizations

All visualizations are saved in `reports/visualizations/`:

1. `01_portfolio_overview.png` - **Portfolio Overview Dashboard**: Overall loss ratio KPI, monthly premium/claims trends, claim frequency distribution, and premium distribution box plot providing high-level portfolio health metrics.

2. `03_loss_ratio_province.png` - **Loss Ratio by Province**: Horizontal bar chart with color-coded risk levels (red: >60%, orange: 40-60%, green: <40%) identifying high and low-risk provinces for targeted marketing and premium adjustments.

3. `04_loss_ratio_vehicle_type.png` - **Loss Ratio by Vehicle Type**: Top 10 vehicle types by loss ratio with premium vs claims scatter plot, enabling vehicle-specific risk assessment and pricing optimization.

4. `07_temporal_trends.png` - **Temporal Trends Analysis**: Dual-axis monthly trends showing premium and claims over 18 months, plus loss ratio trend with risk threshold markers (40% and 60%) highlighting seasonal patterns and portfolio performance.

5. `09_vehicle_make_analysis.png` - **Vehicle Make Analysis**: Top 10 makes by total claims volume and top 10 by loss ratio (highest risk), supporting make-specific risk profiling and partnership opportunities.

**Generation Script**: `scripts/generate_visualizations.py`
---

**Project Status**: Task 1 & 2 Completed | Task 3 In Progress  
**Last Updated**: December 2025  
**Team**: AlphaCare Insurance Solutions Data Analytics Team  
**Data Records Analyzed**: 1,000,098

