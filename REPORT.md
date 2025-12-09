# End-to-End Insurance Risk Analytics & Predictive Modeling
## Project Report

**AlphaCare Insurance Solutions (ACIS)**  
**Date**: December 2025  
**Data Period**: February 2014 - August 2015  
**Status**: Task 1, 2 & 3 Completed | Task 4 In Progress

---

## Business Context

AlphaCare Insurance Solutions is committed to developing cutting-edge risk and predictive analytics for car insurance planning and marketing in South Africa. This project analyzes historical insurance claim data to optimize marketing strategy and discover "low-risk" targets for premium reduction, creating opportunities to attract new clients while maintaining profitability.

**Business Challenge**: Current portfolio shows **104.77% loss ratio**, indicating unprofitability. Urgent need to identify low-risk segments for premium optimization and targeted marketing.

**Business Impact**:
- **Attract New Clients**: Reduced premiums for low-risk segments increase market competitiveness
- **Risk Management**: Better understanding of risk factors enables informed pricing decisions
- **Profitability**: Optimize margins while expanding customer base
- **Data-Driven Decisions**: Replace intuition with statistical evidence

---

## Project Objective

**Primary Objective**: Analyze historical insurance claim data (February 2014 - August 2015) to identify low-risk customer segments and develop predictive models that enable AlphaCare Insurance Solutions to optimize premium pricing, reduce risk exposure, and attract new clients through targeted marketing strategies.

**Specific Objectives**:
1. **Risk Segmentation**: Identify low-risk segments (by province, zipcode, vehicle type, gender) for premium reduction opportunities
2. **Premium Optimization**: Develop machine learning models to predict optimal premium values
3. **Marketing Strategy**: Enable data-driven marketing to low-risk segments with competitive pricing
4. **Profitability Analysis**: Understand risk and margin differences through statistical hypothesis testing
5. **Predictive Modeling**: Build models predicting total claims by zipcode and optimal premium values

**Success Criteria**:
- Identify at least 3 low-risk segments with loss ratio < 40%
- Develop predictive models with R² > 0.7 for premium prediction
- Complete hypothesis testing for all specified dimensions
- Generate comprehensive visualizations supporting business decisions

---

## Key Statistics & Data Quality

### Descriptive Statistics

**TotalPremium Statistics** (n=1,000,098):
| Statistic | Value |
|----------|-------|
| Mean | 61.91 |
| Median | 2.18 |
| Standard Deviation | 230.28 |
| Minimum | -782.58 |
| Maximum | 65,282.60 |
| Skewness | 138.60 (highly right-skewed) |

**TotalClaims Statistics** (n=1,000,098):
| Statistic | Value |
|----------|-------|
| Mean | 64.86 |
| Median | 0.00 |
| Standard Deviation | 2,384.08 |
| Minimum | -12,002.41 |
| Maximum | 393,092.10 |
| Skewness | 69.93 (highly right-skewed) |

### Data Quality Assessment

| Variable | Missing Values | Missing % | Status |
|----------|---------------|-----------|--------|
| TotalPremium | 0 | 0.00% | ✅ Complete |
| TotalClaims | 0 | 0.00% | ✅ Complete |
| SumInsured | 0 | 0.00% | ✅ Complete |
| CustomValueEstimate | 0 | 0.00% | ✅ Complete |

### Outlier Detection (IQR Method)

| Variable | Outliers | Percentage | Treatment |
|----------|----------|------------|-----------|
| TotalPremium | 209,042 | 20.90% | Retained for analysis, capped at 99th percentile for modeling |
| TotalClaims | 2,793 | 0.28% | Retained for analysis |
| CustomValueEstimate | 217,880 | 21.79% | Retained for analysis |

### Overall Loss Ratio

**Portfolio Loss Ratio**: **104.77%**

**Interpretation**: 
- Loss ratio > 100% indicates portfolio unprofitability
- Premiums collected are less than claims paid
- Urgent need for premium optimization and risk segmentation

---

## Implementation Status Tables

### Task 1: Git and GitHub Setup ✅ COMPLETED

| Priority | Component | Description | Status |
|----------|-----------|-------------|--------|
| **P0 (Critical)** | Repository Initialization | Initialize git repository with proper structure | ✅ Completed |
| **P0 (Critical)** | .gitignore Configuration | Exclude venv, cache, data files, and sensitive information | ✅ Completed |
| **P1 (High)** | Branch Strategy | Create main, task-1, task-2 branches | ✅ Completed |
| **P1 (High)** | README.md | Comprehensive project documentation | ✅ Completed |
| **P1 (High)** | Project Structure | Modular OOP architecture | ✅ Completed |
| **P2 (Medium)** | CI/CD Workflow | Automated linting and testing pipeline | ✅ Completed |
| **P2 (Medium)** | PR Template | Standardized pull request template | ✅ Completed |

### Task 2: Data Version Control (DVC) ✅ COMPLETED

| Component | Description | Status |
|-----------|-------------|--------|
| DVC Initialization | DVC repository initialized | ✅ Completed |
| Remote Storage | Local storage configured (`data_storage/`) | ✅ Completed |
| Data Tracking | Data file tracked with DVC (`MachineLearningRating_v3.txt`) | ✅ Completed |
| Git Integration | DVC artifacts committed with proper .gitignore | ✅ Completed |

### Task 3: EDA & Statistics ✅ COMPLETED

| Task | Status |
|------|--------|
| Comprehensive EDA on all dimensions | ✅ Completed |
| Loss ratio calculations by dimensions | ✅ Completed |
| Temporal trend analysis | ✅ Completed |
| Vehicle make/model analysis | ✅ Completed |
| 5 key visualizations generated | ✅ Completed |
| Statistical hypothesis testing | ✅ Completed |
| Hypothesis test visualizations | ✅ Completed |
| Business recommendations | ✅ Completed |

### Task 4: Machine Learning & Predictive Modeling 📋 PLANNED

| Task | Status |
|------|--------|
| Linear regression models by zipcode | ⏳ Pending |
| Premium prediction ML models | ⏳ Pending |
| Feature importance analysis | ⏳ Pending |
| Model evaluation and validation | ⏳ Pending |

---

## Generated Visualizations

### 1. Portfolio Overview Dashboard
**File**: `reports/visualizations/01_portfolio_overview.png`

**Components**: Overall Loss Ratio KPI (104.77%), Monthly Premium vs Claims trends, Claim frequency distribution, Premium distribution box plot

**Key Insight**: Portfolio shows unprofitability with loss ratio > 100%. Highly right-skewed distributions indicate need for robust statistical methods.

---

### 2. Loss Ratio by Province
**File**: `reports/visualizations/03_loss_ratio_province.png`

**Visualization**: Horizontal bar chart with color-coded risk levels (Red: >60%, Orange: 40-60%, Green: <40%)

**Key Insight**: Significant risk variation across provinces. Low-risk provinces (<40%) identified for premium reduction and market expansion.

**Business Action**: Consider premium reductions in green provinces, premium increases in red provinces.

---

### 3. Loss Ratio by Vehicle Type
**File**: `reports/visualizations/04_loss_ratio_vehicle_type.png`

**Components**: Top 10 vehicle types by loss ratio, Premium vs Claims scatter plot

**Key Insight**: Clear differentiation in risk profiles across vehicle types. Pricing efficiency opportunities identified.

**Business Action**: Adjust premiums for high-risk vehicle types, develop specialized products for low-risk segments.

---

### 4. Temporal Trends Analysis
**File**: `reports/visualizations/07_temporal_trends.png`

**Components**: Monthly Premium and Claims trends (dual-axis), Monthly Loss Ratio trend with risk thresholds

**Key Insight**: Seasonal patterns identified in claims and premiums over 18-month period. Risk threshold visualization highlights critical periods.

**Business Action**: Implement seasonal pricing adjustments, enhance risk management during high-risk periods.

---

### 5. Vehicle Make Analysis
**File**: `reports/visualizations/09_vehicle_make_analysis.png`

**Components**: Top 10 makes by total claims, Top 10 makes by loss ratio (highest risk)

**Key Insight**: Significant variation in risk by vehicle manufacturer. Top risk makes identified for premium adjustment.

**Business Action**: Develop partnerships with low-risk manufacturers, adjust premiums based on make risk profiles.

---

## Task 3: Statistical Hypothesis Testing Results

### Methodology

**Metrics Used**:
- **Claim Frequency**: Proportion of policies with at least one claim (binary: 1 if TotalClaims > 0, 0 otherwise)
- **Claim Severity**: Average claim amount given a claim occurred (mean of TotalClaims where TotalClaims > 0)
- **Margin**: TotalPremium - TotalClaims

**Statistical Tests**:
- **Claim Frequency**: Chi-square test for categorical data (proportion with claims vs. no claims)
- **Claim Severity**: Kruskal-Wallis test (non-parametric) for multiple groups, Mann-Whitney U for two groups
- **Margin**: Kruskal-Wallis test for multiple groups

**Significance Level**: α = 0.05

---

### Hypothesis Test 1: Risk Differences Across Provinces

**Null Hypothesis (H₀)**: There are no risk differences across provinces

**Results**:
- **Claim Frequency Test**: Chi-square, p < 0.0001 ***
- **Claim Severity Test**: Kruskal-Wallis, p < 0.0001 ***
- **Conclusion**: **REJECT H₀** - Risk differences exist across provinces

**Key Findings**:

| Province | Claim Frequency | Claim Severity | Policies | Claims |
|----------|----------------|----------------|----------|--------|
| Gauteng | 0.34% | R 22,243.88 | 393,865 | 1,322 |
| KwaZulu-Natal | 0.28% | R 29,609.49 | 169,781 | 483 |
| Western Cape | 0.22% | R 28,095.85 | 170,796 | 370 |
| Northern Cape | 0.13% | R 11,186.31 | 6,380 | 8 |

**Business Interpretation**:
- **Gauteng** exhibits the highest claim frequency (0.34%), 2.6x higher than Northern Cape (0.13%)
- **KwaZulu-Natal** shows the highest claim severity (R 29,609.49 average per claim)
- **Northern Cape** has the lowest risk profile (0.13% frequency, R 11,186.31 severity)

**Business Recommendation**: 
> We reject the null hypothesis for provinces (p < 0.0001). Specifically, Gauteng exhibits a 2.6x higher claim frequency than Northern Cape, suggesting regional risk adjustment to our premiums may be warranted. Consider implementing province-based pricing tiers with premium increases for high-risk provinces (Gauteng, KwaZulu-Natal) and premium reductions for low-risk provinces (Northern Cape, Free State).

---

### Hypothesis Test 2: Risk Differences Between Zipcodes

**Null Hypothesis (H₀)**: There are no risk differences between zipcodes

**Results**:
- **Claim Frequency Test**: Chi-square, p < 0.0001 ***
- **Claim Severity Test**: Kruskal-Wallis, p < 0.0001 ***
- **Conclusion**: **REJECT H₀** - Risk differences exist between zipcodes
- **Zipcodes Tested**: Top 20 zipcodes by policy count

**Key Findings**:

**Top 5 Highest Risk Zipcodes** (by claim frequency):
| Zipcode | Claim Frequency | Claim Severity | Policies |
|---------|----------------|----------------|----------|
| 1863 | 0.51% | R 30,915.85 | 8,655 |
| 400 | 0.51% | R 7,133.12 | 6,692 |
| 8000 | 0.43% | R 33,685.33 | 11,794 |

**Top 5 Lowest Risk Zipcodes** (by claim frequency):
| Zipcode | Claim Frequency | Claim Severity | Policies |
|---------|----------------|----------------|----------|
| 7405 | 0.16% | R 21,002.02 | 18,518 |
| 7784 | 0.17% | R 35,156.65 | 28,585 |
| 7750 | 0.18% | R 21,929.24 | 9,408 |

**Business Interpretation**:
- Zipcode **1863** has 3.2x higher claim frequency (0.51%) than zipcode **7405** (0.16%)
- Significant variation in both frequency and severity across zipcodes

**Business Recommendation**:
> We reject the null hypothesis for zipcodes (p < 0.0001). Risk differences exist between zipcodes, with the highest-risk zipcode (1863) showing 3.2x higher claim frequency than the lowest-risk zipcode (7405). Recommendation: Implement zipcode-based pricing tiers to reflect risk differences and optimize premium pricing.

---

### Hypothesis Test 3: Margin Differences Between Zipcodes

**Null Hypothesis (H₀)**: There is no significant margin (profit) difference between zip codes

**Results**:
- **Test Method**: Kruskal-Wallis
- **P-value**: p < 0.0001 ***
- **Conclusion**: **REJECT H₀** - Margin differences exist between zipcodes

**Key Findings**:

**Top 5 Most Profitable Zipcodes**:
| Zipcode | Mean Margin | Std Dev | Policies |
|---------|-------------|---------|----------|
| 400 | R 38.81 | R 1,182.56 | 6,692 |
| 152 | R 27.91 | R 1,551.27 | 9,423 |
| 299 | R 19.56 | R 1,289.07 | 25,546 |

**Top 5 Least Profitable Zipcodes**:
| Zipcode | Mean Margin | Std Dev | Policies |
|---------|-------------|---------|----------|
| 1863 | -R 100.57 | R 3,974.94 | 8,655 |
| 4001 | -R 57.58 | R 3,528.40 | 6,647 |
| 302 | -R 56.35 | R 3,843.54 | 9,531 |

**Business Interpretation**:
- Zipcode **400** shows positive margin of R 38.81 per policy
- Zipcode **1863** shows negative margin of -R 100.57 per policy (loss-making)
- Margin difference of R 139.38 between most and least profitable zipcodes

**Business Recommendation**:
> We reject the null hypothesis for zipcode margins (p < 0.0001). Significant margin differences exist between zipcodes, with zipcode 400 showing R 38.81 profit per policy while zipcode 1863 shows -R 100.57 loss per policy. Recommendation: Adjust pricing strategy to optimize margins by zipcode, with premium increases for unprofitable zipcodes and competitive pricing for profitable segments.

---

### Hypothesis Test 4: Risk Differences Between Women and Men

**Null Hypothesis (H₀)**: There is no significant risk difference between Women and Men

**Results**:
- **Claim Frequency Test**: Chi-square, p = 0.9515
- **Claim Severity Test**: Mann-Whitney U, p = 0.2235
- **Conclusion**: **FAIL TO REJECT H₀** - No significant risk differences between genders

**Key Findings**:

| Gender | Claim Frequency | Claim Severity | Sample Size |
|--------|----------------|----------------|-------------|
| Female | 0.21% | R 17,874.72 | 6,755 |
| Male | 0.22% | R 14,858.55 | 42,817 |

**Business Interpretation**:
- Claim frequency difference: 0.01% (statistically insignificant)
- Claim severity difference: R 3,016.17 (statistically insignificant)
- Both p-values > 0.05, indicating no significant difference

**Business Recommendation**:
> We fail to reject the null hypothesis for gender (p = 0.9515 for frequency, p = 0.2235 for severity). No significant risk differences between genders detected. Recommendation: Gender-neutral pricing may be appropriate, which also aligns with regulatory requirements in many jurisdictions.

---

### Hypothesis Testing Summary

| Test | Null Hypothesis | Test Method | P-Value | Reject H₀? | Conclusion |
|------|----------------|-------------|---------|------------|------------|
| Province Risk | No risk differences across provinces | Chi-square (Freq) & Kruskal-Wallis (Sev) | < 0.0001 | ✅ Yes | Risk differences exist |
| Zipcode Risk | No risk differences between zipcodes | Chi-square (Freq) & Kruskal-Wallis (Sev) | < 0.0001 | ✅ Yes | Risk differences exist |
| Zipcode Margin | No margin differences between zipcodes | Kruskal-Wallis | < 0.0001 | ✅ Yes | Margin differences exist |
| Gender Risk | No risk differences between genders | Chi-square (Freq) & Mann-Whitney U (Sev) | 0.9515 / 0.2235 | ❌ No | No significant differences |

**Key Insights**:
1. **Geographic Risk Segmentation**: Both provinces and zipcodes show significant risk variations, supporting location-based pricing strategies
2. **Profitability by Location**: Zipcode-level margin analysis reveals opportunities for premium optimization
3. **Gender Neutrality**: No statistical evidence for gender-based risk differences, supporting gender-neutral pricing

**Strategic Recommendations**:
1. **Implement Geographic Pricing Tiers**: Create province and zipcode-based premium adjustments
2. **Optimize Unprofitable Segments**: Increase premiums for high-risk, low-margin zipcodes (e.g., 1863, 4001)
3. **Expand Low-Risk Markets**: Target low-risk provinces and zipcodes with competitive pricing to attract new customers
4. **Maintain Gender-Neutral Pricing**: Continue gender-neutral approach as no statistical difference detected

---

## Summary

### Key Findings

1. **Critical Portfolio Issue**: Overall loss ratio of 104.77% indicates urgent need for premium optimization
2. **Risk Segmentation**: Significant variations identified across provinces and vehicle types
3. **Data Quality**: 0% missing values after preprocessing, outliers documented (20.90% in TotalPremium)
4. **Distribution Characteristics**: Highly right-skewed distributions require robust statistical methods
5. **Temporal Patterns**: Seasonal trends identified requiring dynamic pricing strategies

### Completed Work

- ✅ **Task 1**: Git repository, OOP architecture, CI/CD pipeline established
- ✅ **Task 2**: DVC initialized, data versioning workflow established
- ✅ **Task 3**: EDA completed, hypothesis testing completed, 8 visualizations generated
- ✅ **Data Quality**: Comprehensive assessment completed with 0% missing values
- ✅ **Outlier Analysis**: IQR method applied, treatment strategy documented
- ✅ **Hypothesis Testing**: All 4 hypothesis tests completed with statistical validation
- ✅ **Business Recommendations**: Actionable insights provided for each test result

### Current Status

- **Data Records Analyzed**: 1,000,098 transactions
- **Visualizations Generated**: 8 visualizations (hypothesis testing + EDA)
- **Overall Loss Ratio**: 104.77% (critical finding)
- **Low-Risk Segments**: Identified through province, zipcode, and vehicle analysis
- **Hypothesis Tests Completed**: 4/4 tests with statistical validation
- **Key Finding**: Geographic risk segmentation validated; gender-neutral pricing supported

---

## Next Focus

### Immediate Priority: Task 4 - Machine Learning & Predictive Modeling

1. **Linear Regression by Zipcode**: Fit models for each zipcode to predict claims
2. **Premium Prediction Model**: Feature engineering and ML model training
3. **Model Evaluation**: Analyze feature importance, validate model performance
4. **Model Deployment**: Create performance visualizations, document interpretability

### Next Phase: Task 4 - Machine Learning & Predictive Modeling

1. **Linear Regression by Zipcode**: Fit models for each zipcode, evaluate performance
2. **Premium Prediction Model**: Feature engineering, train Random Forest/Gradient Boosting models
3. **Model Evaluation**: Analyze feature importance, validate model performance
4. **Model Deployment**: Create performance visualizations, document interpretability

---

**Project Status**: Task 1, 2 & 3 Completed | Task 4 In Progress  
**Last Updated**: December 2025  
**Team**: AlphaCare Insurance Solutions Data Analytics Team
