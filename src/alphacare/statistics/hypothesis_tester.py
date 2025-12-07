"""
Hypothesis Testing Module

This module provides the HypothesisTester class for performing A/B tests
and statistical hypothesis testing on insurance data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HypothesisTester:
    """
    A class for performing statistical hypothesis tests on insurance data.
    
    This class provides methods for A/B testing, comparing groups,
    and testing various hypotheses about risk differences.
    
    Attributes:
        data (pd.DataFrame): The dataset to analyze
        results (Dict): Dictionary to store test results
        alpha (float): Significance level (default: 0.05)
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the HypothesisTester.
        
        Args:
            data: DataFrame containing the insurance data
            alpha: Significance level for tests (default: 0.05)
        """
        self.data = data.copy()
        self.results: Dict = {}
        self.alpha = alpha
        logger.info(f"HypothesisTester initialized with alpha={alpha}")
    
    def test_risk_difference_by_province(self) -> Dict:
        """
        Test the null hypothesis: There are no risk differences across provinces.
        
        Uses ANOVA or Kruskal-Wallis test depending on data distribution.
        
        Returns:
            Dict: Test results including p-value, statistic, and conclusion
        """
        if "Province" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("Province and TotalClaims columns are required")
        
        # Group data by province
        provinces = self.data["Province"].unique()
        province_groups = [self.data[self.data["Province"] == prov]["TotalClaims"].values 
                          for prov in provinces if len(self.data[self.data["Province"] == prov]) > 0]
        
        if len(province_groups) < 2:
            raise ValueError("Need at least 2 provinces for comparison")
        
        # Test for normality (Shapiro-Wilk test on sample)
        sample_sizes = [len(group) for group in province_groups]
        min_sample = min(sample_sizes)
        
        # Use parametric test if data appears normal, otherwise use non-parametric
        if min_sample > 3:
            # Test normality on a sample
            sample_data = np.concatenate([group[:min(1000, len(group))] for group in province_groups])
            _, p_norm = stats.shapiro(sample_data[:5000] if len(sample_data) > 5000 else sample_data)
            
            if p_norm > 0.05 and min_sample > 30:
                # Use ANOVA
                statistic, p_value = stats.f_oneway(*province_groups)
                test_name = "ANOVA"
            else:
                # Use Kruskal-Wallis (non-parametric)
                statistic, p_value = stats.kruskal(*province_groups)
                test_name = "Kruskal-Wallis"
        else:
            statistic, p_value = stats.kruskal(*province_groups)
            test_name = "Kruskal-Wallis"
        
        # Calculate means for each province
        province_means = self.data.groupby("Province")["TotalClaims"].agg(['mean', 'std', 'count']).to_dict('index')
        
        result = {
            'test_name': test_name,
            'null_hypothesis': 'There are no risk differences across provinces',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Reject H0: Risk differences exist across provinces' if p_value < self.alpha 
                         else 'Fail to reject H0: No significant risk differences across provinces',
            'province_means': province_means
        }
        
        self.results['province_risk_test'] = result
        logger.info(f"Province risk test: {result['conclusion']} (p={p_value:.4f})")
        return result
    
    def test_risk_difference_by_zipcode(self) -> Dict:
        """
        Test the null hypothesis: There are no risk differences between zipcodes.
        
        Uses ANOVA or Kruskal-Wallis test.
        
        Returns:
            Dict: Test results
        """
        if "PostalCode" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("PostalCode and TotalClaims columns are required")
        
        # Sample zipcodes if too many (for computational efficiency)
        zipcode_counts = self.data["PostalCode"].value_counts()
        top_zipcodes = zipcode_counts.head(20).index  # Top 20 zipcodes
        
        zipcode_groups = [
            self.data[self.data["PostalCode"] == zc]["TotalClaims"].values 
            for zc in top_zipcodes 
            if len(self.data[self.data["PostalCode"] == zc]) > 10
        ]
        
        if len(zipcode_groups) < 2:
            raise ValueError("Need at least 2 zipcodes with sufficient data")
        
        # Use Kruskal-Wallis (non-parametric, more robust)
        statistic, p_value = stats.kruskal(*zipcode_groups)
        
        zipcode_means = self.data[self.data["PostalCode"].isin(top_zipcodes)].groupby("PostalCode")["TotalClaims"].agg(['mean', 'std', 'count']).to_dict('index')
        
        result = {
            'test_name': 'Kruskal-Wallis',
            'null_hypothesis': 'There are no risk differences between zipcodes',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Reject H0: Risk differences exist between zipcodes' if p_value < self.alpha 
                         else 'Fail to reject H0: No significant risk differences between zipcodes',
            'zipcode_means': zipcode_means,
            'zipcodes_tested': len(zipcode_groups)
        }
        
        self.results['zipcode_risk_test'] = result
        logger.info(f"Zipcode risk test: {result['conclusion']} (p={p_value:.4f})")
        return result
    
    def test_margin_difference_by_zipcode(self) -> Dict:
        """
        Test the null hypothesis: There is no significant margin (profit) difference between zip codes.
        
        Margin = TotalPremium - TotalClaims
        
        Returns:
            Dict: Test results
        """
        if "PostalCode" not in self.data.columns:
            raise ValueError("PostalCode column is required")
        if "TotalPremium" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("TotalPremium and TotalClaims columns are required")
        
        # Calculate margin
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        # Sample zipcodes
        zipcode_counts = self.data["PostalCode"].value_counts()
        top_zipcodes = zipcode_counts.head(20).index
        
        zipcode_groups = [
            self.data[self.data["PostalCode"] == zc]["Margin"].values 
            for zc in top_zipcodes 
            if len(self.data[self.data["PostalCode"] == zc]) > 10
        ]
        
        if len(zipcode_groups) < 2:
            raise ValueError("Need at least 2 zipcodes with sufficient data")
        
        statistic, p_value = stats.kruskal(*zipcode_groups)
        
        zipcode_margins = self.data[self.data["PostalCode"].isin(top_zipcodes)].groupby("PostalCode")["Margin"].agg(['mean', 'std', 'count']).to_dict('index')
        
        result = {
            'test_name': 'Kruskal-Wallis',
            'null_hypothesis': 'There is no significant margin (profit) difference between zip codes',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Reject H0: Margin differences exist between zipcodes' if p_value < self.alpha 
                         else 'Fail to reject H0: No significant margin differences between zipcodes',
            'zipcode_margins': zipcode_margins,
            'zipcodes_tested': len(zipcode_groups)
        }
        
        self.results['zipcode_margin_test'] = result
        logger.info(f"Zipcode margin test: {result['conclusion']} (p={p_value:.4f})")
        return result
    
    def test_risk_difference_by_gender(self) -> Dict:
        """
        Test the null hypothesis: There is no significant risk difference between Women and men.
        
        Uses Mann-Whitney U test (non-parametric) or t-test.
        
        Returns:
            Dict: Test results
        """
        if "Gender" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("Gender and TotalClaims columns are required")
        
        # Filter for Women and Men (case-insensitive)
        gender_data = self.data[self.data["Gender"].str.upper().isin(["FEMALE", "MALE", "WOMAN", "MAN", "F", "M"], na=False)]
        
        if len(gender_data) == 0:
            raise ValueError("No valid gender data found. Check Gender column values.")
        
        # Map to standard values
        gender_map = {
            'FEMALE': 'Female', 'F': 'Female', 'WOMAN': 'Female',
            'MALE': 'Male', 'M': 'Male', 'MAN': 'Male'
        }
        gender_data = gender_data.copy()
        gender_data['Gender_Standard'] = gender_data['Gender'].str.upper().map(gender_map)
        gender_data = gender_data.dropna(subset=['Gender_Standard'])
        
        female_claims = gender_data[gender_data['Gender_Standard'] == 'Female']['TotalClaims'].values
        male_claims = gender_data[gender_data['Gender_Standard'] == 'Male']['TotalClaims'].values
        
        if len(female_claims) == 0 or len(male_claims) == 0:
            raise ValueError("Need data for both genders")
        
        # Use Mann-Whitney U test (non-parametric, more robust)
        statistic, p_value = stats.mannwhitneyu(female_claims, male_claims, alternative='two-sided')
        
        female_mean = np.mean(female_claims)
        male_mean = np.mean(male_claims)
        
        result = {
            'test_name': 'Mann-Whitney U',
            'null_hypothesis': 'There is no significant risk difference between Women and men',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'reject_null': p_value < self.alpha,
            'conclusion': 'Reject H0: Risk differences exist between genders' if p_value < self.alpha 
                         else 'Fail to reject H0: No significant risk differences between genders',
            'female_mean_claims': female_mean,
            'male_mean_claims': male_mean,
            'difference': abs(female_mean - male_mean),
            'female_count': len(female_claims),
            'male_count': len(male_claims)
        }
        
        self.results['gender_risk_test'] = result
        logger.info(f"Gender risk test: {result['conclusion']} (p={p_value:.4f})")
        return result
    
    def run_all_tests(self) -> Dict:
        """
        Run all hypothesis tests.
        
        Returns:
            Dict: All test results
        """
        logger.info("Running all hypothesis tests...")
        
        try:
            self.test_risk_difference_by_province()
        except Exception as e:
            logger.warning(f"Province test failed: {e}")
        
        try:
            self.test_risk_difference_by_zipcode()
        except Exception as e:
            logger.warning(f"Zipcode risk test failed: {e}")
        
        try:
            self.test_margin_difference_by_zipcode()
        except Exception as e:
            logger.warning(f"Zipcode margin test failed: {e}")
        
        try:
            self.test_risk_difference_by_gender()
        except Exception as e:
            logger.warning(f"Gender test failed: {e}")
        
        logger.info("All hypothesis tests completed")
        return self.results
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all test results in a DataFrame.
        
        Returns:
            pd.DataFrame: Summary of test results
        """
        summary_data = []
        
        for test_name, result in self.results.items():
            summary_data.append({
                'Test': test_name,
                'Null Hypothesis': result.get('null_hypothesis', 'N/A'),
                'Test Method': result.get('test_name', 'N/A'),
                'P-Value': result.get('p_value', np.nan),
                'Reject H0': result.get('reject_null', False),
                'Conclusion': result.get('conclusion', 'N/A')
            })
        
        return pd.DataFrame(summary_data)

