"""
A/B Risk Hypothesis Testing Module

This module provides the ABRiskHypothesisTester class for performing A/B tests
and statistical hypothesis testing on insurance risk data.

The class tests risk differences across various dimensions (provinces, zipcodes, gender)
using Claim Frequency and Claim Severity metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ABRiskHypothesisTester:
    """
    A class for performing A/B statistical hypothesis tests on insurance risk data.
    
    This class provides methods for A/B testing, comparing groups,
    and testing various hypotheses about risk differences using:
    - Claim Frequency: proportion of policies with at least one claim
    - Claim Severity: average claim amount given a claim occurred
    - Margin: TotalPremium - TotalClaims
    
    Attributes:
        data (pd.DataFrame): The dataset to analyze
        results (Dict): Dictionary to store test results
        alpha (float): Significance level (default: 0.05)
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize the ABRiskHypothesisTester.
        
        Args:
            data: DataFrame containing the insurance data
            alpha: Significance level for tests (default: 0.05)
        """
        self.data = data.copy()
        self.results: Dict = {}
        self.alpha = alpha
        
        # Calculate risk metrics
        # Claim Frequency: proportion of policies with at least one claim
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        
        # Claim Severity: average claim amount for policies with claims
        # Will be calculated per group in tests
        
        logger.info(f"ABRiskHypothesisTester initialized with alpha={alpha}")
    
    def _calculate_claim_frequency(self, group_data: pd.DataFrame) -> float:
        """Calculate claim frequency (proportion with at least one claim)."""
        if len(group_data) == 0:
            return 0.0
        return group_data['HasClaim'].mean()
    
    def _calculate_claim_severity(self, group_data: pd.DataFrame) -> float:
        """Calculate claim severity (average claim amount given a claim occurred)."""
        claims_data = group_data[group_data['TotalClaims'] > 0]
        if len(claims_data) == 0:
            return 0.0
        return claims_data['TotalClaims'].mean()
    
    def test_risk_difference_by_province(self) -> Dict:
        """
        Test the null hypothesis: There are no risk differences across provinces.
        
        Tests both Claim Frequency (chi-square) and Claim Severity (Kruskal-Wallis/ANOVA).
        
        Returns:
            Dict: Test results including p-value, statistic, and conclusion
        """
        if "Province" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("Province and TotalClaims columns are required")
        
        provinces = self.data["Province"].unique()
        province_data = [self.data[self.data["Province"] == prov] for prov in provinces 
                        if len(self.data[self.data["Province"] == prov]) > 0]
        
        if len(province_data) < 2:
            raise ValueError("Need at least 2 provinces for comparison")
        
        # Test 1: Claim Frequency (Chi-square test)
        # Create contingency table: Province x HasClaim
        contingency = pd.crosstab(self.data['Province'], self.data['HasClaim'])
        if contingency.shape[1] < 2:
            # All provinces have same claim status
            freq_statistic, freq_p_value = np.nan, 1.0
            freq_test_name = "Chi-square (not applicable)"
        else:
            freq_statistic, freq_p_value, _, _ = stats.chi2_contingency(contingency)
            freq_test_name = "Chi-square"
        
        # Test 2: Claim Severity (Kruskal-Wallis test on claim amounts for those with claims)
        severity_groups = []
        for prov_data in province_data:
            claims_only = prov_data[prov_data['TotalClaims'] > 0]['TotalClaims'].values
            if len(claims_only) > 0:
                severity_groups.append(claims_only)
        
        if len(severity_groups) < 2:
            sev_statistic, sev_p_value = np.nan, 1.0
            sev_test_name = "Kruskal-Wallis (not applicable)"
        else:
            sev_statistic, sev_p_value = stats.kruskal(*severity_groups)
            sev_test_name = "Kruskal-Wallis"
        
        # Calculate metrics for each province
        province_metrics = {}
        for prov in provinces:
            prov_data = self.data[self.data["Province"] == prov]
            if len(prov_data) > 0:
                province_metrics[prov] = {
                    'frequency': self._calculate_claim_frequency(prov_data),
                    'severity': self._calculate_claim_severity(prov_data),
                    'count': len(prov_data),
                    'claims_count': prov_data['HasClaim'].sum()
                }
        
        # Overall conclusion: reject if either frequency or severity shows significance
        reject_freq = freq_p_value < self.alpha if not np.isnan(freq_p_value) else False
        reject_sev = sev_p_value < self.alpha if not np.isnan(sev_p_value) else False
        reject_null = reject_freq or reject_sev
        
        result = {
            'test_name': f'{freq_test_name} (Frequency) & {sev_test_name} (Severity)',
            'null_hypothesis': 'There are no risk differences across provinces',
            'frequency_test': freq_test_name,
            'frequency_statistic': freq_statistic,
            'frequency_p_value': freq_p_value,
            'severity_test': sev_test_name,
            'severity_statistic': sev_statistic,
            'severity_p_value': sev_p_value,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Reject H0: Risk differences exist across provinces' if reject_null 
                         else 'Fail to reject H0: No significant risk differences across provinces',
            'province_metrics': province_metrics
        }
        
        self.results['province_risk_test'] = result
        logger.info(f"Province risk test: {result['conclusion']} (freq p={freq_p_value:.4f}, sev p={sev_p_value:.4f})")
        return result
    
    def test_risk_difference_by_zipcode(self) -> Dict:
        """
        Test the null hypothesis: There are no risk differences between zipcodes.
        
        Tests both Claim Frequency (chi-square) and Claim Severity (Kruskal-Wallis).
        
        Returns:
            Dict: Test results
        """
        if "PostalCode" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("PostalCode and TotalClaims columns are required")
        
        # Sample zipcodes if too many (for computational efficiency)
        zipcode_counts = self.data["PostalCode"].value_counts()
        top_zipcodes = zipcode_counts.head(20).index  # Top 20 zipcodes
        
        zipcode_data = self.data[self.data["PostalCode"].isin(top_zipcodes)]
        zipcode_data = zipcode_data[zipcode_data.groupby("PostalCode")["PostalCode"].transform('count') > 10]
        
        if len(zipcode_data) == 0:
            raise ValueError("No zipcodes with sufficient data")
        
        unique_zipcodes = zipcode_data["PostalCode"].unique()
        if len(unique_zipcodes) < 2:
            raise ValueError("Need at least 2 zipcodes with sufficient data")
        
        # Test 1: Claim Frequency (Chi-square test)
        contingency = pd.crosstab(zipcode_data['PostalCode'], zipcode_data['HasClaim'])
        if contingency.shape[1] < 2:
            freq_statistic, freq_p_value = np.nan, 1.0
            freq_test_name = "Chi-square (not applicable)"
        else:
            freq_statistic, freq_p_value, _, _ = stats.chi2_contingency(contingency)
            freq_test_name = "Chi-square"
        
        # Test 2: Claim Severity (Kruskal-Wallis test on claim amounts for those with claims)
        severity_groups = []
        for zc in unique_zipcodes:
            zc_data = zipcode_data[zipcode_data["PostalCode"] == zc]
            claims_only = zc_data[zc_data['TotalClaims'] > 0]['TotalClaims'].values
            if len(claims_only) > 0:
                severity_groups.append(claims_only)
        
        if len(severity_groups) < 2:
            sev_statistic, sev_p_value = np.nan, 1.0
            sev_test_name = "Kruskal-Wallis (not applicable)"
        else:
            sev_statistic, sev_p_value = stats.kruskal(*severity_groups)
            sev_test_name = "Kruskal-Wallis"
        
        # Calculate metrics for each zipcode
        zipcode_metrics = {}
        for zc in unique_zipcodes:
            zc_data = zipcode_data[zipcode_data["PostalCode"] == zc]
            if len(zc_data) > 0:
                zipcode_metrics[zc] = {
                    'frequency': self._calculate_claim_frequency(zc_data),
                    'severity': self._calculate_claim_severity(zc_data),
                    'count': len(zc_data),
                    'claims_count': zc_data['HasClaim'].sum()
                }
        
        # Overall conclusion
        reject_freq = freq_p_value < self.alpha if not np.isnan(freq_p_value) else False
        reject_sev = sev_p_value < self.alpha if not np.isnan(sev_p_value) else False
        reject_null = reject_freq or reject_sev
        
        result = {
            'test_name': f'{freq_test_name} (Frequency) & {sev_test_name} (Severity)',
            'null_hypothesis': 'There are no risk differences between zipcodes',
            'frequency_test': freq_test_name,
            'frequency_statistic': freq_statistic,
            'frequency_p_value': freq_p_value,
            'severity_test': sev_test_name,
            'severity_statistic': sev_statistic,
            'severity_p_value': sev_p_value,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Reject H0: Risk differences exist between zipcodes' if reject_null 
                         else 'Fail to reject H0: No significant risk differences between zipcodes',
            'zipcode_metrics': zipcode_metrics,
            'zipcodes_tested': len(unique_zipcodes)
        }
        
        self.results['zipcode_risk_test'] = result
        logger.info(f"Zipcode risk test: {result['conclusion']} (freq p={freq_p_value:.4f}, sev p={sev_p_value:.4f})")
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
        
        Tests both Claim Frequency (chi-square) and Claim Severity (Mann-Whitney U).
        
        Returns:
            Dict: Test results
        """
        if "Gender" not in self.data.columns or "TotalClaims" not in self.data.columns:
            raise ValueError("Gender and TotalClaims columns are required")
        
        # Filter for Women and Men (case-insensitive)
        gender_upper = self.data["Gender"].str.upper()
        valid_genders = ["FEMALE", "MALE", "WOMAN", "MAN", "F", "M"]
        gender_data = self.data[gender_upper.isin(valid_genders)].copy()
        
        if len(gender_data) == 0:
            raise ValueError("No valid gender data found. Check Gender column values.")
        
        # Map to standard values
        gender_map = {
            'FEMALE': 'Female', 'F': 'Female', 'WOMAN': 'Female',
            'MALE': 'Male', 'M': 'Male', 'MAN': 'Male'
        }
        gender_data['Gender_Standard'] = gender_data['Gender'].str.upper().map(gender_map)
        gender_data = gender_data.dropna(subset=['Gender_Standard'])
        
        female_data = gender_data[gender_data['Gender_Standard'] == 'Female']
        male_data = gender_data[gender_data['Gender_Standard'] == 'Male']
        
        if len(female_data) == 0 or len(male_data) == 0:
            raise ValueError("Need data for both genders")
        
        # Test 1: Claim Frequency (Chi-square test)
        contingency = pd.crosstab(gender_data['Gender_Standard'], gender_data['HasClaim'])
        if contingency.shape[1] < 2:
            freq_statistic, freq_p_value = np.nan, 1.0
            freq_test_name = "Chi-square (not applicable)"
        else:
            freq_statistic, freq_p_value, _, _ = stats.chi2_contingency(contingency)
            freq_test_name = "Chi-square"
        
        # Test 2: Claim Severity (Mann-Whitney U test on claim amounts for those with claims)
        female_severity = female_data[female_data['TotalClaims'] > 0]['TotalClaims'].values
        male_severity = male_data[male_data['TotalClaims'] > 0]['TotalClaims'].values
        
        if len(female_severity) == 0 or len(male_severity) == 0:
            sev_statistic, sev_p_value = np.nan, 1.0
            sev_test_name = "Mann-Whitney U (not applicable)"
        else:
            sev_statistic, sev_p_value = stats.mannwhitneyu(female_severity, male_severity, alternative='two-sided')
            sev_test_name = "Mann-Whitney U"
        
        # Calculate metrics
        female_frequency = self._calculate_claim_frequency(female_data)
        male_frequency = self._calculate_claim_frequency(male_data)
        female_severity_mean = self._calculate_claim_severity(female_data)
        male_severity_mean = self._calculate_claim_severity(male_data)
        
        # Overall conclusion
        reject_freq = freq_p_value < self.alpha if not np.isnan(freq_p_value) else False
        reject_sev = sev_p_value < self.alpha if not np.isnan(sev_p_value) else False
        reject_null = reject_freq or reject_sev
        
        result = {
            'test_name': f'{freq_test_name} (Frequency) & {sev_test_name} (Severity)',
            'null_hypothesis': 'There is no significant risk difference between Women and men',
            'frequency_test': freq_test_name,
            'frequency_statistic': freq_statistic,
            'frequency_p_value': freq_p_value,
            'severity_test': sev_test_name,
            'severity_statistic': sev_statistic,
            'severity_p_value': sev_p_value,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'conclusion': 'Reject H0: Risk differences exist between genders' if reject_null 
                         else 'Fail to reject H0: No significant risk differences between genders',
            'female_frequency': female_frequency,
            'male_frequency': male_frequency,
            'female_severity': female_severity_mean,
            'male_severity': male_severity_mean,
            'frequency_difference': abs(female_frequency - male_frequency),
            'severity_difference': abs(female_severity_mean - male_severity_mean),
            'female_count': len(female_data),
            'male_count': len(male_data)
        }
        
        self.results['gender_risk_test'] = result
        logger.info(f"Gender risk test: {result['conclusion']} (freq p={freq_p_value:.4f}, sev p={sev_p_value:.4f})")
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
            # Handle both old format (single p_value) and new format (frequency/severity)
            if 'frequency_p_value' in result:
                # New format with frequency and severity
                freq_p = result.get('frequency_p_value', np.nan)
                sev_p = result.get('severity_p_value', np.nan)
                p_value_str = f"Freq: {freq_p:.4f}, Sev: {sev_p:.4f}"
                overall_p = min(freq_p, sev_p) if not (np.isnan(freq_p) or np.isnan(sev_p)) else (freq_p if not np.isnan(freq_p) else sev_p)
            else:
                # Old format (for margin test)
                overall_p = result.get('p_value', np.nan)
                p_value_str = f"{overall_p:.4f}"
            
            summary_data.append({
                'Test': test_name,
                'Null Hypothesis': result.get('null_hypothesis', 'N/A'),
                'Test Method': result.get('test_name', 'N/A'),
                'P-Value': p_value_str,
                'Overall P-Value': overall_p,
                'Reject H0': result.get('reject_null', False),
                'Conclusion': result.get('conclusion', 'N/A')
            })
        
        return pd.DataFrame(summary_data)

