"""
Comprehensive Hypothesis Testing Script for Task 3

This script performs A/B hypothesis testing for risk drivers:
1. Risk differences across provinces
2. Risk differences between zip codes
3. Margin differences between zip codes
4. Risk differences between Women and Men

Metrics used:
- Claim Frequency: proportion of policies with at least one claim
- Claim Severity: average claim amount given a claim occurred
- Margin: TotalPremium - TotalClaims
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphacare.data import DataLoader
from alphacare.statistics import ABRiskHypothesisTester
from alphacare.utils import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def format_p_value(p_value: float) -> str:
    """Format p-value with significance indicators."""
    if np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return f"{p_value:.4f} ***"
    elif p_value < 0.01:
        return f"{p_value:.4f} **"
    elif p_value < 0.05:
        return f"{p_value:.4f} *"
    else:
        return f"{p_value:.4f}"


def analyze_province_results(result: dict) -> str:
    """Analyze and format province test results."""
    output = []
    
    freq_p = result.get('frequency_p_value', np.nan)
    sev_p = result.get('severity_p_value', np.nan)
    reject = result.get('reject_null', False)
    
    output.append(f"\nTest Results:")
    output.append(f"  Claim Frequency Test: {result.get('frequency_test', 'N/A')}")
    output.append(f"    P-value: {format_p_value(freq_p)}")
    output.append(f"  Claim Severity Test: {result.get('severity_test', 'N/A')}")
    output.append(f"    P-value: {format_p_value(sev_p)}")
    output.append(f"\nOverall Conclusion: {result.get('conclusion', 'N/A')}")
    
    # Province metrics
    metrics = result.get('province_metrics', {})
    if metrics:
        output.append(f"\nProvince Metrics:")
        output.append(f"{'Province':<20} {'Frequency':<12} {'Severity':<15} {'Count':<10} {'Claims':<10}")
        output.append("-" * 70)
        
        # Sort by frequency (descending)
        sorted_provinces = sorted(metrics.items(), key=lambda x: x[1]['frequency'], reverse=True)
        for prov, m in sorted_provinces:
            output.append(f"{prov:<20} {m['frequency']:<12.4f} {m['severity']:<15.2f} {m['count']:<10} {m['claims_count']:<10}")
    
    # Business interpretation
    if reject:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We reject the null hypothesis (p < 0.05).")
        if not np.isnan(freq_p) and freq_p < 0.05:
            output.append(f"  - Claim frequency varies significantly across provinces (p = {freq_p:.4f})")
        if not np.isnan(sev_p) and sev_p < 0.05:
            output.append(f"  - Claim severity varies significantly across provinces (p = {sev_p:.4f})")
        
        # Find highest and lowest risk provinces
        if metrics:
            highest_freq = max(metrics.items(), key=lambda x: x[1]['frequency'])
            lowest_freq = min(metrics.items(), key=lambda x: x[1]['frequency'])
            output.append(f"\n  Risk Insights:")
            output.append(f"  - Highest claim frequency: {highest_freq[0]} ({highest_freq[1]['frequency']:.2%})")
            output.append(f"  - Lowest claim frequency: {lowest_freq[0]} ({lowest_freq[1]['frequency']:.2%})")
            output.append(f"  - Difference: {(highest_freq[1]['frequency'] - lowest_freq[1]['frequency']):.2%}")
            output.append(f"\n  Recommendation: Consider regional premium adjustments based on province risk profiles.")
    else:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We fail to reject the null hypothesis (p >= 0.05).")
        output.append(f"  No significant risk differences across provinces detected.")
        output.append(f"  Recommendation: Uniform pricing across provinces may be appropriate.")
    
    return "\n".join(output)


def analyze_zipcode_risk_results(result: dict) -> str:
    """Analyze and format zipcode risk test results."""
    output = []
    
    freq_p = result.get('frequency_p_value', np.nan)
    sev_p = result.get('severity_p_value', np.nan)
    reject = result.get('reject_null', False)
    n_zipcodes = result.get('zipcodes_tested', 0)
    
    output.append(f"\nTest Results:")
    output.append(f"  Zipcodes Tested: {n_zipcodes}")
    output.append(f"  Claim Frequency Test: {result.get('frequency_test', 'N/A')}")
    output.append(f"    P-value: {format_p_value(freq_p)}")
    output.append(f"  Claim Severity Test: {result.get('severity_test', 'N/A')}")
    output.append(f"    P-value: {format_p_value(sev_p)}")
    output.append(f"\nOverall Conclusion: {result.get('conclusion', 'N/A')}")
    
    # Top 5 and bottom 5 zipcodes
    metrics = result.get('zipcode_metrics', {})
    if metrics:
        sorted_zipcodes = sorted(metrics.items(), key=lambda x: x[1]['frequency'], reverse=True)
        output.append(f"\nTop 5 Highest Risk Zipcodes (by frequency):")
        output.append(f"{'Zipcode':<15} {'Frequency':<12} {'Severity':<15} {'Count':<10}")
        output.append("-" * 55)
        for zc, m in sorted_zipcodes[:5]:
            output.append(f"{str(zc):<15} {m['frequency']:<12.4f} {m['severity']:<15.2f} {m['count']:<10}")
        
        output.append(f"\nTop 5 Lowest Risk Zipcodes (by frequency):")
        output.append(f"{'Zipcode':<15} {'Frequency':<12} {'Severity':<15} {'Count':<10}")
        output.append("-" * 55)
        for zc, m in sorted_zipcodes[-5:]:
            output.append(f"{str(zc):<15} {m['frequency']:<12.4f} {m['severity']:<15.2f} {m['count']:<10}")
    
    # Business interpretation
    if reject:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We reject the null hypothesis (p < 0.05).")
        output.append(f"  Risk differences exist between zipcodes.")
        output.append(f"  Recommendation: Implement zipcode-based pricing tiers to reflect risk differences.")
    else:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We fail to reject the null hypothesis (p >= 0.05).")
        output.append(f"  No significant risk differences between zipcodes detected.")
    
    return "\n".join(output)


def analyze_zipcode_margin_results(result: dict) -> str:
    """Analyze and format zipcode margin test results."""
    output = []
    
    p_value = result.get('p_value', np.nan)
    reject = result.get('reject_null', False)
    n_zipcodes = result.get('zipcodes_tested', 0)
    
    output.append(f"\nTest Results:")
    output.append(f"  Zipcodes Tested: {n_zipcodes}")
    output.append(f"  Test Method: {result.get('test_name', 'N/A')}")
    output.append(f"  P-value: {format_p_value(p_value)}")
    output.append(f"\nConclusion: {result.get('conclusion', 'N/A')}")
    
    # Top and bottom zipcodes by margin
    margins = result.get('zipcode_margins', {})
    if margins:
        sorted_margins = sorted(margins.items(), key=lambda x: x[1]['mean'], reverse=True)
        output.append(f"\nTop 5 Most Profitable Zipcodes:")
        output.append(f"{'Zipcode':<15} {'Mean Margin':<15} {'Std Dev':<15} {'Count':<10}")
        output.append("-" * 60)
        for zc, m in sorted_margins[:5]:
            output.append(f"{str(zc):<15} {m['mean']:<15.2f} {m['std']:<15.2f} {m['count']:<10}")
        
        output.append(f"\nTop 5 Least Profitable Zipcodes:")
        output.append(f"{'Zipcode':<15} {'Mean Margin':<15} {'Std Dev':<15} {'Count':<10}")
        output.append("-" * 60)
        for zc, m in sorted_margins[-5:]:
            output.append(f"{str(zc):<15} {m['mean']:<15.2f} {m['std']:<15.2f} {m['count']:<10}")
    
    # Business interpretation
    if reject:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We reject the null hypothesis (p < 0.05).")
        output.append(f"  Significant margin differences exist between zipcodes.")
        output.append(f"  Recommendation: Adjust pricing strategy to optimize margins by zipcode.")
    else:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We fail to reject the null hypothesis (p >= 0.05).")
        output.append(f"  No significant margin differences between zipcodes detected.")
    
    return "\n".join(output)


def analyze_gender_results(result: dict) -> str:
    """Analyze and format gender test results."""
    output = []
    
    freq_p = result.get('frequency_p_value', np.nan)
    sev_p = result.get('severity_p_value', np.nan)
    reject = result.get('reject_null', False)
    
    output.append(f"\nTest Results:")
    output.append(f"  Claim Frequency Test: {result.get('frequency_test', 'N/A')}")
    output.append(f"    P-value: {format_p_value(freq_p)}")
    output.append(f"  Claim Severity Test: {result.get('severity_test', 'N/A')}")
    output.append(f"    P-value: {format_p_value(sev_p)}")
    output.append(f"\nOverall Conclusion: {result.get('conclusion', 'N/A')}")
    
    # Gender metrics
    output.append(f"\nGender Metrics:")
    output.append(f"  Female:")
    output.append(f"    Claim Frequency: {result.get('female_frequency', 0):.4f} ({result.get('female_frequency', 0):.2%})")
    output.append(f"    Claim Severity: {result.get('female_severity', 0):.2f}")
    output.append(f"    Sample Size: {result.get('female_count', 0):,}")
    output.append(f"  Male:")
    output.append(f"    Claim Frequency: {result.get('male_frequency', 0):.4f} ({result.get('male_frequency', 0):.2%})")
    output.append(f"    Claim Severity: {result.get('male_severity', 0):.2f}")
    output.append(f"    Sample Size: {result.get('male_count', 0):,}")
    output.append(f"\n  Frequency Difference: {result.get('frequency_difference', 0):.4f} ({result.get('frequency_difference', 0):.2%})")
    output.append(f"  Severity Difference: {result.get('severity_difference', 0):.2f}")
    
    # Business interpretation
    if reject:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We reject the null hypothesis (p < 0.05).")
        if not np.isnan(freq_p) and freq_p < 0.05:
            output.append(f"  - Claim frequency differs significantly between genders (p = {freq_p:.4f})")
        if not np.isnan(sev_p) and sev_p < 0.05:
            output.append(f"  - Claim severity differs significantly between genders (p = {sev_p:.4f})")
        
        female_freq = result.get('female_frequency', 0)
        male_freq = result.get('male_frequency', 0)
        if female_freq > male_freq:
            output.append(f"\n  Women have {((female_freq - male_freq) / male_freq * 100):.1f}% higher claim frequency than men.")
        else:
            output.append(f"\n  Men have {((male_freq - female_freq) / female_freq * 100):.1f}% higher claim frequency than women.")
        
        output.append(f"  Recommendation: Consider gender-based premium adjustments if legally permissible.")
    else:
        output.append(f"\n📊 Business Interpretation:")
        output.append(f"  We fail to reject the null hypothesis (p >= 0.05).")
        output.append(f"  No significant risk differences between genders detected.")
        output.append(f"  Recommendation: Gender-neutral pricing may be appropriate.")
    
    return "\n".join(output)


def generate_visualizations(tester: ABRiskHypothesisTester, output_dir: Path):
    """Generate visualizations for hypothesis test results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = tester.results
    
    # 1. Province Risk Analysis
    if 'province_risk_test' in results:
        result = results['province_risk_test']
        metrics = result.get('province_metrics', {})
        if metrics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Frequency plot
            provinces = list(metrics.keys())
            frequencies = [m['frequency'] for m in metrics.values()]
            sorted_data = sorted(zip(provinces, frequencies), key=lambda x: x[1], reverse=True)
            provs, freqs = zip(*sorted_data)
            
            ax1.barh(provs, freqs, color='steelblue')
            ax1.set_xlabel('Claim Frequency', fontsize=12)
            ax1.set_title('Claim Frequency by Province', fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            
            # Severity plot
            severities = [m['severity'] for m in metrics.values()]
            sorted_data = sorted(zip(provinces, severities), key=lambda x: x[1], reverse=True)
            provs, sevs = zip(*sorted_data)
            
            ax2.barh(provs, sevs, color='coral')
            ax2.set_xlabel('Claim Severity (Average Claim Amount)', fontsize=12)
            ax2.set_title('Claim Severity by Province', fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'hypothesis_province_risk.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: hypothesis_province_risk.png")
    
    # 2. Gender Risk Analysis
    if 'gender_risk_test' in results:
        result = results['gender_risk_test']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        genders = ['Female', 'Male']
        frequencies = [result.get('female_frequency', 0), result.get('male_frequency', 0)]
        severities = [result.get('female_severity', 0), result.get('male_severity', 0)]
        
        ax1.bar(genders, frequencies, color=['pink', 'lightblue'])
        ax1.set_ylabel('Claim Frequency', fontsize=12)
        ax1.set_title('Claim Frequency by Gender', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(genders, severities, color=['pink', 'lightblue'])
        ax2.set_ylabel('Claim Severity (Average Claim Amount)', fontsize=12)
        ax2.set_title('Claim Severity by Gender', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hypothesis_gender_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: hypothesis_gender_risk.png")
    
    # 3. Summary of all tests
    if results:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        test_names = []
        p_values = []
        colors = []
        
        for test_name, result in results.items():
            if 'frequency_p_value' in result:
                # Use minimum of frequency and severity p-values
                freq_p = result.get('frequency_p_value', np.nan)
                sev_p = result.get('severity_p_value', np.nan)
                p_val = min(freq_p, sev_p) if not (np.isnan(freq_p) or np.isnan(sev_p)) else (freq_p if not np.isnan(freq_p) else sev_p)
            else:
                p_val = result.get('p_value', np.nan)
            
            test_names.append(test_name.replace('_', ' ').title())
            p_values.append(p_val if not np.isnan(p_val) else 1.0)
            colors.append('red' if (p_val < 0.05 if not np.isnan(p_val) else False) else 'gray')
        
        bars = ax.barh(test_names, p_values, color=colors)
        ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (α=0.05)')
        ax.set_xlabel('P-Value', fontsize=12)
        ax.set_title('Hypothesis Test Results Summary', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(p_values) * 1.1)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # Add p-value labels
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if not np.isnan(p_val) and p_val < 1.0:
                ax.text(min(p_val + 0.01, max(p_values) * 0.9), i, f'{p_val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        try:
            plt.savefig(output_dir / 'hypothesis_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: hypothesis_summary.png")
        except Exception as e:
            print(f"⚠ Warning: Could not save summary plot: {e}")
            plt.close()


def main():
    """Main function to run all hypothesis tests."""
    print_section("Task 3: Statistical Hypothesis Testing for Risk Drivers")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "reports" / "hypothesis_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print_section("Step 1: Data Loading and Preprocessing")
    loader = DataLoader(data_path=project_root / "Data")
    data = loader.load_data("MachineLearningRating_v3.txt")
    processed_data = loader.preprocess_data()
    print(f"✓ Loaded {len(processed_data):,} records")
    print(f"✓ Columns: {len(processed_data.columns)}")
    
    # Step 2: Initialize A/B Risk Hypothesis Tester
    print_section("Step 2: Initialize A/B Risk Hypothesis Tester")
    tester = ABRiskHypothesisTester(processed_data, alpha=0.05)
    print(f"✓ A/B Risk Hypothesis Tester initialized with α = 0.05")
    
    # Step 3: Run all hypothesis tests
    print_section("Step 3: Running Hypothesis Tests")
    
    # Test 1: Province Risk Differences
    print_subsection("Test 1: Risk Differences Across Provinces")
    try:
        result1 = tester.test_risk_difference_by_province()
        print(analyze_province_results(result1))
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error(f"Province test failed: {e}", exc_info=True)
    
    # Test 2: Zipcode Risk Differences
    print_subsection("Test 2: Risk Differences Between Zipcodes")
    try:
        result2 = tester.test_risk_difference_by_zipcode()
        print(analyze_zipcode_risk_results(result2))
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error(f"Zipcode risk test failed: {e}", exc_info=True)
    
    # Test 3: Zipcode Margin Differences
    print_subsection("Test 3: Margin Differences Between Zipcodes")
    try:
        result3 = tester.test_margin_difference_by_zipcode()
        print(analyze_zipcode_margin_results(result3))
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error(f"Zipcode margin test failed: {e}", exc_info=True)
    
    # Test 4: Gender Risk Differences
    print_subsection("Test 4: Risk Differences Between Women and Men")
    try:
        result4 = tester.test_risk_difference_by_gender()
        print(analyze_gender_results(result4))
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error(f"Gender test failed: {e}", exc_info=True)
    
    # Step 4: Generate Summary
    print_section("Step 4: Test Results Summary")
    summary = tester.get_results_summary()
    print("\n" + summary.to_string(index=False))
    
    # Save summary to CSV
    summary_path = output_dir / "hypothesis_test_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Step 5: Generate Visualizations
    print_section("Step 5: Generating Visualizations")
    try:
        generate_visualizations(tester, output_dir)
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
    
    # Step 6: Save detailed results
    print_section("Step 6: Saving Detailed Results")
    import json
    
    # Convert results to JSON-serializable format
    results_json = {}
    for test_name, result in tester.results.items():
        results_json[test_name] = {}
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating)):
                results_json[test_name][key] = float(value)
            elif isinstance(value, np.ndarray):
                results_json[test_name][key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested dicts (like province_metrics)
                results_json[test_name][key] = {}
                for k, v in value.items():
                    # Convert keys to strings if they're not
                    k_str = str(k) if not isinstance(k, str) else k
                    if isinstance(v, dict):
                        results_json[test_name][key][k_str] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating)) else v2 
                                                         for k2, v2 in v.items()}
                    else:
                        results_json[test_name][key][k_str] = float(v) if isinstance(v, (np.integer, np.floating)) else v
            else:
                results_json[test_name][key] = value
    
    results_path = output_dir / "hypothesis_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"✓ Detailed results saved to: {results_path}")
    
    print_section("Task 3 Complete!")
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext Steps:")
    print("  1. Review the hypothesis test summary")
    print("  2. Examine visualizations in reports/hypothesis_tests/")
    print("  3. Update REPORT.md with findings and business recommendations")


if __name__ == "__main__":
    main()

