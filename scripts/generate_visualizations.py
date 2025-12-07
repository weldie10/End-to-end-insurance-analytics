"""
Generate key visualizations for the insurance analytics project.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphacare.data import DataLoader
from alphacare.eda import EDAAnalyzer
from alphacare.utils import setup_logging

# Setup
setup_logging(log_level="INFO")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path(__file__).parent.parent / "reports" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)


def visualization_1_portfolio_overview(data, eda):
    """Overall Loss Ratio and Portfolio Overview"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall Loss Ratio KPI
    overall_loss_ratio = eda.results.get('overall_loss_ratio', 0)
    axes[0, 0].text(0.5, 0.5, f'Overall Loss Ratio\n{overall_loss_ratio:.2%}', 
                    ha='center', va='center', fontsize=24, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Overall Loss Ratio', fontsize=14, fontweight='bold')
    
    # 2. Total Premium vs Total Claims over time
    if 'temporal_trends' in eda.results:
        trends = eda.results['temporal_trends']
        if 'Month' in trends.columns:
            ax2 = axes[0, 1]
            ax2_twin = ax2.twinx()
            
            premium_col = [c for c in trends.columns if 'TotalPremium' in c and 'sum' in c]
            claims_col = [c for c in trends.columns if 'TotalClaims' in c and 'sum' in c]
            
            if premium_col and claims_col:
                ax2.plot(range(len(trends)), trends[premium_col[0]], 
                        'b-', label='Total Premium', linewidth=2)
                ax2_twin.plot(range(len(trends)), trends[claims_col[0]], 
                             'r-', label='Total Claims', linewidth=2)
                ax2.set_xticks(range(0, len(trends), max(1, len(trends)//6)))
                ax2.set_xticklabels([str(trends['Month'].iloc[i]) for i in range(0, len(trends), max(1, len(trends)//6))], 
                                    rotation=45, ha='right')
                ax2.set_ylabel('Total Premium', color='b', fontweight='bold')
                ax2_twin.set_ylabel('Total Claims', color='r', fontweight='bold')
                ax2.set_title('Premium vs Claims Over Time', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
    
    # 3. Claim frequency distribution
    if 'TotalClaims' in data.columns:
        claims_data = data[data['TotalClaims'] > 0]['TotalClaims']
        axes[1, 0].hist(claims_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 0].set_xlabel('Total Claims', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Claim Frequency Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
    
    # 4. Premium distribution box plot
    if 'TotalPremium' in data.columns:
        premium_data = data[data['TotalPremium'] > 0]['TotalPremium']
        axes[1, 1].boxplot(premium_data, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1, 1].set_ylabel('Total Premium', fontweight='bold')
        axes[1, 1].set_title('Premium Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_portfolio_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 1: Portfolio Overview saved")


def visualization_2_loss_ratio_province(data, eda):
    """Loss Ratio by Province"""
    loss_ratio_by_province = eda.calculate_loss_ratio(group_by=["Province"])
    
    if 'Province' in loss_ratio_by_province.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by loss ratio
        loss_ratio_sorted = loss_ratio_by_province.sort_values('LossRatio', ascending=True)
        
        # Create horizontal bar chart
        colors = ['red' if x > 0.6 else 'orange' if x > 0.4 else 'green' 
                 for x in loss_ratio_sorted['LossRatio']]
        
        bars = ax.barh(range(len(loss_ratio_sorted)), loss_ratio_sorted['LossRatio'], 
                      color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(loss_ratio_sorted)))
        ax.set_yticklabels(loss_ratio_sorted['Province'].values)
        ax.set_xlabel('Loss Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Loss Ratio by Province', fontsize=16, fontweight='bold')
        ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2, label='High Risk (60%)')
        ax.axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Medium Risk (40%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (idx, row) in enumerate(loss_ratio_sorted.iterrows()):
            ax.text(row['LossRatio'] + 0.01, i, f"{row['LossRatio']:.2%}", 
                   va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "03_loss_ratio_province.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization 2: Loss Ratio by Province saved")


def visualization_3_loss_ratio_vehicle_type(data, eda):
    """Loss Ratio by Vehicle Type"""
    loss_ratio_by_vehicle = eda.calculate_loss_ratio(group_by=["VehicleType"])
    
    if 'VehicleType' in loss_ratio_by_vehicle.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sort by loss ratio
        loss_ratio_sorted = loss_ratio_by_vehicle.sort_values('LossRatio', ascending=False).head(10)
        
        # Bar chart
        colors = plt.cm.RdYlGn_r(loss_ratio_sorted['LossRatio'] / loss_ratio_sorted['LossRatio'].max())
        axes[0].barh(range(len(loss_ratio_sorted)), loss_ratio_sorted['LossRatio'], 
                    color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_yticks(range(len(loss_ratio_sorted)))
        axes[0].set_yticklabels(loss_ratio_sorted['VehicleType'].values)
        axes[0].set_xlabel('Loss Ratio', fontweight='bold')
        axes[0].set_title('Top 10 Vehicle Types by Loss Ratio', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Scatter plot: Premium vs Claims
        for vehicle_type in loss_ratio_sorted['VehicleType'].head(5):
            vehicle_data = data[data['VehicleType'] == vehicle_type]
            if len(vehicle_data) > 0:
                axes[1].scatter(vehicle_data['TotalPremium'].mean(), 
                               vehicle_data['TotalClaims'].mean(),
                               s=200, alpha=0.6, label=vehicle_type)
        
        axes[1].set_xlabel('Average Premium', fontweight='bold')
        axes[1].set_ylabel('Average Claims', fontweight='bold')
        axes[1].set_title('Premium vs Claims by Vehicle Type', fontsize=14, fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "04_loss_ratio_vehicle_type.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Visualization 3: Loss Ratio by Vehicle Type saved")


def visualization_4_temporal_trends(data, eda):
    """Temporal Trends Analysis"""
    trends = eda.analyze_temporal_trends()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Monthly trends
    if 'Month' in trends.columns:
        months = range(len(trends))
        
        # Premium and Claims trends
        premium_col = [c for c in trends.columns if 'TotalPremium' in c and 'sum' in c]
        claims_col = [c for c in trends.columns if 'TotalClaims' in c and 'sum' in c]
        
        if premium_col and claims_col:
            ax1 = axes[0]
            ax1_twin = ax1.twinx()
            
            line1 = ax1.plot(months, trends[premium_col[0]], 'b-o', 
                            label='Total Premium', linewidth=2, markersize=6)
            line2 = ax1_twin.plot(months, trends[claims_col[0]], 'r-s', 
                                 label='Total Claims', linewidth=2, markersize=6)
            
            ax1.set_xlabel('Month', fontweight='bold')
            ax1.set_ylabel('Total Premium', color='b', fontweight='bold')
            ax1_twin.set_ylabel('Total Claims', color='r', fontweight='bold')
            ax1.set_title('Monthly Premium and Claims Trends', fontsize=14, fontweight='bold')
            ax1.set_xticks(months[::max(1, len(months)//6)])
            ax1.set_xticklabels([str(trends['Month'].iloc[i]) for i in range(0, len(trends), max(1, len(trends)//6))], 
                               rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
        
        # Loss ratio trend
        if 'TotalPremium_sum' in trends.columns and 'TotalClaims_sum' in trends.columns:
            trends['LossRatio'] = trends['TotalClaims_sum'] / trends['TotalPremium_sum'].replace(0, np.nan)
            axes[1].plot(months, trends['LossRatio'], 'g-o', linewidth=2, markersize=6, label='Loss Ratio')
            axes[1].axhline(y=0.6, color='r', linestyle='--', linewidth=2, label='High Risk Threshold (60%)')
            axes[1].axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold (40%)')
            axes[1].fill_between(months, 0, trends['LossRatio'], alpha=0.3, color='green')
            axes[1].set_xlabel('Month', fontweight='bold')
            axes[1].set_ylabel('Loss Ratio', fontweight='bold')
            axes[1].set_title('Monthly Loss Ratio Trend', fontsize=14, fontweight='bold')
            axes[1].set_xticks(months[::max(1, len(months)//6)])
            axes[1].set_xticklabels([str(trends['Month'].iloc[i]) for i in range(0, len(trends), max(1, len(trends)//6))], 
                                   rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "07_temporal_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 4: Temporal Trends saved")


def visualization_5_vehicle_make_analysis(data, eda):
    """Top Vehicle Makes by Claims"""
    vehicle_analysis = eda.analyze_vehicle_make_model_claims()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 10 makes by total claims
    top_makes = vehicle_analysis.nlargest(10, 'TotalClaims_Sum')
    
    axes[0].barh(range(len(top_makes)), top_makes['TotalClaims_Sum'], 
                color='crimson', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(top_makes)))
    axes[0].set_yticklabels(top_makes['Make'].values)
    axes[0].set_xlabel('Total Claims (Sum)', fontweight='bold')
    axes[0].set_title('Top 10 Vehicle Makes by Total Claims', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(top_makes.iterrows()):
        axes[0].text(row['TotalClaims_Sum'] + max(top_makes['TotalClaims_Sum']) * 0.01, 
                    i, f"{row['TotalClaims_Sum']:,.0f}", 
                    va='center', fontweight='bold')
    
    # Top 10 makes by loss ratio (worst performers)
    worst_loss_ratio = vehicle_analysis.nlargest(10, 'LossRatio')
    
    colors_worst = plt.cm.Reds(np.linspace(0.4, 0.9, len(worst_loss_ratio)))
    axes[1].barh(range(len(worst_loss_ratio)), worst_loss_ratio['LossRatio'], 
                color=colors_worst, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(worst_loss_ratio)))
    axes[1].set_yticklabels(worst_loss_ratio['Make'].values)
    axes[1].set_xlabel('Loss Ratio', fontweight='bold')
    axes[1].set_title('Top 10 Vehicle Makes by Loss Ratio (Highest Risk)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(worst_loss_ratio.iterrows()):
        axes[1].text(row['LossRatio'] + 0.01, i, f"{row['LossRatio']:.2%}", 
                    va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "09_vehicle_make_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 5: Vehicle Make Analysis saved")


def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("Generating Visualizations for Insurance Analytics")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    loader = DataLoader(data_path="Data")
    data = loader.load_data("MachineLearningRating_v3.txt")
    processed_data = loader.preprocess_data()
    print(f"   ✓ Loaded {len(processed_data):,} records")
    
    # Initialize EDA
    print("\n2. Initializing EDA analyzer...")
    eda = EDAAnalyzer(processed_data)
    print("   ✓ EDA analyzer ready")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    print("-" * 60)
    
    try:
        visualization_1_portfolio_overview(processed_data, eda)
    except Exception as e:
        print(f"   ✗ Error in Visualization 1: {e}")
    
    try:
        visualization_2_loss_ratio_province(processed_data, eda)
    except Exception as e:
        print(f"   ✗ Error in Visualization 2: {e}")
    
    try:
        visualization_3_loss_ratio_vehicle_type(processed_data, eda)
    except Exception as e:
        print(f"   ✗ Error in Visualization 3: {e}")
    
    try:
        visualization_4_temporal_trends(processed_data, eda)
    except Exception as e:
        print(f"   ✗ Error in Visualization 4: {e}")
    
    try:
        visualization_5_vehicle_make_analysis(processed_data, eda)
    except Exception as e:
        print(f"   ✗ Error in Visualization 5: {e}")
    
    print("-" * 60)
    print(f"\n✓ Visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

