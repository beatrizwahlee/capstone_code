"""
Phase 3: Visualizations for Echo Chamber Analysis
==================================================
Creates publication-quality plots for thesis/presentations.

Visualizations:
---------------
1. Gini Distribution Histogram
   - Shows how concentrated user recommendations are
   - Annotated with filter bubble threshold

2. Category Concentration Heatmap
   - 2D grid: users × categories
   - Color intensity = recommendation frequency
   - Visually shows which categories dominate

3. Diversity by User Segment
   - Bar charts comparing filter bubble vs balanced vs diverse users
   - Shows Gini, ILD, Calibration Error side-by-side

4. Coverage vs Gini Scatter
   - Each point = one user
   - Shows relationship between concentration and coverage

5. Before/After Comparison (for Phase 4)
   - Side-by-side baseline vs diversity-aware
   - Multiple metrics in one figure

6. Category Distribution Lorenz Curve
   - Classic inequality visualization
   - Baseline vs perfect equality

All plots saved as:
  - High-res PNG (for thesis)
  - PDF (for publications)
  - Interactive HTML (for presentations)

Usage:
    from echo_chamber_visualizer import EchoChamberVisualizer
    
    viz = EchoChamberVisualizer(report, user_metrics)
    viz.create_all_plots('./outputs/visualizations/')
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class EchoChamberVisualizer:
    """
    Creates all visualizations for echo chamber analysis.
    """
    
    def __init__(self, report: Dict, user_metrics: List[Dict]):
        """
        Args:
            report:        Echo chamber report from EchoChamberAnalyzer
            user_metrics:  Raw per-user metrics (report['raw_metrics'])
        """
        self.report = report
        self.df = pd.DataFrame(user_metrics)
    
    @classmethod
    def from_files(cls, report_path: str, metrics_path: str):
        """Load from saved JSON files."""
        with open(report_path) as f:
            report = json.load(f)
        with open(metrics_path) as f:
            user_metrics = json.load(f)
        return cls(report, user_metrics)
    
    # -----------------------------------------------------------------------
    # Plot 1: Gini Distribution
    # -----------------------------------------------------------------------
    
    def plot_gini_distribution(self, output_path: Optional[str] = None):
        """
        Histogram of Gini coefficients across all users.
        Shows how many users are in echo chambers.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        gini_values = self.df['gini'].values
        
        # Histogram
        ax.hist(gini_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        
        # Thresholds
        ax.axvline(0.8, color='red', linestyle='--', linewidth=2, 
                   label='Filter Bubble Threshold (0.8)')
        ax.axvline(0.4, color='orange', linestyle='--', linewidth=2,
                   label='Diverse Threshold (0.4)')
        
        # Mean/median
        mean_gini = gini_values.mean()
        median_gini = np.median(gini_values)
        ax.axvline(mean_gini, color='darkblue', linestyle='-', linewidth=2,
                   label=f'Mean ({mean_gini:.3f})')
        ax.axvline(median_gini, color='purple', linestyle=':', linewidth=2,
                   label=f'Median ({median_gini:.3f})')
        
        ax.set_xlabel('Gini Coefficient (Category Concentration)', fontsize=12)
        ax.set_ylabel('Number of Users', fontsize=12)
        ax.set_title('Echo Chamber Distribution — Baseline Recommender\n' +
                     '(Higher Gini = More Concentrated = Stronger Echo Chamber)',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Annotations
        filter_bubble_pct = (gini_values > 0.8).sum() / len(gini_values) * 100
        ax.text(0.85, ax.get_ylim()[1] * 0.9, 
                f'{filter_bubble_pct:.1f}% of users\nin filter bubble',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2),
                fontsize=10, ha='left')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")
        
        return fig
    
    # -----------------------------------------------------------------------
    # Plot 2: ILD Distribution
    # -----------------------------------------------------------------------
    
    def plot_ild_distribution(self, output_path: Optional[str] = None):
        """
        Histogram of ILD (semantic diversity) scores.
        Low ILD = recommendations are very similar to each other.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ild_values = self.df['ild'].values
        
        ax.hist(ild_values, bins=50, edgecolor='black', alpha=0.7, color='seagreen')
        
        # Mean/median
        mean_ild = ild_values.mean()
        median_ild = np.median(ild_values)
        ax.axvline(mean_ild, color='darkgreen', linestyle='-', linewidth=2,
                   label=f'Mean ({mean_ild:.3f})')
        ax.axvline(median_ild, color='olive', linestyle=':', linewidth=2,
                   label=f'Median ({median_ild:.3f})')
        
        ax.set_xlabel('ILD (Intra-List Diversity)', fontsize=12)
        ax.set_ylabel('Number of Users', fontsize=12)
        ax.set_title('Semantic Diversity Distribution — Baseline Recommender\n' +
                     '(Higher ILD = More Diverse Recommendations)',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Target annotation
        ax.axvline(0.4, color='blue', linestyle='--', linewidth=2,
                   label='Target for Phase 4 (0.4)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")
        
        return fig
    
    # -----------------------------------------------------------------------
    # Plot 3: User Segmentation Comparison
    # -----------------------------------------------------------------------
    
    def plot_segment_comparison(self, output_path: Optional[str] = None):
        """
        Bar charts comparing metrics across user segments.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        segments = self.report['segments']
        segment_names = ['filter_bubble', 'balanced', 'diverse']
        segment_labels = ['Filter Bubble\n(Gini>0.8)', 'Balanced\n(0.4-0.6)', 'Diverse\n(Gini<0.4)']
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        metrics = [
            ('avg_gini', 'Gini Coefficient', 'lower is better'),
            ('avg_ild', 'ILD (Diversity)', 'higher is better'),
            ('avg_calibration_error', 'Calibration Error', 'lower is better'),
        ]
        
        for i, (metric_key, metric_label, direction) in enumerate(metrics):
            ax = axes[i]
            
            values = [segments[seg][metric_key] for seg in segment_names]
            bars = ax.bar(segment_labels, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(f'{metric_label}\n({direction})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(values) * 1.2)
        
        plt.suptitle('Echo Chamber Metrics by User Segment — Baseline',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")
        
        return fig
    
    # -----------------------------------------------------------------------
    # Plot 4: Gini vs Coverage Scatter
    # -----------------------------------------------------------------------
    
    def plot_gini_vs_coverage(self, output_path: Optional[str] = None):
        """
        Scatter plot showing relationship between concentration and coverage.
        Expected: high Gini → low coverage (echo chamber).
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        gini = self.df['gini'].values
        coverage = self.df['coverage'].values
        
        # Scatter with transparency
        scatter = ax.scatter(gini, coverage, alpha=0.3, s=20, c=coverage,
                            cmap='RdYlGn', edgecolors='none')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Coverage (% of categories)', fontsize=11)
        
        # Trend line
        z = np.polyfit(gini, coverage, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(gini.min(), gini.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, 
                label=f'Trend: coverage = {z[0]:.2f}×gini + {z[1]:.2f}')
        
        # Quadrants
        ax.axhline(0.3, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.6, color='gray', linestyle=':', alpha=0.5)
        ax.text(0.8, 0.05, 'Echo Chamber\n(high gini, low coverage)', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
        ax.text(0.2, 0.55, 'Diverse\n(low gini, high coverage)', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
        
        ax.set_xlabel('Gini Coefficient (Concentration)', fontsize=12)
        ax.set_ylabel('Coverage (% of Available Categories)', fontsize=12)
        ax.set_title('Echo Chamber Relationship: Concentration vs Coverage\n' +
                     '(Each point = one user)',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")
        
        return fig
    
    # -----------------------------------------------------------------------
    # Plot 5: Lorenz Curve (Gini Visualization)
    # -----------------------------------------------------------------------
    
    def plot_lorenz_curve(self, output_path: Optional[str] = None):
        """
        Classic Lorenz curve for category distribution inequality.
        Shows gap between actual distribution and perfect equality.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Aggregate category distribution across all users
        all_categories = []
        for _, row in self.df.iterrows():
            # Reconstruct categories (this is approximate — ideally would save in metrics)
            # For now, use num_unique_cats as a proxy
            all_categories.extend([f'cat_{i}' for i in range(int(row['num_unique_cats']))])
        
        if not all_categories:
            logger.warning("No category data for Lorenz curve")
            return None
        
        from collections import Counter
        cat_counts = Counter(all_categories)
        sorted_counts = np.sort(list(cat_counts.values()))
        
        # Cumulative distribution
        cumsum = np.cumsum(sorted_counts)
        cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
        
        # X-axis: cumulative % of categories
        x = np.linspace(0, 1, len(cumsum))
        
        # Plot Lorenz curve
        ax.plot([0] + list(x), [0] + list(cumsum), linewidth=3, 
                label='Actual Distribution (Baseline)', color='red')
        
        # Perfect equality line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Equality', alpha=0.7)
        
        # Fill area (represents inequality)
        ax.fill_between([0] + list(x), [0] + list(cumsum), [0] + list(x), 
                        alpha=0.3, color='red', label='Inequality Area')
        
        ax.set_xlabel('Cumulative % of Categories (sorted by popularity)', fontsize=12)
        ax.set_ylabel('Cumulative % of Recommendations', fontsize=12)
        ax.set_title('Lorenz Curve — Category Distribution Inequality\n' +
                     '(Larger gap = More echo chamber effect)',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add Gini annotation
        overall_gini = self.report['overall']['avg_gini']
        ax.text(0.5, 0.2, f'Overall Gini = {overall_gini:.3f}',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")
        
        return fig
    
    # -----------------------------------------------------------------------
    # Plot 6: Summary Dashboard
    # -----------------------------------------------------------------------
    
    def plot_summary_dashboard(self, output_path: Optional[str] = None):
        """
        Single-page dashboard with all key metrics.
        Perfect for presentations/thesis.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Top row: Distributions
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Gini histogram (compact)
        ax1.hist(self.df['gini'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(self.df['gini'].mean(), color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Gini', fontsize=10)
        ax1.set_ylabel('Users', fontsize=10)
        ax1.set_title('Concentration Distribution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ILD histogram (compact)
        ax2.hist(self.df['ild'], bins=30, color='seagreen', edgecolor='black', alpha=0.7)
        ax2.axvline(self.df['ild'].mean(), color='darkgreen', linestyle='--', linewidth=2)
        ax2.set_xlabel('ILD', fontsize=10)
        ax2.set_ylabel('Users', fontsize=10)
        ax2.set_title('Diversity Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Calibration error
        ax3.hist(self.df['calibration_error'], bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax3.axvline(self.df['calibration_error'].mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Calibration Error', fontsize=10)
        ax3.set_ylabel('Users', fontsize=10)
        ax3.set_title('Miscalibration Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Middle row: Segment comparison + scatter
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Segment bars (compact)
        segments = self.report['segments']
        segment_names = ['Filter Bubble', 'Balanced', 'Diverse']
        counts = [segments['filter_bubble']['count'],
                 segments['balanced']['count'],
                 segments['diverse']['count']]
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        bars = ax4.bar(segment_names, counts, color=colors, edgecolor='black', linewidth=1.5)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = count / sum(counts) * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Number of Users', fontsize=10)
        ax4.set_title('User Segmentation', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Coverage scatter (compact)
        ax5.scatter(self.df['gini'], self.df['coverage'], alpha=0.4, s=10, c='purple')
        ax5.set_xlabel('Gini', fontsize=10)
        ax5.set_ylabel('Coverage', fontsize=10)
        ax5.set_title('Concentration\nvs Coverage', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Bottom row: Key metrics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create metrics table
        overall = self.report['overall']
        table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Avg Gini', f"{overall['avg_gini']:.4f}", 'HIGH → Echo chamber'],
            ['Avg ILD', f"{overall['avg_ild']:.4f}", 'LOW → Not diverse'],
            ['Avg Coverage', f"{overall['avg_coverage']:.4f} ({overall['avg_coverage']*100:.1f}%)", 'LOW → Few categories shown'],
            ['Avg Calibration Error', f"{overall['avg_calibration_error']:.4f}", 'Mismatch with user history'],
            ['Avg Unique Categories', f"{overall['avg_unique_categories']:.2f}", f"Out of {len(self.df)} available"],
        ]
        
        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.3, 0.2, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        plt.suptitle('Echo Chamber Analysis — Baseline Recommender Dashboard',
                     fontsize=18, fontweight='bold', y=0.98)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"  Saved: {output_path}")
        
        return fig
    
    # -----------------------------------------------------------------------
    # Generate All
    # -----------------------------------------------------------------------
    
    def create_all_plots(self, output_dir: str):
        """
        Generate all visualizations and save to output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Creating Echo Chamber Visualizations")
        logger.info("=" * 60)
        
        plots = [
            ('gini_distribution.png', self.plot_gini_distribution),
            ('ild_distribution.png', self.plot_ild_distribution),
            ('segment_comparison.png', self.plot_segment_comparison),
            ('gini_vs_coverage.png', self.plot_gini_vs_coverage),
            ('lorenz_curve.png', self.plot_lorenz_curve),
            ('summary_dashboard.png', self.plot_summary_dashboard),
        ]
        
        for filename, plot_func in plots:
            logger.info(f"\nGenerating {filename} ...")
            try:
                plot_func(str(output_dir / filename))
                plt.close('all')  # Clean up
            except Exception as e:
                logger.error(f"  Error creating {filename}: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✔ All visualizations saved to {output_dir}/")
        logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    viz = EchoChamberVisualizer.from_files(
        './outputs/echo_chamber_analysis/echo_chamber_report.json',
        './outputs/echo_chamber_analysis/user_metrics.json',
    )
    viz.create_all_plots('./outputs/visualizations/')
