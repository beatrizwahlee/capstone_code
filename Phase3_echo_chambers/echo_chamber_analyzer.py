"""
Phase 3: Echo Chamber Detection & Measurement
==============================================
Diversity-Aware News Recommender System — Capstone Project

This phase quantifies the echo chamber problem in the baseline recommender.

Five Echo Chamber Metrics:
---------------------------
1. Gini Coefficient (category concentration)
   - 0.0 = perfectly distributed across all categories
   - 1.0 = everything from one category (pure echo chamber)
   
2. ILD (Intra-List Diversity) — semantic diversity
   - Average pairwise cosine dissimilarity in recommendations
   - Uses SBERT embeddings from Phase 1
   - 0.0 = all items identical, 1.0 = maximally diverse
   
3. Catalog Coverage
   - % of available categories appearing in recommendations
   - Aggregated across all users
   
4. Calibration Error (KL divergence)
   - Distance between recommended category distribution
     and user's historical category distribution
   - 0.0 = perfect match, higher = more miscalibration
   
5. Category Entropy
   - Shannon entropy of category distribution per user
   - Low entropy = concentrated (echo chamber)
   - High entropy = distributed (diverse)

User Segmentation:
------------------
- Filter Bubble Users: Gini > 0.8 (concentrated interests)
- Balanced Users: 0.4 < Gini < 0.6
- Diverse Users: Gini < 0.4

Output:
-------
- echo_chamber_report.json — all metrics
- user_segments.json — per-segment analysis
- visualizations/ — plots for thesis
  - gini_distribution.png
  - category_concentration.png
  - diversity_by_segment.png
  - calibration_error_histogram.png

Usage:
    from echo_chamber_analyzer import EchoChamberAnalyzer
    
    analyzer = EchoChamberAnalyzer.from_baseline('./outputs/baseline')
    report = analyzer.analyze(test_data, k=10)
    analyzer.save_report('./outputs/echo_chamber_analysis')
"""

import json
import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Echo Chamber Analyzer
# ---------------------------------------------------------------------------

class EchoChamberAnalyzer:
    """
    Comprehensive echo chamber detection and measurement.
    """
    
    def __init__(
        self,
        recommender,
        news_df: pd.DataFrame,
        embeddings: np.ndarray,
        news_id_to_idx: Dict[str, int],
    ):
        """
        Args:
            recommender:     Trained BaselineRecommender from Phase 2
            news_df:         News metadata with category info
            embeddings:      Final 532-dim embeddings from Phase 1
            news_id_to_idx:  news_id → row index mapping
        """
        self.recommender = recommender
        self.news_df = news_df
        self.embeddings = embeddings
        self.news_id_to_idx = news_id_to_idx
        
        # Build lookup dicts
        self.news_to_category = dict(zip(news_df['news_id'], news_df['category']))
        self.all_categories = set(news_df['category'].unique())
    
    @classmethod
    def from_baseline(cls, baseline_dir: str, embeddings_dir: str = None):
        """
        Load from Phase 2 baseline outputs.
        
        Args:
            baseline_dir:    Path to outputs/baseline/ directory
            embeddings_dir:  Path to embeddings/ (defaults to ../embeddings)
        """
        baseline_dir = Path(baseline_dir)
        
        if embeddings_dir is None:
            embeddings_dir = baseline_dir.parent.parent / "embeddings"
        
        # Load recommender
        from baseline_recommender_phase2 import BaselineRecommender
        recommender = BaselineRecommender.load(
            str(baseline_dir / "baseline_recommender.pkl"),
            str(embeddings_dir),
        )
        
        # Load news metadata
        news_df = recommender.news_metadata
        if news_df is None:
            # Try to load from processed data
            processed_dir = baseline_dir.parent.parent.parent / "Phase0_data_processing" / "processed_data"
            news_path = processed_dir / "news_features_train.csv"
            if news_path.exists():
                news_df = pd.read_csv(news_path)
            else:
                raise ValueError("News metadata not found. Re-run Phase 2 training.")
        
        return cls(
            recommender=recommender,
            news_df=news_df,
            embeddings=recommender.final_embeddings,
            news_id_to_idx=recommender.news_id_to_idx,
        )
    
    # -----------------------------------------------------------------------
    # Core Metrics
    # -----------------------------------------------------------------------
    
    def calculate_gini(self, categories: List[str]) -> float:
        """
        Gini coefficient for category distribution.
        
        0.0 = perfectly equal, 1.0 = maximum inequality (echo chamber)
        """
        if not categories:
            return 0.0
        
        counts = Counter(categories)
        values = np.array(list(counts.values()))
        
        if len(values) == 1:
            return 1.0  # All same category
        
        sorted_vals = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
    
    def calculate_ild(self, news_ids: List[str]) -> float:
        """
        Intra-List Diversity: average pairwise cosine dissimilarity.
        
        Uses semantic embeddings from Phase 1.
        0.0 = all identical, 1.0 = maximally diverse
        """
        # Filter to known IDs
        known_ids = [nid for nid in news_ids if nid in self.news_id_to_idx]
        
        if len(known_ids) < 2:
            return 0.0
        
        # Get embeddings
        embs = np.array([
            self.embeddings[self.news_id_to_idx[nid]]
            for nid in known_ids
        ])
        
        # Pairwise cosine dissimilarity
        n = len(embs)
        total_dissim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                dissim = 1.0 - float(embs[i] @ embs[j])  # Already L2-normalized
                total_dissim += dissim
                count += 1
        
        return total_dissim / count if count > 0 else 0.0
    
    def calculate_coverage(self, categories: List[str]) -> float:
        """
        Catalog coverage: % of available categories represented.
        """
        if not categories:
            return 0.0
        
        unique_cats = set(categories)
        return len(unique_cats) / len(self.all_categories)
    
    def calculate_calibration_error(
        self,
        recommended_cats: List[str],
        history_cats: List[str],
    ) -> float:
        """
        KL divergence between recommended and historical distributions.
        
        Measures how well recommendations match user's past interests.
        0.0 = perfect calibration, higher = more error
        """
        if not recommended_cats or not history_cats:
            return 0.0
        
        # Build distributions
        all_cats = sorted(self.all_categories)
        
        def build_dist(cats):
            counts = Counter(cats)
            total = sum(counts.values())
            # Smoothing to avoid zero probabilities
            return np.array([
                (counts.get(c, 0) + 0.01) / (total + 0.01 * len(all_cats))
                for c in all_cats
            ])
        
        p_history = build_dist(history_cats)
        p_recommended = build_dist(recommended_cats)
        
        # KL divergence: D_KL(P || Q) = sum(P * log(P/Q))
        return float(entropy(p_history, p_recommended))
    
    def calculate_entropy(self, categories: List[str]) -> float:
        """
        Shannon entropy of category distribution.
        
        Low entropy = concentrated (echo chamber)
        High entropy = distributed (diverse)
        """
        if not categories:
            return 0.0
        
        counts = Counter(categories)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        
        return float(entropy(probs))
    
    # -----------------------------------------------------------------------
    # Analysis Pipeline
    # -----------------------------------------------------------------------
    
    def analyze(
        self,
        test_data: List[Dict],
        k: int = 10,
        max_users: Optional[int] = None,
    ) -> Dict:
        """
        Run full echo chamber analysis.
        
        Args:
            test_data:   List of test impressions (from Phase 2)
            k:           Number of recommendations to generate per user
            max_users:   Limit analysis to first N users (for speed)
        
        Returns:
            Comprehensive analysis report
        """
        logger.info("=" * 60)
        logger.info("Phase 3 — Echo Chamber Analysis")
        logger.info("=" * 60)
        logger.info(f"Analyzing {len(test_data):,} users (k={k}) ...")
        
        if max_users:
            test_data = test_data[:max_users]
            logger.info(f"  Limited to {max_users:,} users for speed")
        
        # Collect metrics per user
        user_metrics = []
        
        for i, sample in enumerate(test_data):
            if i % 5000 == 0 and i > 0:
                logger.info(f"  Processed {i:,}/{len(test_data):,} users ...")
            
            history = sample['history']
            
            if not history:
                continue  # Skip cold-start users
            
            # Generate recommendations
            recs = self.recommender.recommend(history, k=k)
            
            if not recs:
                continue
            
            rec_ids = [nid for nid, _ in recs]
            
            # Get categories
            rec_cats = [
                self.news_to_category[nid]
                for nid in rec_ids
                if nid in self.news_to_category
            ]
            
            history_cats = [
                self.news_to_category[nid]
                for nid in history
                if nid in self.news_to_category
            ]
            
            if not rec_cats or not history_cats:
                continue
            
            # Calculate all metrics
            metrics = {
                'user_id': sample.get('user_id', f'user_{i}'),
                'gini': self.calculate_gini(rec_cats),
                'ild': self.calculate_ild(rec_ids),
                'coverage': self.calculate_coverage(rec_cats),
                'calibration_error': self.calculate_calibration_error(rec_cats, history_cats),
                'entropy': self.calculate_entropy(rec_cats),
                'num_unique_cats': len(set(rec_cats)),
                'history_length': len(history),
                'history_diversity': self.calculate_gini(history_cats),
            }
            
            user_metrics.append(metrics)
        
        logger.info(f"  Analysis complete for {len(user_metrics):,} users")
        
        # Aggregate results
        report = self._aggregate_metrics(user_metrics)
        report['num_users_analyzed'] = len(user_metrics)
        report['k'] = k
        
        return report
    
    def _aggregate_metrics(self, user_metrics: List[Dict]) -> Dict:
        """Aggregate per-user metrics into summary statistics."""
        df = pd.DataFrame(user_metrics)
        
        # Overall averages
        overall = {
            'avg_gini': float(df['gini'].mean()),
            'median_gini': float(df['gini'].median()),
            'avg_ild': float(df['ild'].mean()),
            'median_ild': float(df['ild'].median()),
            'avg_coverage': float(df['coverage'].mean()),
            'avg_calibration_error': float(df['calibration_error'].mean()),
            'avg_entropy': float(df['entropy'].mean()),
            'avg_unique_categories': float(df['num_unique_cats'].mean()),
        }
        
        # User segmentation
        filter_bubble = df[df['gini'] > 0.8]
        balanced = df[(df['gini'] >= 0.4) & (df['gini'] <= 0.6)]
        diverse = df[df['gini'] < 0.4]
        
        segments = {
            'filter_bubble': {
                'count': len(filter_bubble),
                'pct': len(filter_bubble) / len(df) * 100,
                'avg_gini': float(filter_bubble['gini'].mean()) if len(filter_bubble) > 0 else 0,
                'avg_ild': float(filter_bubble['ild'].mean()) if len(filter_bubble) > 0 else 0,
                'avg_calibration_error': float(filter_bubble['calibration_error'].mean()) if len(filter_bubble) > 0 else 0,
            },
            'balanced': {
                'count': len(balanced),
                'pct': len(balanced) / len(df) * 100,
                'avg_gini': float(balanced['gini'].mean()) if len(balanced) > 0 else 0,
                'avg_ild': float(balanced['ild'].mean()) if len(balanced) > 0 else 0,
                'avg_calibration_error': float(balanced['calibration_error'].mean()) if len(balanced) > 0 else 0,
            },
            'diverse': {
                'count': len(diverse),
                'pct': len(diverse) / len(df) * 100,
                'avg_gini': float(diverse['gini'].mean()) if len(diverse) > 0 else 0,
                'avg_ild': float(diverse['ild'].mean()) if len(diverse) > 0 else 0,
                'avg_calibration_error': float(diverse['calibration_error'].mean()) if len(diverse) > 0 else 0,
            },
        }
        
        # Distribution stats
        distributions = {
            'gini_percentiles': {
                '25th': float(df['gini'].quantile(0.25)),
                '50th': float(df['gini'].quantile(0.50)),
                '75th': float(df['gini'].quantile(0.75)),
                '90th': float(df['gini'].quantile(0.90)),
            },
            'ild_percentiles': {
                '25th': float(df['ild'].quantile(0.25)),
                '50th': float(df['ild'].quantile(0.50)),
                '75th': float(df['ild'].quantile(0.75)),
                '90th': float(df['ild'].quantile(0.90)),
            },
        }
        
        return {
            'overall': overall,
            'segments': segments,
            'distributions': distributions,
            'raw_metrics': user_metrics,  # Keep for visualization
        }
    
    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------
    
    def print_report(self, report: Dict):
        """Pretty-print the echo chamber analysis report."""
        print("\n" + "=" * 60)
        print("ECHO CHAMBER ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\nUsers analyzed: {report['num_users_analyzed']:,}  (k={report['k']})")
        
        overall = report['overall']
        print("\n--- Overall Echo Chamber Metrics ---")
        print(f"  Gini Coefficient:       {overall['avg_gini']:.4f}  (median: {overall['median_gini']:.4f})")
        print(f"  ILD (diversity):        {overall['avg_ild']:.4f}  (median: {overall['median_ild']:.4f})")
        print(f"  Coverage:               {overall['avg_coverage']:.4f}  ({overall['avg_coverage']*100:.1f}% of categories)")
        print(f"  Calibration Error:      {overall['avg_calibration_error']:.4f}")
        print(f"  Entropy:                {overall['avg_entropy']:.4f}")
        print(f"  Avg Unique Categories:  {overall['avg_unique_categories']:.2f}")
        
        print("\n--- User Segmentation ---")
        for segment_name, segment_data in report['segments'].items():
            print(f"\n{segment_name.replace('_', ' ').title()}:")
            print(f"  Count:              {segment_data['count']:,}  ({segment_data['pct']:.1f}%)")
            print(f"  Avg Gini:           {segment_data['avg_gini']:.4f}")
            print(f"  Avg ILD:            {segment_data['avg_ild']:.4f}")
            print(f"  Avg Calib Error:    {segment_data['avg_calibration_error']:.4f}")
        
        print("\n--- Distribution Percentiles ---")
        print("Gini:")
        for pct, val in report['distributions']['gini_percentiles'].items():
            print(f"  {pct}: {val:.4f}")
        
        print("ILD:")
        for pct, val in report['distributions']['ild_percentiles'].items():
            print(f"  {pct}: {val:.4f}")
        
        print("\n" + "=" * 60)
    
    def save_report(self, output_dir: str, report: Dict):
        """Save analysis report and visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report (without raw_metrics to keep file size reasonable)
        report_clean = {k: v for k, v in report.items() if k != 'raw_metrics'}
        with open(output_dir / "echo_chamber_report.json", 'w') as f:
            json.dump(report_clean, f, indent=2)
        
        logger.info(f"✔ Report saved to {output_dir}/echo_chamber_report.json")
        
        # Save raw metrics for further analysis
        with open(output_dir / "user_metrics.json", 'w') as f:
            json.dump(report['raw_metrics'], f, indent=2)
        
        logger.info(f"✔ User metrics saved to {output_dir}/user_metrics.json")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def quick_analyze(baseline_dir: str = "./outputs/baseline", max_users: int = 10000):
    """Quick analysis for testing."""
    logger.info("Running quick echo chamber analysis ...")
    
    analyzer = EchoChamberAnalyzer.from_baseline(baseline_dir)
    
    # Load test data (would normally come from Phase 2)
    # For now, create mock data from baseline
    test_data = []
    for i in range(min(max_users, len(analyzer.recommender.news_ids))):
        history = analyzer.recommender.news_ids[i:i+10]
        test_data.append({'user_id': f'user_{i}', 'history': history})
    
    report = analyzer.analyze(test_data, k=10)
    analyzer.print_report(report)
    
    return report


if __name__ == "__main__":
    report = quick_analyze()
