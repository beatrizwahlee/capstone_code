"""
Data Quality and Validation Utilities for MIND Dataset

This module provides functions to validate data quality, check for issues,
and generate diagnostic reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Performs comprehensive data quality checks on MIND dataset."""
    
    def __init__(self, loader, preprocessor):
        """
        Initialize quality checker.
        
        Args:
            loader: MINDDataLoader instance
            preprocessor: MINDPreprocessor instance
        """
        self.loader = loader
        self.preprocessor = preprocessor
        self.issues = []
    
    def run_all_checks(self) -> Dict:
        """
        Run all quality checks and return report.
        
        Returns:
            Dictionary with check results
        """
        logger.info("Running data quality checks...")
        
        results = {
            'missing_data': self.check_missing_data(),
            'duplicates': self.check_duplicates(),
            'entity_coverage': self.check_entity_coverage(),
            'category_balance': self.check_category_balance(),
            'temporal_issues': self.check_temporal_consistency(),
            'interaction_quality': self.check_interaction_quality(),
            'text_quality': self.check_text_quality(),
            'issues': self.issues
        }
        
        logger.info(f"Quality check complete. Found {len(self.issues)} issues.")
        
        return results
    
    def check_missing_data(self) -> Dict:
        """Check for missing or null values."""
        logger.info("Checking for missing data...")
        
        news_missing = {
            col: self.loader.news_df[col].isna().sum()
            for col in self.loader.news_df.columns
        }
        
        behaviors_missing = {
            col: self.loader.behaviors_df[col].isna().sum()
            for col in self.loader.behaviors_df.columns
        }
        
        # Check for empty histories
        empty_histories = (
            self.loader.behaviors_df['history'].apply(len) == 0
        ).sum()
        
        if empty_histories > 0:
            pct = (empty_histories / len(self.loader.behaviors_df)) * 100
            self.issues.append(
                f"Warning: {empty_histories} ({pct:.2f}%) users have empty history"
            )
        
        return {
            'news': news_missing,
            'behaviors': behaviors_missing,
            'empty_histories': empty_histories
        }
    
    def check_duplicates(self) -> Dict:
        """Check for duplicate records."""
        logger.info("Checking for duplicates...")
        
        news_duplicates = self.loader.news_df['news_id'].duplicated().sum()
        impression_duplicates = self.loader.behaviors_df['impression_id'].duplicated().sum()
        
        if news_duplicates > 0:
            self.issues.append(f"Error: {news_duplicates} duplicate news IDs found")
        
        if impression_duplicates > 0:
            self.issues.append(
                f"Error: {impression_duplicates} duplicate impression IDs found"
            )
        
        return {
            'duplicate_news': news_duplicates,
            'duplicate_impressions': impression_duplicates
        }
    
    def check_entity_coverage(self) -> Dict:
        """Check entity embedding coverage."""
        logger.info("Checking entity coverage...")
        
        if not self.loader.entity_embeddings:
            self.issues.append("Warning: No entity embeddings loaded")
            return {'coverage': 0, 'missing_entities': []}
        
        all_entities = set()
        for entities in self.loader.news_df['all_entities']:
            all_entities.update(entities)
        
        covered_entities = sum(
            1 for e in all_entities 
            if e in self.loader.entity_embeddings
        )
        
        coverage = (covered_entities / len(all_entities)) * 100 if all_entities else 0
        
        missing_entities = [
            e for e in all_entities 
            if e not in self.loader.entity_embeddings
        ]
        
        if coverage < 80:
            self.issues.append(
                f"Warning: Low entity embedding coverage ({coverage:.2f}%)"
            )
        
        return {
            'total_unique_entities': len(all_entities),
            'covered_entities': covered_entities,
            'coverage_pct': coverage,
            'missing_count': len(missing_entities),
            'sample_missing': missing_entities[:10]  # First 10 examples
        }
    
    def check_category_balance(self) -> Dict:
        """Check for category imbalance."""
        logger.info("Checking category balance...")
        
        category_counts = self.loader.news_df['category'].value_counts()
        
        # Calculate imbalance ratio (max / min)
        imbalance_ratio = category_counts.max() / category_counts.min()
        
        # Check if severely imbalanced
        if imbalance_ratio > 100:
            self.issues.append(
                f"Warning: Severe category imbalance (ratio: {imbalance_ratio:.2f})"
            )
        
        return {
            'category_counts': category_counts.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': self._calculate_gini(category_counts.values)
        }
    
    def _calculate_gini(self, values) -> float:
        """Calculate Gini coefficient for distribution."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((n - np.arange(1, n + 1) + 1) * sorted_values)) / (n * cumsum[-1]) - 1
    
    def check_temporal_consistency(self) -> Dict:
        """Check for temporal issues."""
        logger.info("Checking temporal consistency...")
        
        time_issues = []
        
        # Check for future dates (relative to dataset collection)
        max_expected_date = pd.Timestamp('2019-12-31')  # Adjust based on dataset
        future_dates = (self.loader.behaviors_df['time'] > max_expected_date).sum()
        
        if future_dates > 0:
            time_issues.append(f"{future_dates} records with unexpected future dates")
        
        # Check for reasonable time range
        time_range = (
            self.loader.behaviors_df['time'].max() - 
            self.loader.behaviors_df['time'].min()
        )
        
        return {
            'time_range_days': time_range.days,
            'future_dates': future_dates,
            'issues': time_issues
        }
    
    def check_interaction_quality(self) -> Dict:
        """Check quality of user interactions."""
        logger.info("Checking interaction quality...")
        
        # Click-through rate
        total_impressions = sum(
            len(row['impressions']) 
            for _, row in self.loader.behaviors_df.iterrows()
        )
        
        total_clicks = sum(
            sum(label for _, label in row['impressions'])
            for _, row in self.loader.behaviors_df.iterrows()
        )
        
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        
        # History length distribution
        history_lengths = self.loader.behaviors_df['history'].apply(len)
        
        # Check for users with very short histories
        short_history_users = (history_lengths < 3).sum()
        
        if short_history_users > len(self.loader.behaviors_df) * 0.3:
            self.issues.append(
                f"Warning: {short_history_users} users have very short histories (<3 items)"
            )
        
        return {
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'overall_ctr': ctr,
            'avg_history_length': history_lengths.mean(),
            'median_history_length': history_lengths.median(),
            'short_history_users': short_history_users
        }
    
    def check_text_quality(self) -> Dict:
        """Check quality of text data."""
        logger.info("Checking text quality...")
        
        # Check for very short titles
        short_titles = (
            self.loader.news_df['title'].str.len() < 10
        ).sum()
        
        # Check for missing abstracts
        missing_abstracts = (
            self.loader.news_df['abstract'].str.len() == 0
        ).sum()
        
        # Check for non-English characters (potential encoding issues)
        # This is a simple check - adjust as needed
        def has_encoding_issues(text):
            if pd.isna(text):
                return False
            try:
                text.encode('utf-8').decode('utf-8')
                return False
            except:
                return True
        
        encoding_issues = self.loader.news_df['title'].apply(
            has_encoding_issues
        ).sum()
        
        if missing_abstracts > len(self.loader.news_df) * 0.5:
            self.issues.append(
                f"Warning: {missing_abstracts} news items missing abstracts"
            )
        
        return {
            'short_titles': short_titles,
            'missing_abstracts': missing_abstracts,
            'potential_encoding_issues': encoding_issues,
            'avg_title_length': self.loader.news_df['title'].str.len().mean(),
            'avg_abstract_length': self.loader.news_df['abstract'].str.len().mean()
        }
    
    def generate_report(self) -> str:
        """Generate human-readable quality report."""
        results = self.run_all_checks()
        
        report = []
        report.append("=" * 80)
        report.append("DATA QUALITY REPORT")
        report.append("=" * 80)
        
        # Missing data
        report.append("\n1. MISSING DATA")
        report.append("-" * 40)
        report.append(f"Empty user histories: {results['missing_data']['empty_histories']}")
        
        # Duplicates
        report.append("\n2. DUPLICATES")
        report.append("-" * 40)
        report.append(f"Duplicate news: {results['duplicates']['duplicate_news']}")
        report.append(f"Duplicate impressions: {results['duplicates']['duplicate_impressions']}")
        
        # Entity coverage
        report.append("\n3. ENTITY COVERAGE")
        report.append("-" * 40)
        report.append(f"Coverage: {results['entity_coverage']['coverage_pct']:.2f}%")
        report.append(f"Missing entities: {results['entity_coverage']['missing_count']}")
        
        # Category balance
        report.append("\n4. CATEGORY BALANCE")
        report.append("-" * 40)
        report.append(f"Imbalance ratio: {results['category_balance']['imbalance_ratio']:.2f}")
        report.append(f"Gini coefficient: {results['category_balance']['gini_coefficient']:.3f}")
        
        # Interaction quality
        report.append("\n5. INTERACTION QUALITY")
        report.append("-" * 40)
        report.append(f"Overall CTR: {results['interaction_quality']['overall_ctr']:.2f}%")
        report.append(f"Avg history length: {results['interaction_quality']['avg_history_length']:.1f}")
        
        # Issues
        report.append("\n6. IDENTIFIED ISSUES")
        report.append("-" * 40)
        if results['issues']:
            for issue in results['issues']:
                report.append(f"- {issue}")
        else:
            report.append("No major issues found!")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def print_dataset_statistics(loader, preprocessor):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 80)
    print("MIND DATASET STATISTICS")
    print("=" * 80)
    
    print(f"\nDataset Type: {loader.dataset_type}")
    print(f"Data Directory: {loader.data_dir}")
    
    # Behaviors stats
    print("\n--- BEHAVIORS ---")
    print(f"Total impressions: {len(loader.behaviors_df):,}")
    print(f"Unique users: {loader.behaviors_df['user_id'].nunique():,}")
    print(f"Date range: {loader.behaviors_df['time'].min()} to {loader.behaviors_df['time'].max()}")
    
    # News stats
    print("\n--- NEWS ---")
    print(f"Total articles: {len(loader.news_df):,}")
    print(f"Categories: {loader.news_df['category'].nunique()}")
    print(f"Subcategories: {loader.news_df['subcategory'].nunique()}")
    
    print("\nTop 10 Categories:")
    for cat, count in loader.news_df['category'].value_counts().head(10).items():
        pct = (count / len(loader.news_df)) * 100
        print(f"  {cat:20s}: {count:6,} ({pct:5.2f}%)")
    
    # Entity stats
    print("\n--- ENTITIES ---")
    print(f"Entity embeddings: {len(loader.entity_embeddings):,}")
    print(f"Relation embeddings: {len(loader.relation_embeddings):,}")
    
    entity_counts = loader.news_df['all_entities'].apply(len)
    print(f"Avg entities per article: {entity_counts.mean():.2f}")
    print(f"Articles with entities: {(entity_counts > 0).sum():,}")
    
    # Interaction stats
    print("\n--- INTERACTIONS ---")
    total_clicks = sum(
        sum(label for _, label in row['impressions'])
        for _, row in loader.behaviors_df.iterrows()
    )
    total_impr = sum(
        len(row['impressions'])
        for _, row in loader.behaviors_df.iterrows()
    )
    
    print(f"Total clicks: {total_clicks:,}")
    print(f"Total impressions: {total_impr:,}")
    print(f"Overall CTR: {(total_clicks/total_impr*100):.2f}%")
    
    history_lengths = loader.behaviors_df['history'].apply(len)
    print(f"Avg history length: {history_lengths.mean():.2f}")
    print(f"Median history length: {history_lengths.median():.0f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    from mind_data_loader import MINDDataLoader, MINDPreprocessor
    
    # Example usage
    data_dir = "./MINDlarge_train"
    
    loader = MINDDataLoader(data_dir, dataset_type='train')
    loader.load_all_data()
    
    preprocessor = MINDPreprocessor(loader)
    preprocessor.preprocess_all()
    
    # Run quality checks
    checker = DataQualityChecker(loader, preprocessor)
    print(checker.generate_report())
    
    # Print statistics
    print_dataset_statistics(loader, preprocessor)
