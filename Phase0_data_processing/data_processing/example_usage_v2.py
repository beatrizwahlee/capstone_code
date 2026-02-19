"""
UPDATED Example Usage Script for MIND Dataset Loading and Preprocessing
WITH DATA QUALITY FIXES INTEGRATED

This script demonstrates how to use the MINDDataLoader and MINDPreprocessor
classes with integrated fixes for data quality issues identified in the report:
- Handles empty user histories (cold start problem)
- Creates popularity baseline for fallback
- Filters problematic samples

Run this script to get started with your capstone project!
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC_DIR = Path(__file__).resolve().parent
for _p in [str(DATA_PROC_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mind_data_loader import MINDDataLoader, MINDPreprocessor, get_data_summary
from data_quality_checker import DataQualityChecker, print_dataset_statistics


def filter_empty_histories(behaviors_df: pd.DataFrame, min_history_length: int = 1):
    """
    Filter out users with insufficient click history.
    
    This addresses the cold start problem where users have no prior interactions
    to build a content-based profile from.
    
    Args:
        behaviors_df: Behaviors DataFrame
        min_history_length: Minimum required history length
    
    Returns:
        Filtered DataFrame
    """
    print(f"\n[DATA QUALITY FIX] Filtering users with history < {min_history_length}...")
    
    before_count = len(behaviors_df)
    
    # Filter based on history length
    behaviors_df['_history_length'] = behaviors_df['history'].apply(len)
    filtered_df = behaviors_df[behaviors_df['_history_length'] >= min_history_length].copy()
    filtered_df = filtered_df.drop('_history_length', axis=1)
    
    after_count = len(filtered_df)
    removed = before_count - after_count
    removed_pct = (removed / before_count) * 100
    
    print(f"  ✓ Removed {removed:,} impressions ({removed_pct:.2f}%) with empty/short histories")
    print(f"  ✓ Remaining impressions: {after_count:,}")
    
    return filtered_df


def create_popularity_scores(behaviors_df: pd.DataFrame, news_df: pd.DataFrame):
    """
    Create popularity scores for all news articles.
    
    This provides a fallback recommendation strategy for cold start users
    who have no history to build a content-based profile from.
    
    Args:
        behaviors_df: Behaviors DataFrame with user click histories
        news_df: News DataFrame
    
    Returns:
        Dictionary mapping news_id to popularity score [0, 1]
    """
    print("\n[DATA QUALITY FIX] Creating popularity baseline for cold start...")
    
    click_counts = defaultdict(int)
    
    # Count clicks from user histories
    for _, row in behaviors_df.iterrows():
        for news_id in row['history']:
            click_counts[news_id] += 1
    
    # Also count from impressions with positive labels
    for _, row in behaviors_df.iterrows():
        for news_id, label in row['impressions']:
            if label == 1:
                click_counts[news_id] += 1
    
    # Normalize to [0, 1] range
    if click_counts:
        max_count = max(click_counts.values())
        popularity_scores = {
            news_id: count / max_count 
            for news_id, count in click_counts.items()
        }
    else:
        popularity_scores = {}
    
    # Add zero scores for news with no clicks
    for news_id in news_df['news_id']:
        if news_id not in popularity_scores:
            popularity_scores[news_id] = 0.0
    
    print(f"  ✓ Created popularity scores for {len(popularity_scores):,} articles")
    print(f"  ✓ Articles with clicks: {len(click_counts):,}")
    print(f"  ✓ Max clicks per article: {max(click_counts.values()) if click_counts else 0}")
    
    return popularity_scores


def filter_no_positive_impressions(behaviors_df: pd.DataFrame):
    """
    Remove impressions where user didn't click on anything.
    
    These impressions can't be used for evaluation (no ground truth).
    
    Args:
        behaviors_df: Behaviors DataFrame
    
    Returns:
        Filtered DataFrame
    """
    print("\n[DATA QUALITY FIX] Filtering impressions with no clicks...")
    
    before_count = len(behaviors_df)
    
    # Check if any impression has a positive label
    def has_positive_label(impressions):
        return any(label == 1 for _, label in impressions)
    
    filtered_df = behaviors_df[
        behaviors_df['impressions'].apply(has_positive_label)
    ].copy()
    
    after_count = len(filtered_df)
    removed = before_count - after_count
    removed_pct = (removed / before_count) * 100
    
    print(f"  ✓ Removed {removed:,} impressions ({removed_pct:.2f}%) with no clicks")
    print(f"  ✓ Remaining impressions: {after_count:,}")
    
    return filtered_df


def augment_with_quality_fixes(preprocessor: MINDPreprocessor, 
                                popularity_scores: dict):
    """
    Add data quality fixes to preprocessed data.
    
    Args:
        preprocessor: MINDPreprocessor instance
        popularity_scores: Dictionary of popularity scores
    
    Returns:
        Updated processed data dictionary
    """
    print("\nAugmenting processed data with quality fixes...")
    
    # Get the processed data
    processed_data = preprocessor.preprocess_all()
    
    # Add popularity scores
    processed_data['popularity_scores'] = popularity_scores
    
    # Add metadata about fixes applied
    processed_data['quality_fixes_applied'] = {
        'empty_histories_filtered': True,
        'no_click_impressions_filtered': True,
        'popularity_baseline_created': True,
        'min_history_length': 1
    }
    
    print("  ✓ Added popularity scores to processed data")
    print("  ✓ Added quality fix metadata")
    
    return processed_data


def main():
    """Main execution function with integrated data quality fixes."""
    
    print("\n" + "=" * 80)
    print("MIND DATASET LOADING AND PREPROCESSING")
    print("WITH INTEGRATED DATA QUALITY FIXES")
    print("Diversity-Aware News Recommender System - Capstone Project")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: Configure paths
    # =========================================================================
    print("\n[STEP 1] Configuring data paths...")
    
    # UPDATE THESE PATHS to match your data location
    TRAIN_DATA_DIR = "./MINDlarge_train"  # or "./MINDsmall_train"
    VALID_DATA_DIR = "./MINDlarge_dev"    # or "./MINDsmall_dev"
    
    # Check if using zip files
    TRAIN_ZIP = None  # Set to "./MINDlarge_train.zip" if using zip
    VALID_ZIP = None  # Set to "./MINDlarge_dev.zip" if using zip
    
    print(f"Train data: {TRAIN_DATA_DIR}")
    print(f"Valid data: {VALID_DATA_DIR}")
    
    # =========================================================================
    # STEP 2: Load training data
    # =========================================================================
    print("\n[STEP 2] Loading training data...")
    
    train_loader = MINDDataLoader(TRAIN_DATA_DIR, dataset_type='train')
    
    # If you have zip files, extract them first
    if TRAIN_ZIP:
        train_loader.load_all_data(extract_zip=True, zip_path=TRAIN_ZIP)
    else:
        train_loader.load_all_data()
    
    print("✓ Training data loaded successfully!")
    
    # =========================================================================
    # STEP 3: Load validation data
    # =========================================================================
    print("\n[STEP 3] Loading validation data...")
    
    valid_loader = MINDDataLoader(VALID_DATA_DIR, dataset_type='valid')
    
    if VALID_ZIP:
        valid_loader.load_all_data(extract_zip=True, zip_path=VALID_ZIP)
    else:
        valid_loader.load_all_data()
    
    print("✓ Validation data loaded successfully!")
    
    # =========================================================================
    # STEP 4: Data quality checks (BEFORE fixes)
    # =========================================================================
    print("\n[STEP 4] Running data quality checks (pre-fix)...")
    
    train_preprocessor = MINDPreprocessor(train_loader)
    
    checker = DataQualityChecker(train_loader, train_preprocessor)
    quality_report = checker.generate_report()
    
    print(quality_report)
    
    # Save quality report
    with open('data_quality_report_prefixes.txt', 'w') as f:
        f.write(quality_report)
    print("✓ Pre-fix quality report saved to 'data_quality_report_prefixes.txt'")
    
    # =========================================================================
    # STEP 5: Apply Data Quality Fixes (NEW!)
    # =========================================================================
    print("\n[STEP 5] Applying data quality fixes...")
    
    print("\n" + "=" * 80)
    print("DATA QUALITY FIXES")
    print("=" * 80)
    
    # Fix 1: Filter empty histories
    train_loader.behaviors_df = filter_empty_histories(
        train_loader.behaviors_df, 
        min_history_length=1
    )
    
    # Fix 2: Filter impressions with no clicks
    train_loader.behaviors_df = filter_no_positive_impressions(
        train_loader.behaviors_df
    )
    
    # Fix 3: Create popularity baseline
    popularity_scores = create_popularity_scores(
        train_loader.behaviors_df,
        train_loader.news_df
    )
    
    print("\n" + "=" * 80)
    print("✓ All data quality fixes applied successfully!")
    print("=" * 80)
    
    # =========================================================================
    # STEP 6: Data quality checks (AFTER fixes)
    # =========================================================================
    print("\n[STEP 6] Running data quality checks (post-fix)...")
    
    # Re-create preprocessor with fixed data
    train_preprocessor = MINDPreprocessor(train_loader)
    
    checker_post = DataQualityChecker(train_loader, train_preprocessor)
    quality_report_post = checker_post.generate_report()
    
    print(quality_report_post)
    
    # Save post-fix quality report
    with open('data_quality_report_postfixes.txt', 'w') as f:
        f.write(quality_report_post)
    print("✓ Post-fix quality report saved to 'data_quality_report_postfixes.txt'")
    
    # =========================================================================
    # STEP 7: Preprocess training data (with fixes applied)
    # =========================================================================
    print("\n[STEP 7] Preprocessing training data (with fixes)...")
    
    train_processed = augment_with_quality_fixes(
        train_preprocessor,
        popularity_scores
    )
    
    print("✓ Training data preprocessed!")
    print(f"  - News features: {len(train_processed['news_features'])} articles")
    print(f"  - Training samples: {len(train_processed['train_data'])}")
    print(f"  - Validation samples: {len(train_processed['val_data'])}")
    print(f"  - Popularity scores: {len(train_processed['popularity_scores'])} articles")
    
    # =========================================================================
    # STEP 8: Apply same fixes to validation data
    # =========================================================================
    print("\n[STEP 8] Applying fixes to validation data...")
    
    # Apply same filters
    valid_loader.behaviors_df = filter_empty_histories(
        valid_loader.behaviors_df,
        min_history_length=1
    )
    valid_loader.behaviors_df = filter_no_positive_impressions(
        valid_loader.behaviors_df
    )
    
    valid_preprocessor = MINDPreprocessor(valid_loader)
    valid_processed = valid_preprocessor.preprocess_all()
    
    # Use same popularity scores from training
    valid_processed['popularity_scores'] = popularity_scores
    valid_processed['quality_fixes_applied'] = train_processed['quality_fixes_applied']
    
    print("✓ Validation data preprocessed!")
    print(f"  - Test samples: {len(valid_processed['val_data'])}")
    
    # =========================================================================
    # STEP 9: Generate comprehensive statistics
    # =========================================================================
    print("\n[STEP 9] Generating dataset statistics...")
    
    print_dataset_statistics(train_loader, train_preprocessor)
    
    # =========================================================================
    # STEP 10: Save processed data
    # =========================================================================
    print("\n[STEP 10] Saving processed data...")
    
    # Create output directory (write alongside Phase0 folder, not relative to cwd)
    output_dir = Path(__file__).resolve().parents[1] / "processed_data"
    output_dir.mkdir(exist_ok=True)
    
    # Save news features
    train_processed['news_features'].to_csv(
        output_dir / 'news_features_train.csv', 
        index=False
    )
    print(f"✓ Saved news features: {output_dir / 'news_features_train.csv'}")
    
    # Save diversity statistics
    with open(output_dir / 'diversity_stats.json', 'w') as f:
        diversity_stats = train_processed['diversity_stats'].copy()
        # Convert numpy values to Python types
        for key in diversity_stats:
            if isinstance(diversity_stats[key], dict):
                diversity_stats[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in diversity_stats[key].items()
                }
        json.dump(diversity_stats, f, indent=2)
    print(f"✓ Saved diversity stats: {output_dir / 'diversity_stats.json'}")
    
    # Save encoders
    encoders = train_processed['encoders']
    with open(output_dir / 'encoders.json', 'w') as f:
        json.dump({
            'category': encoders['category'],
            'subcategory': encoders['subcategory']
        }, f, indent=2)
    print(f"✓ Saved encoders: {output_dir / 'encoders.json'}")
    
    # Save popularity scores (NEW!)
    with open(output_dir / 'popularity_scores.json', 'w') as f:
        json.dump(popularity_scores, f, indent=2)
    print(f"✓ Saved popularity scores: {output_dir / 'popularity_scores.json'}")
    
    # Save quality fixes metadata (NEW!)
    with open(output_dir / 'quality_fixes_applied.json', 'w') as f:
        json.dump(train_processed['quality_fixes_applied'], f, indent=2)
    print(f"✓ Saved quality fixes metadata: {output_dir / 'quality_fixes_applied.json'}")
    
    # Save interaction data (sample for inspection)
    sample_train = pd.DataFrame(train_processed['train_data'][:1000])
    sample_train.to_csv(output_dir / 'sample_train_interactions.csv', index=False)
    print(f"✓ Saved sample interactions: {output_dir / 'sample_train_interactions.csv'}")
    
    # =========================================================================
    # STEP 11: Generate data summary for your project report
    # =========================================================================
    print("\n[STEP 11] Generating comprehensive summary...")
    
    summary = get_data_summary(train_loader, train_preprocessor)
    
    # Add quality fix information
    summary['quality_fixes'] = {
        'empty_histories_removed': True,
        'no_click_impressions_removed': True,
        'popularity_baseline_created': True,
        'total_popularity_scores': len(popularity_scores)
    }
    
    with open(output_dir / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("SUMMARY (POST-FIXES)")
    print("=" * 80)
    print(f"Total users: {summary['unique_users']:,}")
    print(f"Total news articles: {summary['total_news']:,}")
    print(f"Total impressions: {summary['total_impressions']:,}")
    print(f"Categories: {summary['categories']}")
    print(f"Subcategories: {summary['subcategories']}")
    print(f"Entity embeddings: {summary['num_entity_embeddings']:,}")
    print(f"Time range: {summary['time_range']['duration_days']} days")
    print("\nQuality Fixes Applied:")
    print(f"  ✓ Empty histories filtered")
    print(f"  ✓ No-click impressions filtered")
    print(f"  ✓ Popularity baseline created ({len(popularity_scores):,} scores)")
    print("=" * 80)
    
    # =========================================================================
    # STEP 12: Create comparison report
    # =========================================================================
    print("\n[STEP 12] Creating before/after comparison...")
    
    comparison = {
        'before_fixes': {
            'file': 'data_quality_report_prefixes.txt',
            'empty_histories': '46,065 (2.06%)',
            'category_imbalance_ratio': '32,020:1'
        },
        'after_fixes': {
            'file': 'data_quality_report_postfixes.txt',
            'empty_histories': '0 (filtered out)',
            'category_imbalance_ratio': '32,020:1 (expected, validates diversity need)'
        },
        'actions_taken': [
            'Filtered users with empty click histories (cold start)',
            'Removed impressions with no positive labels',
            'Created popularity-based recommendation fallback',
            'Preserved category imbalance (shows need for diversity)'
        ]
    }
    
    with open(output_dir / 'quality_fixes_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("\n" + "=" * 80)
    print("BEFORE vs AFTER DATA QUALITY FIXES")
    print("=" * 80)
    print("\nBEFORE:")
    print(f"  • Empty histories: 46,065 impressions (2.06%)")
    print(f"  • Category imbalance: 32,020:1 ratio")
    print(f"  • Entity coverage: 96.71%")
    
    print("\nAFTER:")
    print(f"  • Empty histories: FILTERED OUT")
    print(f"  • No-click impressions: FILTERED OUT")
    print(f"  • Popularity baseline: CREATED")
    print(f"  • Category imbalance: PRESERVED (intentional)")
    
    print("\nWHY PRESERVE CATEGORY IMBALANCE?")
    print("  → Shows real-world news distribution")
    print("  → Demonstrates echo chamber problem")
    print("  → Motivates diversity-aware re-ranking (Phase 2)")
    print("  → Baseline will show this bias, diversity methods will fix it")
    print("=" * 80)
    
    # =========================================================================
    # STEP 13: What's next?
    # =========================================================================
    print("\n[STEP 13] Next steps for your project:")
    print("-" * 80)
    print("1. ✓ Data loading and preprocessing complete!")
    print("2. ✓ Data quality issues identified and fixed!")
    print("3. → Text encoding: Implement NLP embeddings for news content")
    print("   - Consider using BERT, Sentence-BERT, or News-specific models")
    print("4. → Baseline training: Use the ORIGINAL training script")
    print("   - NO CHANGES NEEDED to train_baseline_recommender.py")
    print("   - It will automatically use the cleaned data from the processed_data/ directory")
    print("5. → Diversity re-ranking: Implement multi-objective ranking")
    print("   - MMR, xQuAD, Calibration, Fairness algorithms")
    print("6. → Evaluation: Compare baseline vs diversity-aware results")
    print("-" * 80)
    
    print(f"\n✓ All preprocessing complete! Clean data saved to '{output_dir}'")
    print("\nIMPORTANT: When training, load data from the 'processed_data/' directory")
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'train_processed': train_processed,
        'valid_processed': valid_processed
    }


def quick_data_exploration(train_loader):
    """
    Quick exploration of the loaded data.
    Useful for understanding the data structure.
    """
    print("\n" + "=" * 80)
    print("QUICK DATA EXPLORATION")
    print("=" * 80)
    
    # Sample behavior
    print("\n--- Sample Behavior Record ---")
    sample_behavior = train_loader.behaviors_df.iloc[0]
    print(f"Impression ID: {sample_behavior['impression_id']}")
    print(f"User ID: {sample_behavior['user_id']}")
    print(f"Time: {sample_behavior['time']}")
    print(f"History length: {len(sample_behavior['history'])}")
    print(f"First 5 history items: {sample_behavior['history'][:5]}")
    print(f"Number of impressions: {len(sample_behavior['impressions'])}")
    print(f"First 3 impressions: {sample_behavior['impressions'][:3]}")
    
    # Sample news
    print("\n--- Sample News Article ---")
    sample_news = train_loader.news_df.iloc[0]
    print(f"News ID: {sample_news['news_id']}")
    print(f"Category: {sample_news['category']}")
    print(f"Subcategory: {sample_news['subcategory']}")
    print(f"Title: {sample_news['title']}")
    print(f"Abstract: {sample_news['abstract'][:100]}...")
    print(f"Number of entities: {len(sample_news['all_entities'])}")
    if sample_news['all_entities']:
        print(f"First entity: {sample_news['all_entities'][0]}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the main pipeline
    try:
        results = main()
        
        # Optional: Run quick exploration
        print("\n" + "=" * 80)
        response = input("Would you like to see a quick data exploration? (y/n): ")
        if response.lower() == 'y':
            quick_data_exploration(results['train_loader'])
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Script completed successfully!")
    print(f"\nYour cleaned, fixed data is ready in: {output_dir}")
    print("Use this directory when training your baseline recommender!")
