"""
Train and Evaluate Baseline Content-Based Recommender

This script trains the accuracy-optimized baseline recommender system
and evaluates it using comprehensive metrics.

Usage:
    python train_baseline_recommender.py
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Import our modules
from mind_data_loader import MINDDataLoader, MINDPreprocessor
from content_based_recommender import (
    NewsEncoder, UserProfiler, ContentBasedRecommender, 
    RecommenderEvaluator, save_model
)
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: str = './processed_data'):
    """Load pre-processed data."""
    logger.info("Loading processed data...")
    
    data_dir = Path(data_dir)
    
    # Load news features
    news_df = pd.read_csv(data_dir / 'news_features_train.csv')
    
    # Load encoders
    with open(data_dir / 'encoders.json', 'r') as f:
        encoders = json.load(f)
    
    # Load diversity stats
    with open(data_dir / 'diversity_stats.json', 'r') as f:
        diversity_stats = json.load(f)
    
    # Load sample interactions (for testing, you'd load full data)
    interactions_df = pd.read_csv(
        data_dir / 'sample_train_interactions.csv',
        nrows=5000  # Limit for faster testing
    )
    
    logger.info(f"Loaded {len(news_df)} news articles")
    logger.info(f"Loaded {len(interactions_df)} interaction samples")
    
    return news_df, encoders, diversity_stats, interactions_df


def prepare_train_test_split(interactions_df: pd.DataFrame, 
                             test_ratio: float = 0.2):
    """Split data into train and test sets."""
    logger.info("Splitting data into train/test...")
    
    # Parse history from string representation
    interactions_df['history'] = interactions_df['history'].apply(eval)
    interactions_df['history_indices'] = interactions_df['history_indices'].apply(eval)
    
    # Group by impression to keep impressions together
    impression_ids = interactions_df['impression_id'].unique()
    np.random.shuffle(impression_ids)
    
    split_idx = int(len(impression_ids) * (1 - test_ratio))
    train_impressions = set(impression_ids[:split_idx])
    test_impressions = set(impression_ids[split_idx:])
    
    train_df = interactions_df[
        interactions_df['impression_id'].isin(train_impressions)
    ]
    test_df = interactions_df[
        interactions_df['impression_id'].isin(test_impressions)
    ]
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    return train_df, test_df


def convert_to_test_format(interactions_df: pd.DataFrame) -> List[Dict]:
    """
    Convert interaction DataFrame to test format.
    Groups by impression_id to create proper test samples.
    """
    test_data = []
    
    grouped = interactions_df.groupby('impression_id')
    
    for impression_id, group in grouped:
        # Get first row for user info (same for all in group)
        first_row = group.iloc[0]
        
        # Collect impressions
        impressions = [
            (row['news_id'], row['label'])
            for _, row in group.iterrows()
        ]
        
        test_sample = {
            'user_id': first_row['user_id'],
            'user_idx': first_row['user_idx'],
            'history': first_row['history'],
            'impressions': impressions,
            'impression_id': impression_id
        }
        
        test_data.append(test_sample)
    
    return test_data


def train_recommender(news_df: pd.DataFrame, train_df: pd.DataFrame,
                     entity_embeddings: Dict = None):
    """
    Train the content-based recommender.
    
    Args:
        news_df: News features DataFrame
        train_df: Training interactions DataFrame
        entity_embeddings: Entity embeddings dictionary
    
    Returns:
        Trained ContentBasedRecommender
    """
    logger.info("=" * 80)
    logger.info("TRAINING BASELINE RECOMMENDER")
    logger.info("=" * 80)
    
    # Step 1: Encode news articles
    logger.info("\n[Step 1/3] Encoding news articles...")
    news_encoder = NewsEncoder(
        use_entities=True,
        use_categories=True
    )
    news_embeddings = news_encoder.fit_transform(news_df, entity_embeddings)
    
    # Step 2: Build user profiles
    logger.info("\n[Step 2/3] Building user profiles...")
    user_profiler = UserProfiler(news_encoder)
    
    # Get unique users and their histories from training data
    user_histories = {}
    for _, row in train_df.iterrows():
        user_id = row['user_id']
        if user_id not in user_histories:
            user_histories[user_id] = row['history']
    
    user_profiler.build_profiles_batch(user_histories, method='weighted')
    
    # Step 3: Create recommender
    logger.info("\n[Step 3/3] Creating recommender system...")
    recommender = ContentBasedRecommender(news_encoder, user_profiler)
    
    logger.info("✓ Training complete!")
    
    return recommender


def evaluate_recommender(recommender: ContentBasedRecommender,
                        test_df: pd.DataFrame, news_df: pd.DataFrame):
    """
    Evaluate the recommender on test data.
    
    Args:
        recommender: Trained recommender
        test_df: Test interactions DataFrame
        news_df: News features for diversity metrics
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 80)
    logger.info("EVALUATING RECOMMENDER")
    logger.info("=" * 80)
    
    # Convert test data to proper format
    test_data = convert_to_test_format(test_df)
    
    logger.info(f"Evaluating on {len(test_data)} test impressions...")
    
    # Create evaluator
    evaluator = RecommenderEvaluator(recommender, news_df)
    
    # Calculate accuracy metrics
    logger.info("\nCalculating accuracy metrics...")
    accuracy_metrics = evaluator.evaluate(test_data, k_values=[5, 10, 20])
    
    # Generate recommendations for diversity metrics
    logger.info("\nGenerating recommendations for diversity analysis...")
    all_recommendations = []
    for sample in test_data[:100]:  # Use subset for diversity metrics
        user_profile = recommender.user_profiler.build_user_profile(
            sample['history']
        )
        recs = recommender.recommend(
            user_profile=user_profile,
            candidate_ids=None,  # All news
            top_k=10,
            exclude_history=True,
            user_history=sample['history']
        )
        all_recommendations.append(recs)
    
    # Calculate diversity metrics
    logger.info("\nCalculating diversity metrics...")
    diversity_metrics = evaluator.calculate_diversity_metrics(
        all_recommendations, test_data[:100]
    )
    
    # Combine results
    results = {
        'accuracy': accuracy_metrics,
        'diversity': diversity_metrics,
        'num_test_samples': len(test_data)
    }
    
    return results


def print_results(results: Dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\n--- ACCURACY METRICS ---")
    accuracy = results['accuracy']
    
    # Overall metrics
    print(f"\nOverall:")
    print(f"  AUC:                 {accuracy['auc']:.4f}")
    print(f"  MRR:                 {accuracy['mrr']:.4f}")
    
    # Metrics at different K values
    for k in [5, 10, 20]:
        print(f"\n@{k}:")
        print(f"  Precision@{k}:        {accuracy[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:           {accuracy[f'recall@{k}']:.4f}")
        print(f"  F1@{k}:               {accuracy[f'f1@{k}']:.4f}")
        print(f"  NDCG@{k}:             {accuracy[f'ndcg@{k}']:.4f}")
    
    print("\n--- DIVERSITY METRICS ---")
    diversity = results['diversity']
    print(f"  Avg Category Diversity:  {diversity['avg_category_diversity']:.4f}")
    print(f"  Avg Coverage:            {diversity['avg_coverage']:.2f}")
    print(f"  Gini Coefficient:        {diversity['gini_coefficient']:.4f}")
    
    print("\n" + "=" * 80)


def save_results(results: Dict, filepath: str):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_converted = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")


def main():
    """Main training and evaluation pipeline."""
    
    print("\n" + "=" * 80)
    print("BASELINE CONTENT-BASED RECOMMENDER SYSTEM")
    print("Diversity-Aware News Recommendation - Capstone Project")
    print("=" * 80)
    
    # Create output directories
    Config.create_directories()
    output_dir = Config.OUTPUT_DIR / 'baseline'
    output_dir.mkdir(exist_ok=True)
    
    # Load processed data
    try:
        news_df, encoders, diversity_stats, interactions_df = load_processed_data()
    except FileNotFoundError as e:
        logger.error(
            "Processed data not found! Please run example_usage.py first."
        )
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    # Note: For full training, you would load entity embeddings
    # For now, we'll use None and rely on TF-IDF + categories
    entity_embeddings = None  # Would load from MINDDataLoader
    
    # Split data
    train_df, test_df = prepare_train_test_split(interactions_df, test_ratio=0.2)
    
    # Train recommender
    recommender = train_recommender(news_df, train_df, entity_embeddings)
    
    # Save model
    model_path = output_dir / 'baseline_recommender.pkl'
    save_model(recommender, str(model_path))
    
    # Evaluate
    results = evaluate_recommender(recommender, test_df, news_df)
    
    # Print and save results
    print_results(results)
    
    results_path = output_dir / f'baseline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_results(results, str(results_path))
    
    # Save configuration
    config_path = output_dir / 'config.json'
    Config.save_config(str(config_path))
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. ✓ Baseline recommender trained and evaluated")
    print("2. → Implement diversity-aware re-ranking algorithms")
    print("3. → Compare baseline vs. diversity-aware performance")
    print("4. → Analyze accuracy-diversity trade-offs")
    print("=" * 80)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("- Model: baseline_recommender.pkl")
    print(f"- Results: {results_path.name}")
    print("- Config: config.json")
    
    return recommender, results


if __name__ == "__main__":
    try:
        recommender, results = main()
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Baseline recommender training complete!")
