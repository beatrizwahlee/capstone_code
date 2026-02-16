"""
Complete Training Pipeline with Full MIND Dataset

This script demonstrates how to load the full MIND dataset with entity embeddings
and train the baseline recommender properly.

Usage:
    python train_with_full_dataset.py --data_dir /path/to/MIND
"""

import argparse
import sys
from pathlib import Path
import logging

from mind_data_loader import MINDDataLoader, MINDPreprocessor
from content_based_recommender import (
    NewsEncoder, UserProfiler, ContentBasedRecommender,
    RecommenderEvaluator, save_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """
    Complete training pipeline with full dataset.
    """
    
    print("\n" + "=" * 80)
    print("TRAINING WITH FULL MIND DATASET")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: Load Training Data
    # =========================================================================
    print("\n[STEP 1] Loading training data...")
    
    train_loader = MINDDataLoader(args.train_dir, dataset_type='train')
    train_loader.load_all_data()
    
    # Preprocess
    train_preprocessor = MINDPreprocessor(train_loader)
    train_processed = train_preprocessor.preprocess_all()
    
    print(f"✓ Loaded {len(train_loader.news_df)} news articles")
    print(f"✓ Loaded {len(train_loader.behaviors_df)} impressions")
    print(f"✓ {len(train_loader.entity_embeddings)} entity embeddings")
    
    # =========================================================================
    # STEP 2: Load Validation Data
    # =========================================================================
    print("\n[STEP 2] Loading validation data...")
    
    valid_loader = MINDDataLoader(args.valid_dir, dataset_type='valid')
    valid_loader.load_all_data()
    
    valid_preprocessor = MINDPreprocessor(valid_loader)
    valid_processed = valid_preprocessor.preprocess_all()
    
    print(f"✓ Loaded {len(valid_loader.behaviors_df)} validation impressions")
    
    # =========================================================================
    # STEP 3: Encode News Articles
    # =========================================================================
    print("\n[STEP 3] Encoding news articles...")
    
    news_encoder = NewsEncoder(
        use_entities=True,
        use_categories=True
    )
    
    # Encode training news
    news_embeddings = news_encoder.fit_transform(
        train_loader.news_df,
        train_loader.entity_embeddings
    )
    
    print(f"✓ News embeddings shape: {news_embeddings.shape}")
    
    # =========================================================================
    # STEP 4: Build User Profiles
    # =========================================================================
    print("\n[STEP 4] Building user profiles...")
    
    user_profiler = UserProfiler(news_encoder)
    
    # Build profiles for training users
    user_histories = train_processed['user_item_matrix']['user_histories']
    user_profiler.build_profiles_batch(user_histories, method='weighted')
    
    print(f"✓ Built {len(user_profiler.user_profiles)} user profiles")
    
    # =========================================================================
    # STEP 5: Create Recommender
    # =========================================================================
    print("\n[STEP 5] Creating recommender system...")
    
    recommender = ContentBasedRecommender(news_encoder, user_profiler)
    
    print("✓ Recommender created")
    
    # =========================================================================
    # STEP 6: Evaluate on Validation Set
    # =========================================================================
    print("\n[STEP 6] Evaluating on validation set...")
    
    evaluator = RecommenderEvaluator(recommender, train_loader.news_df)
    
    # Use validation data
    val_data = valid_processed['val_data']
    
    print(f"Evaluating on {len(val_data)} validation samples...")
    
    # Evaluate
    results = evaluator.evaluate(val_data, k_values=[5, 10, 20])
    
    # =========================================================================
    # STEP 7: Print Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"\nAUC:                 {results['auc']:.4f}")
    print(f"MRR:                 {results['mrr']:.4f}")
    
    for k in [5, 10, 20]:
        print(f"\n@{k}:")
        print(f"  Precision@{k}:        {results[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:           {results[f'recall@{k}']:.4f}")
        print(f"  F1@{k}:               {results[f'f1@{k}']:.4f}")
        print(f"  NDCG@{k}:             {results[f'ndcg@{k}']:.4f}")
    
    print("\n" + "=" * 80)
    
    # =========================================================================
    # STEP 8: Save Model
    # =========================================================================
    print("\n[STEP 8] Saving model...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'baseline_recommender.pkl'
    save_model(recommender, str(model_path))
    
    # Save results
    import json
    results_path = output_dir / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Results saved to {results_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    return recommender, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train baseline recommender with full MIND dataset'
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./MINDlarge_train',
        help='Path to training data directory'
    )
    
    parser.add_argument(
        '--valid_dir',
        type=str,
        default='./MINDlarge_dev',
        help='Path to validation data directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/baseline',
        help='Output directory for model and results'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.train_dir).exists():
        logger.error(f"Training directory not found: {args.train_dir}")
        logger.error("Please download and extract MIND dataset first!")
        sys.exit(1)
    
    if not Path(args.valid_dir).exists():
        logger.error(f"Validation directory not found: {args.valid_dir}")
        logger.error("Please download and extract MIND dataset first!")
        sys.exit(1)
    
    try:
        recommender, results = main(args)
        print("\n✓ Success!")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
