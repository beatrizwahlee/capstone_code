"""
Phase 2 Training Script
=======================
Train and evaluate the baseline recommender on the MIND dataset.

This script:
  1. Loads Phase 1 embeddings (instant)
  2. Loads train/validation data from Phase 0
  3. Fits popularity scores from training clicks
  4. Evaluates on validation set
  5. Saves trained recommender + results

Usage:
    python train_phase2_baseline.py

Expected runtime: 10-30 min depending on dataset size and CPU speed.
Most time is spent on evaluation (scoring all impression candidates).
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

base_dir = Path(__file__).resolve().parent
phase1_path = base_dir.parent / "Phase1_NLP_encoding"
if phase1_path.exists():
    sys.path.insert(0, str(phase1_path))
phase0_path = base_dir.parent / "Phase0_data_processing" / "data_processing"
if phase0_path.exists():
    sys.path.insert(0, str(phase0_path))

from baseline_recommender_phase2 import (
    BaselineRecommender,
    RecommenderEvaluator,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Phase 1 embeddings directory
    "embeddings_dir": str(base_dir.parent / "Phase1_NLP_encoding" / "embeddings"),
    
    # Phase 0 processed data directory
    "processed_data_dir": str(base_dir.parent / "Phase0_data_processing" / "processed_data"),
    
    # Raw MIND data (for loading behaviors if processed data missing)
    "train_data_dir": str(base_dir.parent / "MINDlarge_train"),
    "valid_data_dir": str(base_dir.parent / "MINDlarge_dev"),
    
    # Output directory
    "output_dir": str(base_dir / "outputs" / "baseline"),
    
    # Hybrid scoring weights (overridden at runtime by dynamic weighting)
    "content_weight": 0.6,
    "popularity_weight": 0.2,
    "recency_weight": 0.2,

    # Multi-query FAISS: candidates retrieved per query vector
    # Query 1 = pool_size, Query 2 = pool_size//2, Query 3 = pool_size//3
    "candidate_pool_size": 150,

    # User profiler decay
    "decay_lambda": 0.3,
    
    # Evaluation settings
    "k_values": [5, 10, 20],
    "max_test_samples": None,  # Set to e.g. 5000 for fast testing, None for full eval
}


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def load_train_interactions(processed_dir: Path, raw_dir: Path) -> List[Dict]:
    """
    Load training interactions from the full raw MIND behaviors file.

    Prefers raw data over the 1000-row sample CSV so popularity scores
    are fitted on the complete training set. Adds 'impression_time' so
    fit_popularity() can compute per-article recency scores.

    Returns:
        List of dicts with keys: news_id, label, user_id, history,
                                 impression_time (datetime | None)
    """
    from datetime import datetime

    logger.info("Loading training interactions ...")

    # Primary: full raw MIND behaviors (all impressions + timestamps)
    behaviors_path = raw_dir / "behaviors.tsv"
    if raw_dir.exists() and behaviors_path.exists():
        logger.info(f"  Loading from raw MIND data (full dataset): {raw_dir}")
        from mind_data_loader import MINDDataLoader

        loader = MINDDataLoader(str(raw_dir), dataset_type='train')
        loader.load_all_data()

        interactions = []
        for _, row in loader.behaviors_df.iterrows():
            impression_time = None
            try:
                impression_time = datetime.strptime(row['time'], '%m/%d/%Y %I:%M:%S %p')
            except (ValueError, KeyError, TypeError):
                pass

            history = row['history']
            for news_id, label in row['impressions']:
                interactions.append({
                    'news_id': news_id,
                    'label': label,
                    'user_id': row['user_id'],
                    'history': history,
                    'impression_time': impression_time,
                })

        logger.info(f"  Loaded {len(interactions):,} training interactions from raw data")
        return interactions

    # Fallback: 1000-row sample CSV (raw data unavailable)
    train_path = processed_dir / "sample_train_interactions.csv"
    if train_path.exists():
        logger.warning(
            f"  Raw MIND data not found at {raw_dir}. "
            f"Falling back to sample CSV (popularity will be approximate)."
        )
        df = pd.read_csv(train_path)

        if 'history' in df.columns and isinstance(df['history'].iloc[0], str):
            import ast
            df['history'] = df['history'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        interactions = df.to_dict('records')
        # Add missing impression_time so fit_popularity() doesn't break
        for rec in interactions:
            rec.setdefault('impression_time', None)

        logger.info(f"  Loaded {len(interactions):,} training interactions (sample only)")
        return interactions

    logger.error(
        f"No training data found. Expected raw behaviors at {behaviors_path} "
        f"or sample CSV at {train_path}."
    )
    sys.exit(1)


def load_test_data(processed_dir: Path, raw_dir: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load validation/test data.
    
    Returns:
        List of dicts with keys: user_id, history, impressions
    """
    logger.info("Loading test data ...")
    
    # Try to load from raw validation directory
    # (processed_data typically only has training interactions)
    from mind_data_loader import MINDDataLoader
    
    loader = MINDDataLoader(str(raw_dir), dataset_type='valid')
    loader.load_all_data()
    
    test_data = []
    for _, row in loader.behaviors_df.iterrows():
        if row['impressions']:  # Only include if there are impressions to evaluate
            test_data.append({
                'user_id': row['user_id'],
                'history': row['history'],
                'impressions': row['impressions'],
            })
        
        if max_samples and len(test_data) >= max_samples:
            break
    
    logger.info(f"  Loaded {len(test_data):,} test impressions")
    return test_data


def load_news_metadata(processed_dir: Path, raw_dir: Path) -> pd.DataFrame:
    """Load news metadata (for diversity metrics)."""
    logger.info("Loading news metadata ...")
    
    # Try processed CSV
    news_path = processed_dir / "news_features_train.csv"
    if news_path.exists():
        df = pd.read_csv(news_path)
        logger.info(f"  Loaded {len(df):,} articles from processed CSV")
        return df
    
    # Fall back to raw
    from mind_data_loader import MINDDataLoader
    loader = MINDDataLoader(str(raw_dir), dataset_type='train')
    loader.load_all_data()
    logger.info(f"  Loaded {len(loader.news_df):,} articles from raw data")
    return loader.news_df


# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------

def train_baseline(train_interactions: List[Dict], news_df: pd.DataFrame) -> BaselineRecommender:
    """
    Train the baseline recommender.
    
    Training here just means fitting the popularity scores — the content
    embeddings are already pre-computed in Phase 1.
    """
    logger.info("=" * 60)
    logger.info("Training Baseline Recommender")
    logger.info("=" * 60)
    
    # Load Phase 1 embeddings
    recommender = BaselineRecommender.from_embeddings(
        CONFIG["embeddings_dir"],
        content_weight=CONFIG["content_weight"],
        popularity_weight=CONFIG["popularity_weight"],
        recency_weight=CONFIG["recency_weight"],
        decay_lambda=CONFIG["decay_lambda"],
        candidate_pool_size=CONFIG["candidate_pool_size"],
    )
    
    # Fit popularity scores from training data
    recommender.fit_popularity(train_interactions, news_df)
    
    logger.info("✔ Training complete!")
    return recommender


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------

def evaluate_baseline(
    recommender: BaselineRecommender,
    test_data: List[Dict],
    news_df: pd.DataFrame,
) -> Dict:
    """Evaluate the baseline recommender."""
    logger.info("=" * 60)
    logger.info("Evaluating Baseline Recommender")
    logger.info("=" * 60)
    
    evaluator = RecommenderEvaluator(recommender, news_df)
    
    # Accuracy metrics
    results = evaluator.evaluate(test_data, k_values=CONFIG["k_values"])
    
    # Diversity metrics (for baseline comparison)
    logger.info("\nGenerating recommendations for diversity metrics ...")
    sample_recs = []
    for sample in test_data[:1000]:  # Subset for diversity metrics
        recs = recommender.recommend(sample['history'], k=10)
        sample_recs.append(recs)
    
    diversity_results = evaluator.calculate_diversity_metrics(sample_recs)
    results['diversity'] = diversity_results
    
    return results


# ---------------------------------------------------------------------------
# Step 4: Print and save results
# ---------------------------------------------------------------------------

def print_results(results: Dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    
    print("\n--- Accuracy Metrics ---")
    print(f"  AUC:         {results['auc']:.4f}")
    print(f"  MRR:         {results['mrr']:.4f}")
    
    for k in CONFIG["k_values"]:
        print(f"\n@{k}:")
        print(f"  Precision:   {results[f'precision@{k}']:.4f}")
        print(f"  Recall:      {results[f'recall@{k}']:.4f}")
        print(f"  F1:          {results[f'f1@{k}']:.4f}")
        print(f"  NDCG:        {results[f'ndcg@{k}']:.4f}")
    
    if 'diversity' in results:
        print("\n--- Diversity Metrics (Baseline) ---")
        div = results['diversity']
        print(f"  Avg Coverage:              {div['avg_coverage']:.2f}")
        print(f"  Avg Gini (per user):       {div['avg_gini']:.4f}")
        print(f"  Overall Gini:              {div['overall_gini']:.4f}")
        print(f"  Unique categories in recs: {div['unique_categories_recommended']}")
    
    print("\n" + "=" * 60)


def save_results(recommender: BaselineRecommender, results: Dict, output_dir: Path):
    """Save trained model and results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save recommender
    recommender.save(output_dir / "baseline_recommender.pkl")
    
    # Save results
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj
    
    results_clean = convert_types(results)
    
    with open(output_dir / "baseline_results.json", 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    logger.info(f"\n✔ All outputs saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    
    print("\n" + "=" * 60)
    print("PHASE 2 — BASELINE RECOMMENDER TRAINING")
    print("Diversity-Aware News Recommender — Capstone Project")
    print("=" * 60)
    
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k:25s}: {v}")
    
    # Validate paths
    embeddings_dir = Path(CONFIG["embeddings_dir"])
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        logger.error("Run Phase 1 first: python run_phase1_encoding.py")
        sys.exit(1)
    
    processed_dir = Path(CONFIG["processed_data_dir"])
    train_raw_dir = Path(CONFIG["train_data_dir"])
    valid_raw_dir = Path(CONFIG["valid_data_dir"])
    
    # ---- Step 1: Load data ----
    train_interactions = load_train_interactions(processed_dir, train_raw_dir)
    test_data = load_test_data(processed_dir, valid_raw_dir, CONFIG["max_test_samples"])
    news_df = load_news_metadata(processed_dir, train_raw_dir)
    
    # ---- Step 2: Train ----
    recommender = train_baseline(train_interactions, news_df)
    
    # ---- Step 3: Evaluate ----
    results = evaluate_baseline(recommender, test_data, news_df)
    
    # ---- Step 4: Print and save ----
    print_results(results)
    save_results(recommender, results, Path(CONFIG["output_dir"]))
    
    # ---- Summary ----
    total_time = time.time() - t_start
    print(f"\n✔ Phase 2 complete in {total_time/60:.1f} min")
    print(f"\nNext steps:")
    print(f"  Phase 3 — Echo Chamber Analysis (measure diversity in baseline)")
    print(f"  Phase 4 — Diversity Re-ranking (MMR, xQuAD, Calibration, Fairness)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
