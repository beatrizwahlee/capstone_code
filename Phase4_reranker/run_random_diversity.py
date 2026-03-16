"""
Random Diversity Evaluation (targeted)
=======================================
Evaluates ONLY the random re-ranker on top of the baseline recommender.
Reuses the same evaluation pipeline as run_phase4_evaluation.py but skips
all other methods — expected runtime ~10-15 min.

Usage:
    python run_random_diversity.py

Output:
    Prints metrics to console and appends/updates the random entry in
    Phase4_reranker/outputs/diversity_evaluation/diversity_evaluation.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

base_dir = Path(__file__).resolve().parent

for p in [
    base_dir.parent / "Phase2_baseline_rec",
    base_dir.parent / "Phase3_echo_chambers",
    base_dir.parent / "Phase1_NLP_encoding",
    base_dir.parent / "Phase0_data_processing" / "data_processing",
]:
    if p.exists():
        sys.path.insert(0, str(p))

from diversity_reranker import DiversityReranker
from baseline_recommender_phase2 import BaselineRecommender
from echo_chamber_analyzer import EchoChamberAnalyzer

# Import helpers from the main evaluation script
sys.path.insert(0, str(base_dir))
from run_phase4_evaluation import (
    calculate_ranking_metrics_from_impressions,
    calculate_diversity_from_recs,
    CONFIG,
)


def main():
    t_start = time.time()

    print("\n" + "=" * 60)
    print("RANDOM RE-RANKER — DIVERSITY EVALUATION")
    print("=" * 60)

    # ── Load baseline ───────────────────────────────────────────
    logger.info("Loading baseline recommender ...")
    baseline = BaselineRecommender.load(
        str(Path(CONFIG["baseline_dir"]) / "baseline_recommender.pkl"),
        CONFIG["embeddings_dir"],
    )

    # ── Load news metadata ──────────────────────────────────────
    logger.info("Loading news metadata ...")
    processed_dir = (
        Path(CONFIG["baseline_dir"]).parent.parent.parent
        / "Phase0_data_processing" / "processed_data"
    )
    news_df = pd.read_csv(processed_dir / "news_features_train.csv")

    news_categories    = dict(zip(news_df['news_id'], news_df['category']))
    news_subcategories = dict(zip(news_df['news_id'], news_df['subcategory']))
    cat_counts         = news_df['category'].value_counts()
    corpus_category_dist    = (cat_counts / cat_counts.sum()).to_dict()
    subcat_counts           = news_df['subcategory'].value_counts()
    corpus_subcategory_dist = (subcat_counts / subcat_counts.sum()).to_dict()

    # ── Load validation data ────────────────────────────────────
    logger.info("Loading validation data ...")
    from mind_data_loader import MINDDataLoader
    loader = MINDDataLoader(CONFIG["valid_data_dir"], dataset_type='valid')
    loader.load_all_data()

    test_data = []
    for _, row in loader.behaviors_df.iterrows():
        if row['history'] and row['impressions']:
            test_data.append({
                'user_id':     row['user_id'],
                'history':     row['history'],
                'impressions': row['impressions'],
            })
        if CONFIG["max_test_samples"] and len(test_data) >= CONFIG["max_test_samples"]:
            break
    logger.info(f"Loaded {len(test_data)} test samples")

    # ── Build reranker ──────────────────────────────────────────
    reranker = DiversityReranker(
        baseline_recommender=baseline,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
        news_categories=news_categories,
        popularity_scores=baseline.popularity_scores,
        corpus_category_dist=corpus_category_dist,
        corpus_subcategory_dist=corpus_subcategory_dist,
        news_subcategories=news_subcategories,
    )

    analyzer = EchoChamberAnalyzer(
        recommender=baseline,
        news_df=news_df,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
    )

    # ── Evaluate random ─────────────────────────────────────────
    logger.info("\nEvaluating: random")
    rng = np.random.default_rng(seed=42)

    all_recommendations   = []
    all_labels_and_scores = []

    for i, sample in enumerate(test_data):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(test_data)} ...")

        history     = sample['history']
        impressions = sample['impressions']
        if not history or not impressions:
            continue

        # Baseline top-100 candidate pool (same as main eval)
        baseline_candidates = reranker.baseline.recommend(
            user_history=history, k=100, exclude_history=True,
        )
        if not baseline_candidates:
            continue

        # Random re-rank: shuffle the pool, take top-k
        shuffled = list(baseline_candidates)
        rng.shuffle(shuffled)
        diverse_recs = shuffled[:CONFIG['k']]
        all_recommendations.append(diverse_recs)

        # Accuracy: pure random scores on impression candidates (AUC lower bound)
        known = [(nid, lbl) for nid, lbl in impressions if nid in reranker.news_id_to_idx]
        if known:
            labels = [lbl for _, lbl in known]
            scores = list(rng.random(len(known)))
            all_labels_and_scores.append((labels, scores))

    logger.info(f"Generated {len(all_recommendations)} recommendation sets")

    # Accuracy metrics
    all_labels = [l for labels, _ in all_labels_and_scores for l in labels]
    all_scores = [s for _, scores in all_labels_and_scores for s in scores]
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0

    ranking = calculate_ranking_metrics_from_impressions(all_labels_and_scores, CONFIG['k'])
    accuracy_metrics = {'auc': auc, **ranking}

    # Diversity metrics
    diversity_metrics = calculate_diversity_from_recs(
        all_recommendations,
        test_data,
        analyzer,
        corpus_category_dist=corpus_category_dist,
        corpus_subcategory_dist=corpus_subcategory_dist,
        news_subcategories=news_subcategories,
    )

    result = {
        'method':   'random',
        'params':   {'seed': 42},
        'accuracy': accuracy_metrics,
        'diversity': diversity_metrics,
    }

    # ── Print results ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM RE-RANKER RESULTS")
    print("=" * 60)
    print("\nAccuracy:")
    for k, v in accuracy_metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("\nDiversity:")
    for k, v in diversity_metrics.items():
        print(f"  {k:<30}: {v:.4f}")

    # ── Save: prepend random to diversity_evaluation.json ───────
    output_path = Path(CONFIG["output_dir"]) / "diversity_evaluation.json"
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    # Remove any old random entry, then prepend the new one
    existing = [r for r in existing if r.get('method') != 'random']
    updated  = [result] + existing

    with open(output_path, 'w') as f:
        json.dump(updated, f, indent=2)

    logger.info(f"\nSaved to {output_path}")
    logger.info(f"Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
