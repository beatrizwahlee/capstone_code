"""
Quick re-evaluation of the composite re-ranker only.
Reuses all infrastructure from run_phase4_evaluation.py.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

for p in ["Phase2_baseline_rec", "Phase3_echo_chambers", "Phase1_NLP_encoding"]:
    path = base_dir.parent / p
    if path.exists():
        sys.path.insert(0, str(path))

sys.path.insert(0, str(base_dir.parent / "Phase0_data_processing" / "data_processing"))

from diversity_reranker import DiversityReranker
from baseline_recommender_phase2 import BaselineRecommender
from run_phase4_evaluation import (
    CONFIG,
    evaluate_method,
    calculate_diversity_from_recs,
)
from echo_chamber_analyzer import EchoChamberAnalyzer


def main():
    t0 = time.time()

    print("\n" + "=" * 60)
    print("COMPOSITE RE-RANKER — QUICK RE-EVALUATION")
    print("=" * 60)

    # Load baseline
    logger.info("Loading baseline recommender ...")
    baseline = BaselineRecommender.load(
        str(Path(CONFIG["baseline_dir"]) / "baseline_recommender.pkl"),
        CONFIG["embeddings_dir"],
    )

    logger.info("Loading news metadata ...")
    processed_dir = (
        Path(CONFIG["baseline_dir"]).parent.parent.parent
        / "Phase0_data_processing" / "processed_data"
    )
    news_df = pd.read_csv(processed_dir / "news_features_train.csv")

    logger.info("Loading validation data ...")
    from mind_data_loader import MINDDataLoader
    loader = MINDDataLoader(CONFIG["valid_data_dir"], dataset_type='valid')
    loader.load_all_data()

    test_data = []
    for _, row in loader.behaviors_df.iterrows():
        if row['history'] and row['impressions']:
            test_data.append({
                'user_id': row['user_id'],
                'history': row['history'],
                'impressions': row['impressions'],
            })
        if CONFIG["max_test_samples"] and len(test_data) >= CONFIG["max_test_samples"]:
            break

    logger.info(f"Loaded {len(test_data)} test samples")

    # Build reranker
    news_categories    = dict(zip(news_df['news_id'], news_df['category']))
    news_subcategories = dict(zip(news_df['news_id'], news_df['subcategory']))

    cat_counts    = news_df['category'].value_counts()
    subcat_counts = news_df['subcategory'].value_counts()

    reranker = DiversityReranker(
        baseline_recommender=baseline,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
        news_categories=news_categories,
        popularity_scores=baseline.popularity_scores,
        corpus_category_dist=(cat_counts / cat_counts.sum()).to_dict(),
        corpus_subcategory_dist=(subcat_counts / subcat_counts.sum()).to_dict(),
        news_subcategories=news_subcategories,
    )

    composite_params = {
        'w_relevance':   CONFIG['composite_w_relevance'],
        'w_diversity':   CONFIG['composite_w_diversity'],
        'w_calibration': CONFIG['composite_w_calibration'],
        'w_serendipity': CONFIG['composite_w_serendipity'],
        'w_fairness':    CONFIG['composite_w_fairness'],
        'explore_weight': CONFIG['composite_explore_weight'],
    }

    logger.info(f"\nComposite params: {composite_params}")

    result = evaluate_method(
        'composite', reranker, test_data, news_df, composite_params
    )

    elapsed = time.time() - t0
    acc = result['accuracy']
    div = result['diversity']

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  AUC            : {acc['auc']:.4f}")
    print(f"  NDCG@10        : {acc['ndcg@10']:.4f}")
    print(f"  MRR            : {acc['mrr']:.4f}")
    print(f"  Precision@10   : {acc.get('precision@10', 0):.4f}")
    print(f"  Recall@10      : {acc.get('recall@10', 0):.4f}")
    print(f"\n  Avg Gini       : {div['avg_gini']:.4f}  (lower = more diverse)")
    print(f"  Avg ILD        : {div['avg_ild']:.4f}  (higher = more diverse)")
    print(f"  Avg Coverage   : {div['avg_coverage']:.4f}  (higher = better)")
    print(f"  Avg Entropy    : {div['avg_entropy']:.4f}  (higher = more diverse)")
    print(f"  Cal. Error     : {div['avg_calibration_error']:.4f}  (lower = better)")
    print(f"  System HHI     : {div.get('system_hhi', 0):.4f}  (lower = less concentrated)")
    print(f"  System KL-div  : {div.get('system_kl_divergence', 0):.4f}  (lower = more representative)")

    # Compare vs old composite metrics from the saved evaluation
    print("\n" + "-" * 60)
    print("COMPARISON vs previous composite run:")
    print("-" * 60)
    old = {
        'auc': 0.6293, 'ndcg@10': 0.3928, 'mrr': 0.3370,
        'avg_gini': 0.4076, 'avg_ild': 0.5620, 'avg_coverage': 0.2159,
        'avg_entropy': 1.4321, 'avg_calibration_error': 1.0621,
    }
    print(f"  {'Metric':<22} {'Old':>8} {'New':>8} {'Delta':>10}")
    print(f"  {'-'*50}")
    for metric, old_val in old.items():
        if metric in acc:
            new_val = acc[metric]
        elif metric in div:
            new_val = div[metric]
        else:
            continue
        delta = new_val - old_val
        sign = '+' if delta >= 0 else ''
        print(f"  {metric:<22} {old_val:>8.4f} {new_val:>8.4f} {sign}{delta:>9.4f}")

    print(f"\n  Completed in {elapsed/60:.1f} min")

    # Persist result
    out_path = Path(CONFIG["output_dir"]) / "composite_reeval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
