"""
Phase 4 Evaluation Script
=========================
Evaluate all 7 diversity re-ranking algorithms and compare vs baseline.

This script:
  1. Loads baseline recommender
  2. For each algorithm (MMR, xQuAD, Calibrated, Serendipity,
     Bounded Greedy, Max Coverage, Composite):
     - Generates recommendations for test users
     - Measures accuracy (AUC, NDCG@10, MRR, Precision@10, Recall@10)
       using impression-based evaluation (matches Phase 2 protocol)
     - Measures diversity (Gini, ILD, Coverage, Entropy, CalibrationError,
       PopularityMiscalibration)
  3. Creates comparison table (before/after)
  4. Saves comprehensive report

Evaluation note:
  NDCG@10 / MRR / Precision@10 are computed by scoring ALL impression
  candidates for each user (not by checking FAISS top-10 overlap with
  clicked articles). This matches Phase 2 and gives correct scale (~0.37
  for baseline NDCG@10 vs the near-zero values from FAISS overlap).

Usage:
    python run_phase4_evaluation.py

Expected runtime: 60-120 min (8 full evaluations: baseline + 7 algorithms)
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

phase2_path = base_dir.parent / "Phase2_baseline_rec"
if phase2_path.exists():
    sys.path.insert(0, str(phase2_path))

phase3_path = base_dir.parent / "Phase3_echo_chambers"
if phase3_path.exists():
    sys.path.insert(0, str(phase3_path))

phase1_path = base_dir.parent / "Phase1_NLP_encoding"
if phase1_path.exists():
    sys.path.insert(0, str(phase1_path))

phase0_path = base_dir.parent / "Phase0_data_processing" / "data_processing"
if phase0_path.exists():
    sys.path.insert(0, str(phase0_path))

from diversity_reranker import DiversityReranker
from baseline_recommender_phase2 import BaselineRecommender
from echo_chamber_analyzer import EchoChamberAnalyzer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Directories
    "baseline_dir": str(base_dir.parent / "Phase2_baseline_rec" / "outputs" / "baseline"),
    "embeddings_dir": str(base_dir.parent / "Phase1_NLP_encoding" / "embeddings"),
    "valid_data_dir": str(base_dir.parent / "MINDlarge_dev"),
    "output_dir": str(base_dir / "outputs" / "diversity_evaluation"),

    # Evaluation settings
    "k": 10,
    "max_test_samples": 10000,  # Set to None for full evaluation

    # Algorithm parameters
    "mmr_lambda": 0.5,              # 0.5 = balanced relevance/diversity; 0.3 over-diversifies with pool augmentation
    "xquad_lambda": 0.5,            # Balance relevance and coverage
    "xquad_explore_weight": 0.3,    # Uniform prior blend
    "calibration_alpha": 0.6,       # Higher = stricter calibration
    "calibration_diversity_weight": 0.3,  # Uniform distribution blend
    "serendipity_beta": 0.4,        # Higher = more serendipity
    "bounded_greedy_max_per_cat": 2,       # Hard cap per category
    "max_coverage_budget": 0.6,     # Fraction of slots for coverage

    # Composite re-ranker weights (must sum to 1.0)
    "composite_w_relevance":   0.40,  # Relevance component
    "composite_w_diversity":   0.15,  # Embedding diversity (1 - max_sim to selected)
    "composite_w_calibration": 0.15,  # Calibration gap vs user-adjusted target dist
    "composite_w_serendipity": 0.15,  # Unexpectedness from user centroid
    "composite_w_fairness":    0.15,  # Popularity level matching user preference
    "composite_explore_weight": 0.30, # Uniform prior blend in calibration target
}


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_method(
    method_name: str,
    reranker: DiversityReranker,
    test_data: List[Dict],
    news_df: pd.DataFrame,
    method_params: Dict,
) -> Dict:
    """
    Evaluate a single re-ranking method.

    Returns:
        Dict with accuracy and diversity metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {method_name}")
    logger.info(f"Parameters: {method_params}")
    logger.info(f"{'='*60}")

    all_recommendations = []
    all_labels_and_scores = []

    for i, sample in enumerate(test_data):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(test_data)} users ...")

        history = sample['history']
        impressions = sample['impressions']

        if not history or not impressions:
            continue

        # Get baseline top-100 candidates
        # exclude_history=True matches Phase 3 methodology and avoids
        # re-recommending already-read articles (more realistic).
        baseline_candidates = reranker.baseline.recommend(
            user_history=history,
            k=100,
            exclude_history=True,
        )

        if not baseline_candidates:
            continue

        # Re-rank with diversity method
        if method_name == 'baseline':
            diverse_recs = baseline_candidates[:CONFIG['k']]
        else:
            diverse_recs = reranker.rerank(
                candidates=baseline_candidates,
                user_history=history,
                k=CONFIG['k'],
                method=method_name,
                **method_params,
            )

        all_recommendations.append(diverse_recs)

        # Accuracy: score all impressions via baseline
        if impressions:
            known_impressions = [
                (nid, label) for nid, label in impressions
                if nid in reranker.news_id_to_idx
            ]

            if known_impressions:
                candidate_ids = [nid for nid, _ in known_impressions]
                labels = [label for _, label in known_impressions]

                # exclude_history=False intentional: impression lists in MIND
                # can contain already-read articles and all must be scored for AUC.
                full_recs = reranker.baseline.recommend(
                    user_history=history,
                    k=len(candidate_ids),
                    candidates=candidate_ids,
                    exclude_history=False,
                )

                rec_dict = dict(full_recs)
                scores = [rec_dict.get(nid, 0.0) for nid in candidate_ids]

                all_labels_and_scores.append((labels, scores))

    logger.info(f"  Generated {len(all_recommendations)} recommendation sets")

    # Accuracy metrics
    from sklearn.metrics import roc_auc_score

    all_labels = []
    all_scores = []
    for labels, scores in all_labels_and_scores:
        all_labels.extend(labels)
        all_scores.extend(scores)

    accuracy_metrics = {
        'auc': roc_auc_score(all_labels, all_scores)
        if len(set(all_labels)) > 1 else 0.0,
    }

    # Impression-based ranking metrics (matches Phase 2 evaluation protocol)
    ranking_metrics = calculate_ranking_metrics_from_impressions(
        all_labels_and_scores, CONFIG['k']
    )
    accuracy_metrics.update(ranking_metrics)

    # Diversity metrics
    analyzer = EchoChamberAnalyzer(
        recommender=reranker.baseline,
        news_df=news_df,
        embeddings=reranker.embeddings,
        news_id_to_idx=reranker.news_id_to_idx,
    )

    diversity_metrics = calculate_diversity_from_recs(
        all_recommendations,
        test_data,
        analyzer,
        popularity_scores=reranker.popularity_scores,
    )

    return {
        'method': method_name,
        'params': method_params,
        'accuracy': accuracy_metrics,
        'diversity': diversity_metrics,
    }


def calculate_ranking_metrics_from_impressions(
    all_labels_and_scores: List[Tuple[List[int], List[float]]],
    k: int = 10,
) -> Dict:
    """
    Calculate NDCG, MRR, Precision@K from impression-based scoring.

    Matches Phase 2 evaluation protocol: for each impression, score ALL
    impression candidates, rank them by score, then compute ranking metrics
    from the ranked list.  This gives NDCG@10 ~ 0.37 for baseline (vs the
    incorrect ~0.002 that results from checking FAISS top-10 overlap with
    impression clicks — a near-zero set intersection by construction).

    Args:
        all_labels_and_scores: List of (labels, scores) per impression.
                               labels: 0/1 per candidate; scores: model scores.
        k: Cutoff for metrics (default 10)

    Returns:
        Dict with ndcg@k, mrr, precision@k, recall@k
    """
    ndcg_scores = []
    mrr_scores = []
    precision_scores = []
    recall_scores = []

    for labels, scores in all_labels_and_scores:
        if not labels or not any(l == 1 for l in labels):
            continue

        # Rank all impression candidates by model score (descending)
        ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        top_k = ranked[:k]

        # Precision@K
        hits_at_k = sum(1 for _, l in top_k if l == 1)
        precision_scores.append(hits_at_k / k)

        # Recall@K
        total_clicks = sum(1 for l in labels if l == 1)
        recall_scores.append(hits_at_k / total_clicks if total_clicks > 0 else 0.0)

        # MRR: reciprocal rank of the first relevant item in top-K
        for rank, (_, label) in enumerate(top_k, 1):
            if label == 1:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        # NDCG@K
        dcg = sum(label / np.log2(rank + 1) for rank, (_, label) in enumerate(top_k, 1))
        ideal = sorted(labels, reverse=True)[:k]
        idcg = sum(l / np.log2(rank + 1) for rank, l in enumerate(ideal, 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f'ndcg@{k}': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        'mrr': float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        f'precision@{k}': float(np.mean(precision_scores)) if precision_scores else 0.0,
        f'recall@{k}': float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def calculate_diversity_from_recs(
    recommendations: List[List[Tuple]],
    test_data: List[Dict],
    analyzer: EchoChamberAnalyzer,
    popularity_scores: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Calculate diversity metrics from recommendations.

    Metrics:
      - avg_gini:                    Gini coefficient (0 = equal, 1 = monopoly)
      - avg_ild:                     Intra-List Diversity (embedding distance)
      - avg_coverage:                Fraction of all categories represented
      - avg_entropy:                 Shannon entropy of category distribution
      - avg_calibration_error:       KL divergence vs user history distribution
      - avg_popularity_miscalibration: |mean_pop(recs) - mean_pop(history)|
                                       User-side fairness: lower = recommendations
                                       match the popularity level the user prefers.
    """
    gini_scores = []
    ild_scores = []
    coverage_scores = []
    entropy_scores = []
    calibration_errors = []
    popularity_miscalibration = []

    all_categories = sorted(set(analyzer.news_to_category.values()))
    n_cats = max(1, len(all_categories))

    for recs, sample in zip(recommendations, test_data):
        rec_ids = [nid for nid, _ in recs]

        rec_cats = [
            analyzer.news_to_category[nid]
            for nid in rec_ids
            if nid in analyzer.news_to_category
        ]

        if rec_cats:
            gini_scores.append(analyzer.calculate_gini(rec_cats))
            coverage_scores.append(analyzer.calculate_coverage(rec_cats))

            # Shannon entropy of recommendation category distribution
            counts = Counter(rec_cats)
            total = sum(counts.values())
            probs = np.array([counts.get(c, 0) / total for c in all_categories])
            probs_nonzero = probs[probs > 0]
            ent = float(-np.sum(probs_nonzero * np.log2(probs_nonzero)))
            entropy_scores.append(ent)

            # Calibration error: KL(history_dist || rec_dist)
            history = sample.get('history', [])
            hist_cats = [
                analyzer.news_to_category[nid]
                for nid in history
                if nid in analyzer.news_to_category
            ]
            if hist_cats:
                hist_counts = Counter(hist_cats)
                hist_total = sum(hist_counts.values())
                smoothing = 0.01
                denom = smoothing * n_cats
                hist_probs = np.array([
                    (hist_counts.get(c, 0) + smoothing) / (hist_total + denom)
                    for c in all_categories
                ])
                rec_probs = np.array([
                    (counts.get(c, 0) + smoothing) / (total + denom)
                    for c in all_categories
                ])
                # KL divergence (history || recs)
                kl = float(np.sum(
                    hist_probs * np.log(hist_probs / rec_probs)
                ))
                calibration_errors.append(kl)

            # User-side popularity fairness:
            # |mean_pop(recs) - mean_pop(history)| measures whether the
            # recommendation popularity level matches the user's preference.
            if popularity_scores is not None:
                history = sample.get('history', [])
                hist_pops = [
                    popularity_scores.get(nid, 0.0) for nid in history
                    if nid in popularity_scores
                ]
                rec_pops = [
                    popularity_scores.get(nid, 0.0) for nid in rec_ids
                    if nid in popularity_scores
                ]
                if hist_pops and rec_pops:
                    popularity_miscalibration.append(
                        abs(float(np.mean(rec_pops)) - float(np.mean(hist_pops)))
                    )

        ild = analyzer.calculate_ild(rec_ids)
        ild_scores.append(ild)

    return {
        'avg_gini': np.mean(gini_scores) if gini_scores else 0.0,
        'avg_ild': np.mean(ild_scores) if ild_scores else 0.0,
        'avg_coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
        'avg_entropy': np.mean(entropy_scores) if entropy_scores else 0.0,
        'avg_calibration_error': np.mean(calibration_errors) if calibration_errors else 0.0,
        'avg_popularity_miscalibration': (
            np.mean(popularity_miscalibration) if popularity_miscalibration else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("\n" + "=" * 70)
    print("PHASE 4 — DIVERSITY RE-RANKING EVALUATION")
    print("Diversity-Aware News Recommender — Capstone Project")
    print("=" * 70)

    # Load components
    logger.info("\nLoading baseline recommender ...")
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

    logger.info(f"  Loaded {len(test_data)} test samples")

    # Create reranker
    logger.info("\nInitializing diversity reranker ...")
    news_categories = dict(zip(news_df['news_id'], news_df['category']))

    reranker = DiversityReranker(
        baseline_recommender=baseline,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
        news_categories=news_categories,
        popularity_scores=baseline.popularity_scores,
    )

    # Evaluate all methods
    methods = [
        ('baseline', {}),
        ('mmr', {'lambda_param': CONFIG['mmr_lambda']}),
        ('xquad', {
            'lambda_param': CONFIG['xquad_lambda'],
            'explore_weight': CONFIG['xquad_explore_weight'],
        }),
        ('calibrated', {
            'alpha': CONFIG['calibration_alpha'],
            'diversity_weight': CONFIG['calibration_diversity_weight'],
        }),
        ('serendipity', {'beta': CONFIG['serendipity_beta']}),
        ('bounded_greedy', {'max_per_category': CONFIG['bounded_greedy_max_per_cat']}),
        ('max_coverage', {'coverage_budget': CONFIG['max_coverage_budget']}),
        # ★ Composite: single re-ranker explicitly addressing all 4 diversity dimensions
        ('composite', {
            'w_relevance':   CONFIG['composite_w_relevance'],
            'w_diversity':   CONFIG['composite_w_diversity'],
            'w_calibration': CONFIG['composite_w_calibration'],
            'w_serendipity': CONFIG['composite_w_serendipity'],
            'w_fairness':    CONFIG['composite_w_fairness'],
            'explore_weight': CONFIG['composite_explore_weight'],
        }),
    ]

    all_results = []

    for method_name, params in methods:
        result = evaluate_method(method_name, reranker, test_data, news_df, params)
        all_results.append(result)

    # Save results
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "diversity_evaluation.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison
    print_comparison(all_results)

    total_time = time.time() - t_start
    logger.info(f"\n✔ Phase 4 evaluation complete in {total_time/60:.1f} min")
    logger.info(f"\nResults saved to: {output_dir}/")


def print_comparison(results: List[Dict]):
    """Print side-by-side comparison table with all 8 methods and all metrics."""
    n_methods = len(results) - 1  # exclude baseline

    print("\n" + "=" * 110)
    print("ACCURACY vs DIVERSITY COMPARISON")
    print("=" * 110)

    header = (
        f"{'Method':<18} {'AUC':<7} {'NDCG@10':<9} {'MRR':<8} "
        f"{'Gini↓':<8} {'ILD↑':<8} {'Cover↑':<8} "
        f"{'Entropy↑':<10} {'CalErr↓':<9} {'PopFair↓':<9}"
    )
    print(f"\n{header}")
    print("-" * 110)

    for result in results:
        method = result['method']
        acc = result['accuracy']
        div = result['diversity']
        star = " ★" if method == 'composite' else ""

        print(
            f"{method + star:<18} "
            f"{acc.get('auc', 0):<7.4f} "
            f"{acc.get('ndcg@10', 0):<9.4f} "
            f"{acc.get('mrr', 0):<8.4f} "
            f"{div.get('avg_gini', 0):<8.4f} "
            f"{div.get('avg_ild', 0):<8.4f} "
            f"{div.get('avg_coverage', 0):<8.4f} "
            f"{div.get('avg_entropy', 0):<10.4f} "
            f"{div.get('avg_calibration_error', 0):<9.4f} "
            f"{div.get('avg_popularity_miscalibration', 0):<9.4f}"
        )

    print("=" * 110)
    print("  ★ = composite re-ranker (explicitly addresses all 4 diversity dimensions)")
    print("  PopFair↓ = |mean_pop(recs) − mean_pop(history)|  (lower = better fairness)")

    # Key insights
    print("\nKey Insights:")
    baseline = results[0]
    reranked = results[1:]

    best_gini = min(reranked, key=lambda x: x['diversity']['avg_gini'])
    best_cov  = max(reranked, key=lambda x: x['diversity']['avg_coverage'])
    best_ild  = max(reranked, key=lambda x: x['diversity']['avg_ild'])
    best_fair = min(reranked, key=lambda x: x['diversity'].get('avg_popularity_miscalibration', 1e9))

    def pct_change(old, new):
        return (old - new) / max(abs(old), 1e-9) * 100

    gini_imp = pct_change(
        baseline['diversity']['avg_gini'],
        best_gini['diversity']['avg_gini'],
    )
    cov_imp = (
        best_cov['diversity']['avg_coverage']
        - baseline['diversity']['avg_coverage']
    ) * 100
    ndcg_cost = pct_change(
        baseline['accuracy']['ndcg@10'],
        best_gini['accuracy']['ndcg@10'],
    )
    fair_imp = pct_change(
        baseline['diversity'].get('avg_popularity_miscalibration', 1.0),
        best_fair['diversity'].get('avg_popularity_miscalibration', 1.0),
    )

    print(f"  Best Gini reduction    : {best_gini['method']} ({gini_imp:.1f}% reduction)")
    print(f"  Best Coverage gain     : {best_cov['method']} (+{cov_imp:.1f}pp absolute)")
    print(f"  Best ILD               : {best_ild['method']} ({best_ild['diversity']['avg_ild']:.4f})")
    print(f"  Best Pop. Fairness     : {best_fair['method']} ({fair_imp:.1f}% reduction)")
    print(f"  NDCG cost (best div)   : {ndcg_cost:.1f}%")

    ratio = gini_imp / max(abs(ndcg_cost), 0.01)
    print(f"  Trade-off ratio        : {ratio:.2f}× diversity gain per 1% accuracy loss")

    # Composite-specific summary
    composite_result = next((r for r in reranked if r['method'] == 'composite'), None)
    if composite_result:
        c_div = composite_result['diversity']
        c_acc = composite_result['accuracy']
        base_div = baseline['diversity']
        base_acc = baseline['accuracy']
        print(f"\n  Composite re-ranker vs baseline:")
        print(f"    NDCG@10  : {base_acc['ndcg@10']:.4f} → {c_acc['ndcg@10']:.4f}"
              f"  ({pct_change(base_acc['ndcg@10'], c_acc['ndcg@10']):.1f}%)")
        print(f"    Gini     : {base_div['avg_gini']:.4f} → {c_div['avg_gini']:.4f}"
              f"  ({pct_change(base_div['avg_gini'], c_div['avg_gini']):.1f}%)")
        print(f"    Coverage : {base_div['avg_coverage']:.4f} → {c_div['avg_coverage']:.4f}"
              f"  (+{(c_div['avg_coverage'] - base_div['avg_coverage'])*100:.1f}pp)")
        print(f"    ILD      : {base_div['avg_ild']:.4f} → {c_div['avg_ild']:.4f}")
        print(f"    Entropy  : {base_div['avg_entropy']:.4f} → {c_div['avg_entropy']:.4f}")
        print(f"    CalErr   : {base_div['avg_calibration_error']:.4f} → {c_div['avg_calibration_error']:.4f}")
        print(f"    PopFair  : {base_div.get('avg_popularity_miscalibration', 0):.4f}"
              f" → {c_div.get('avg_popularity_miscalibration', 0):.4f}")

    print("\nTarget thresholds (pass/fail):")
    gini_pass = sum(1 for r in reranked if r['diversity']['avg_gini'] < 0.6)
    cov_pass  = sum(1 for r in reranked if r['diversity']['avg_coverage'] > 0.25)
    ild_pass  = sum(1 for r in reranked if r['diversity']['avg_ild'] > 0.45)
    auc_pass  = sum(1 for r in reranked if r['accuracy']['auc'] >= 0.55)

    print(f"  Gini < 0.6   : {gini_pass}/{n_methods} methods {'✔' if gini_pass >= 3 else '✗'} (target: ≥3)")
    print(f"  Coverage >25%: {cov_pass}/{n_methods} methods {'✔' if cov_pass >= 2 else '✗'} (target: ≥2)")
    print(f"  ILD > 0.45   : {ild_pass}/{n_methods} methods {'✔' if ild_pass >= 2 else '✗'} (target: ≥2)")
    print(f"  AUC ≥ 0.55   : {auc_pass}/{n_methods} methods {'✔' if auc_pass >= 5 else '✗'} (target: ≥5)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
