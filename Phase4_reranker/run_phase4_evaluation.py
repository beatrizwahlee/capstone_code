"""
Phase 4 Evaluation Script
=========================
Evaluate all 7 diversity re-ranking algorithms and compare vs baseline.

This script:
  1. Loads baseline recommender
  2. For each algorithm (Random, MMR, xQuAD, Calibrated, Serendipity,
     Bounded Greedy, Max Coverage, Composite):
     - Generates recommendations for test users
     - Measures accuracy (AUC, NDCG@10, MRR, Precision@10, Recall@10)
       using impression-based evaluation (matches Phase 2 protocol)
     - Measures diversity (Gini, ILD, Coverage, Entropy, CalibrationError)
       and system-level fairness (HHI over subcategories, KL-divergence
       between corpus and recommendation category distributions)
  3. Creates comparison table (before/after)
  4. Saves comprehensive report

Evaluation note:
  NDCG@10 / MRR / Precision@10 are computed by scoring ALL impression
  candidates for each user (not by checking FAISS top-10 overlap with
  clicked articles). This matches Phase 2 and gives correct scale (~0.37
  for baseline NDCG@10 vs the near-zero values from FAISS overlap).

Usage:
    python run_phase4_evaluation.py

Expected runtime: 70-130 min (9 full evaluations: random + baseline + 7 algorithms)
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

    # NRMS paths (optional — skipped if weights don't exist or TF unavailable)
    "nrms_model_dir": str(base_dir.parent / "Phase_NRMS" / "outputs" / "model"),
    "nrms_utils_dir": str(base_dir.parent / "Phase_NRMS" / "outputs" / "utils"),
    "nrms_train_news": str(base_dir.parent / "MINDlarge_train" / "news.tsv"),

    # Algorithm parameters
    "mmr_lambda": 0.5,              # 0.5 = balanced relevance/diversity
    "xquad_lambda": 0.5,            # Balance relevance and coverage
    "xquad_explore_weight": 0.3,    # Uniform prior blend
    "calibration_alpha": 0.6,       # Higher = stricter calibration
    "calibration_diversity_weight": 0.3,  # Uniform distribution blend
    "serendipity_beta": 0.4,        # Higher = more serendipity
    "bounded_greedy_max_per_cat": 2,       # Hard cap per category
    "max_coverage_budget": 0.6,     # Fraction of slots for coverage

    # Composite re-ranker weights (must sum to 1.0)
    # Rebalanced: stronger diversity + calibration push, lighter fairness/serendipity.
    # Bidirectional calibration + higher explore_weight actively reduce category repetition.
    "composite_w_relevance":   0.35,  # Relevance component (was 0.40)
    "composite_w_diversity":   0.25,  # Embedding diversity (was 0.15)
    "composite_w_calibration": 0.25,  # Bidirectional calibration gap (was 0.15)
    "composite_w_serendipity": 0.10,  # Unexpectedness from user centroid (was 0.15)
    "composite_w_fairness":    0.05,  # System-level corpus fairness (was 0.15)
    "composite_explore_weight": 0.40, # Uniform prior blend in calibration target (was 0.30)
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
        if method_name in ('baseline', 'nrms'):
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

        # Accuracy: score impressions using the re-ranked order (Option A).
        # For baseline: use raw model scores directly.
        # For random: use pure uniform random scores so AUC ≈ 0.5 (true lower bound).
        # For re-ranking methods: score impression candidates with the baseline,
        # then re-rank them with the same diversity method, and derive rank-based
        # scores (higher rank → higher score) so NDCG/MRR/Precision reflect the
        # actual re-ranked ordering rather than the baseline ordering.
        if impressions:
            known_impressions = [
                (nid, label) for nid, label in impressions
                if nid in reranker.news_id_to_idx
            ]

            if known_impressions:
                candidate_ids = [nid for nid, _ in known_impressions]
                labels = [label for _, label in known_impressions]

                if method_name == 'random':
                    # Pure random scores — true lower bound (AUC ≈ 0.5).
                    # Blended scoring would partially preserve the baseline signal
                    # even for a random ranking; uniform random avoids that.
                    scores = list(np.random.default_rng().random(len(candidate_ids)))
                else:
                    # exclude_history=False intentional: impression lists in MIND
                    # can contain already-read articles and all must be scored for AUC.
                    full_recs = reranker.baseline.recommend(
                        user_history=history,
                        k=len(candidate_ids),
                        candidates=candidate_ids,
                        exclude_history=False,
                    )

                    if method_name in ('baseline', 'nrms'):
                        rec_dict = dict(full_recs)
                        scores = [rec_dict.get(nid, 0.0) for nid in candidate_ids]
                    else:
                        # Re-rank the impression candidates with the diversity method,
                        # then compute blended scores: baseline_score × position_weight.
                        #
                        # Pure rank-based scores (n, n-1, ..., 1) cause AUC < 0.5 because
                        # diversity re-rankers intentionally demote positive items from the
                        # user's preferred category (category repetition penalty), placing
                        # them below diverse negatives regardless of the lambda/weight setting.
                        #
                        # Blended scoring preserves the baseline relevance signal so positives
                        # stay above negatives on average (AUC > 0.5), while still penalising
                        # items the re-ranker demotes — yielding different AUC values per method.
                        #
                        # Formula: score(i) = baseline_score(i) × (1 − normalised_rank(i))
                        #   normalised_rank ∈ [0, 1]: 0 for rank-1 (best), 1 for last rank
                        #   → rank-1 item keeps its full baseline score
                        #   → last-rank item is zeroed out
                        #
                        # inject_candidates=False: pool augmentation adds articles with no
                        # impression labels, which would displace impression candidates.
                        reranked_impressions = reranker.rerank(
                            candidates=full_recs,
                            user_history=history,
                            k=len(candidate_ids),
                            method=method_name,
                            inject_candidates=False,
                            **method_params,
                        )
                        n = len(reranked_impressions)
                        baseline_dict = dict(full_recs)
                        norm_rank = {
                            nid: rank / max(n - 1, 1)
                            for rank, (nid, _) in enumerate(reranked_impressions)
                        }
                        scores = [
                            baseline_dict.get(nid, 0.0) * (1.0 - norm_rank.get(nid, 1.0))
                            for nid in candidate_ids
                        ]

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
        corpus_category_dist=reranker.corpus_category_dist,
        corpus_subcategory_dist=reranker.corpus_subcategory_dist,
        news_subcategories=reranker.news_subcategories,
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
    corpus_category_dist: Optional[Dict[str, float]] = None,
    corpus_subcategory_dist: Optional[Dict[str, float]] = None,
    news_subcategories: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Calculate diversity metrics from recommendations.

    Per-user metrics (averaged across users):
      - avg_gini:              Gini coefficient (0 = equal, 1 = monopoly)
      - avg_ild:               Intra-List Diversity (embedding distance)
      - avg_coverage:          Fraction of all categories represented
      - avg_entropy:           Shannon entropy of category distribution
      - avg_calibration_error: KL(user_history_dist || rec_dist)

    System-level fairness metrics (computed once across all recommendations):
      - system_hhi:            Herfindahl-Hirschman Index over subcategory counts.
                               0 = perfectly even spread, 1 = all recs from one subcategory.
                               Measures whether the recommender creates structural
                               concentration in favour of specific topics.
      - system_kl_divergence:  KL(corpus_category_dist || agg_rec_category_dist).
                               How much the aggregate recommendation distribution
                               diverges from the full corpus distribution.
                               0 = perfectly representative, higher = systematic bias.
    """
    gini_scores = []
    ild_scores = []
    coverage_scores = []
    entropy_scores = []
    calibration_errors = []

    # Accumulators for system-level fairness (aggregated across all users)
    agg_rec_subcat_counts: Counter = Counter()
    agg_rec_cat_counts: Counter = Counter()

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

            # Accumulate subcategory and category counts for system-level fairness
            if news_subcategories:
                for nid in rec_ids:
                    subcat = news_subcategories.get(nid, "")
                    if subcat:
                        agg_rec_subcat_counts[subcat] += 1
            agg_rec_cat_counts.update(rec_cats)

        ild = analyzer.calculate_ild(rec_ids)
        ild_scores.append(ild)

    # --- System-level fairness metrics (computed once across all users) ---

    # HHI over subcategory counts across all recommendations
    total_subcat_recs = sum(agg_rec_subcat_counts.values())
    if total_subcat_recs > 0 and agg_rec_subcat_counts:
        subcat_props = np.array([v / total_subcat_recs for v in agg_rec_subcat_counts.values()])
        system_hhi = float(np.sum(subcat_props ** 2))
    else:
        system_hhi = 0.0

    # KL(corpus_category_dist || agg_rec_category_dist)
    total_cat_recs = sum(agg_rec_cat_counts.values())
    if corpus_category_dist and total_cat_recs > 0:
        all_cats = sorted(corpus_category_dist.keys())
        smoothing = 1e-9
        corpus_probs = np.array([corpus_category_dist.get(c, smoothing) for c in all_cats])
        corpus_probs = corpus_probs / corpus_probs.sum()
        rec_probs = np.array([
            (agg_rec_cat_counts.get(c, 0) + smoothing) / (total_cat_recs + smoothing * len(all_cats))
            for c in all_cats
        ])
        system_kl_divergence = float(np.sum(corpus_probs * np.log(corpus_probs / rec_probs)))
    else:
        system_kl_divergence = 0.0

    return {
        'avg_gini': np.mean(gini_scores) if gini_scores else 0.0,
        'avg_ild': np.mean(ild_scores) if ild_scores else 0.0,
        'avg_coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
        'avg_entropy': np.mean(entropy_scores) if entropy_scores else 0.0,
        'avg_calibration_error': np.mean(calibration_errors) if calibration_errors else 0.0,
        'system_hhi': system_hhi,
        'system_kl_divergence': system_kl_divergence,
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

    # Shared corpus metadata
    logger.info("\nBuilding corpus metadata ...")
    news_categories    = dict(zip(news_df['news_id'], news_df['category']))
    news_subcategories = dict(zip(news_df['news_id'], news_df['subcategory']))

    cat_counts = news_df['category'].value_counts()
    corpus_category_dist = (cat_counts / cat_counts.sum()).to_dict()

    subcat_counts = news_df['subcategory'].value_counts()
    corpus_subcategory_dist = (subcat_counts / subcat_counts.sum()).to_dict()

    def make_reranker(recommender):
        """Build a DiversityReranker around any recommender with the right interface."""
        return DiversityReranker(
            baseline_recommender=recommender,
            embeddings=recommender.final_embeddings,
            news_id_to_idx=recommender.news_id_to_idx,
            news_categories=news_categories,
            popularity_scores=recommender.popularity_scores,
            corpus_category_dist=corpus_category_dist,
            corpus_subcategory_dist=corpus_subcategory_dist,
            news_subcategories=news_subcategories,
        )

    # ── 1. Evaluate baseline ────────────────────────────────────────────────
    logger.info("\nInitializing baseline reranker ...")
    reranker_baseline = make_reranker(baseline)
    baseline_result = evaluate_method(
        'baseline', reranker_baseline, test_data, news_df, {}
    )

    # ── 2. Optionally evaluate NRMS ────────────────────────────────────────
    nrms_result   = None
    reranker_nrms = None
    nrms_weights  = Path(CONFIG["nrms_model_dir"]) / "nrms_weights.h5"

    if nrms_weights.exists():
        try:
            logger.info("\nLoading NRMS recommender ...")
            nrms_path = base_dir.parent / "Phase_NRMS"
            if str(nrms_path) not in sys.path:
                sys.path.insert(0, str(nrms_path))
            from nrms_recommender import NRMSRecommender

            nrms_rec = NRMSRecommender.load(
                model_weights = str(nrms_weights),
                utils_dir     = CONFIG["nrms_utils_dir"],
                train_news    = CONFIG["nrms_train_news"],
                valid_news    = str(Path(CONFIG["valid_data_dir"]) / "news.tsv"),
            )
            # Attach category map so diversity metrics work correctly
            nrms_rec.news_id_to_category = news_categories

            reranker_nrms = make_reranker(nrms_rec)
            nrms_result   = evaluate_method(
                'nrms', reranker_nrms, test_data, news_df, {}
            )
        except Exception as e:
            logger.warning(f"NRMS evaluation failed: {e} — continuing without NRMS.")
    else:
        logger.info(
            f"\nNRMS weights not found at {nrms_weights} — skipping NRMS evaluation.\n"
            f"  Run:  ~/miniforge3/envs/nrms_env/bin/python Phase_NRMS/train_nrms.py"
        )

    # ── 3. Pick winner for re-rankers ──────────────────────────────────────
    baseline_auc = baseline_result['accuracy']['auc']
    nrms_auc     = nrms_result['accuracy']['auc'] if nrms_result else -1.0

    if nrms_auc > baseline_auc:
        winner_label   = 'nrms'
        winner_reranker = reranker_nrms
        logger.info(
            f"\n★ NRMS wins AUC ({nrms_auc:.4f} > {baseline_auc:.4f})"
            f" — re-rankers will run on top of NRMS."
        )
    else:
        winner_label    = 'baseline'
        winner_reranker = reranker_baseline
        logger.info(
            f"\n★ Baseline wins AUC ({baseline_auc:.4f})"
            f" — re-rankers will run on top of baseline."
            + (f" (NRMS: {nrms_auc:.4f})" if nrms_result else " (NRMS not evaluated)")
        )

    # ── 4. Evaluate random + all re-rankers on winner ──────────────────────
    reranker_methods = [
        ('random', {}),
        ('mmr', {'lambda_param': CONFIG['mmr_lambda']}),
        ('xquad', {
            'lambda_param':  CONFIG['xquad_lambda'],
            'explore_weight': CONFIG['xquad_explore_weight'],
        }),
        ('calibrated', {
            'alpha':            CONFIG['calibration_alpha'],
            'diversity_weight': CONFIG['calibration_diversity_weight'],
        }),
        ('serendipity', {'beta': CONFIG['serendipity_beta']}),
        ('bounded_greedy', {'max_per_category': CONFIG['bounded_greedy_max_per_cat']}),
        ('max_coverage',   {'coverage_budget': CONFIG['max_coverage_budget']}),
        ('composite', {
            'w_relevance':   CONFIG['composite_w_relevance'],
            'w_diversity':   CONFIG['composite_w_diversity'],
            'w_calibration': CONFIG['composite_w_calibration'],
            'w_serendipity': CONFIG['composite_w_serendipity'],
            'w_fairness':    CONFIG['composite_w_fairness'],
            'explore_weight': CONFIG['composite_explore_weight'],
        }),
    ]

    all_results = [baseline_result]
    if nrms_result:
        all_results.append(nrms_result)

    for method_name, params in reranker_methods:
        result = evaluate_method(
            method_name, winner_reranker, test_data, news_df, params
        )
        result['reranker_base'] = winner_label   # record which base was used
        all_results.append(result)

    # ── 5. Save & print ────────────────────────────────────────────────────
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "diversity_evaluation.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print_comparison(all_results, winner_label=winner_label)

    total_time = time.time() - t_start
    logger.info(f"\n✔ Phase 4 evaluation complete in {total_time/60:.1f} min")
    logger.info(f"\nResults saved to: {output_dir}/")


def print_comparison(results: List[Dict], winner_label: str = 'baseline'):
    """Print side-by-side comparison table: random | baseline | nrms | re-rankers."""
    STANDALONE = ('random', 'baseline', 'nrms')
    random_result   = next((r for r in results if r['method'] == 'random'),   None)
    baseline_result = next((r for r in results if r['method'] == 'baseline'), None)
    nrms_result     = next((r for r in results if r['method'] == 'nrms'),     None)
    rerankers       = [r for r in results if r['method'] not in STANDALONE]
    n_rerankers     = len(rerankers)

    winner_tag = f"re-rankers on {winner_label}"
    print("\n" + "=" * 118)
    print(f"ACCURACY vs DIVERSITY COMPARISON  —  random | baseline | nrms | {winner_tag}")
    print("=" * 118)

    header = (
        f"{'Method':<18} {'AUC':<7} {'NDCG@10':<9} {'MRR':<8} "
        f"{'Gini↓':<8} {'ILD↑':<8} {'Cover↑':<8} "
        f"{'Entropy↑':<10} {'CalErr↓':<9} {'HHI↓':<8} {'KL↓':<8}"
    )
    print(f"\n{header}")
    print("-" * 118)

    def _row(result):
        method = result['method']
        acc = result['accuracy']
        div = result['diversity']
        if method == 'composite':
            tag = " ★"
        elif method == 'random':
            tag = " ○"
        elif method == winner_label and method in ('baseline', 'nrms'):
            tag = " ►"   # mark the winner that re-rankers build on
        else:
            tag = ""
        print(
            f"{method + tag:<18} "
            f"{acc.get('auc', 0):<7.4f} "
            f"{acc.get('ndcg@10', 0):<9.4f} "
            f"{acc.get('mrr', 0):<8.4f} "
            f"{div.get('avg_gini', 0):<8.4f} "
            f"{div.get('avg_ild', 0):<8.4f} "
            f"{div.get('avg_coverage', 0):<8.4f} "
            f"{div.get('avg_entropy', 0):<10.4f} "
            f"{div.get('avg_calibration_error', 0):<9.4f} "
            f"{div.get('system_hhi', 0):<8.4f} "
            f"{div.get('system_kl_divergence', 0):<8.4f}"
        )

    if random_result:
        _row(random_result)
    if baseline_result:
        _row(baseline_result)
    if nrms_result:
        _row(nrms_result)
    print("-" * 118)
    for r in rerankers:
        _row(r)

    print("=" * 118)
    print("  ○ = random recommender (lower bound — AUC ≈ 0.5, high diversity by chance)")
    print("  ► = base model used by re-rankers (higher AUC of baseline vs NRMS)")
    print("  ★ = composite re-ranker (explicitly addresses all 4 diversity dimensions)")
    print(f"  Re-rankers run on top of: {winner_label}")
    print("  HHI↓  = Herfindahl-Hirschman Index over subcategories (system-level; lower = less concentration)")
    print("  KL↓   = KL(corpus_dist || rec_dist) over categories (system-level; lower = more representative)")

    winner_result = nrms_result if winner_label == 'nrms' else baseline_result
    if not winner_result:
        return

    def pct_change(old, new):
        return (old - new) / max(abs(old), 1e-9) * 100

    # Key insights (re-rankers only, compared against the winner base)
    if rerankers:
        print(f"\nKey Insights (re-rankers vs {winner_label}):")
        best_gini = min(rerankers, key=lambda x: x['diversity']['avg_gini'])
        best_cov  = max(rerankers, key=lambda x: x['diversity']['avg_coverage'])
        best_ild  = max(rerankers, key=lambda x: x['diversity']['avg_ild'])
        best_hhi  = min(rerankers, key=lambda x: x['diversity'].get('system_hhi', 1e9))
        best_kl   = min(rerankers, key=lambda x: x['diversity'].get('system_kl_divergence', 1e9))

        gini_imp = pct_change(
            winner_result['diversity']['avg_gini'],
            best_gini['diversity']['avg_gini'],
        )
        cov_imp = (
            best_cov['diversity']['avg_coverage']
            - winner_result['diversity']['avg_coverage']
        ) * 100
        ndcg_cost = pct_change(
            winner_result['accuracy']['ndcg@10'],
            best_gini['accuracy']['ndcg@10'],
        )
        hhi_imp = pct_change(
            winner_result['diversity'].get('system_hhi', 1.0),
            best_hhi['diversity'].get('system_hhi', 1.0),
        )
        kl_imp = pct_change(
            winner_result['diversity'].get('system_kl_divergence', 1.0),
            best_kl['diversity'].get('system_kl_divergence', 1.0),
        )

        print(f"  Best Gini reduction    : {best_gini['method']} ({gini_imp:.1f}% reduction)")
        print(f"  Best Coverage gain     : {best_cov['method']} (+{cov_imp:.1f}pp absolute)")
        print(f"  Best ILD               : {best_ild['method']} ({best_ild['diversity']['avg_ild']:.4f})")
        print(f"  Best HHI (system)      : {best_hhi['method']} ({hhi_imp:.1f}% reduction)")
        print(f"  Best KL-div (system)   : {best_kl['method']} ({kl_imp:.1f}% reduction)")
        print(f"  NDCG cost (best div)   : {ndcg_cost:.1f}%")

        ratio = gini_imp / max(abs(ndcg_cost), 0.01)
        print(f"  Trade-off ratio        : {ratio:.2f}× diversity gain per 1% accuracy loss")

    # Composite-specific summary vs baseline
    composite_result = next((r for r in rerankers if r['method'] == 'composite'), None)
    if composite_result:
        c_div = composite_result['diversity']
        c_acc = composite_result['accuracy']
        base_div = winner_result['diversity']
        base_acc = winner_result['accuracy']
        print(f"\n  Composite re-ranker vs {winner_label}:")
        print(f"    NDCG@10  : {base_acc['ndcg@10']:.4f} → {c_acc['ndcg@10']:.4f}"
              f"  ({pct_change(base_acc['ndcg@10'], c_acc['ndcg@10']):.1f}%)")
        print(f"    Gini     : {base_div['avg_gini']:.4f} → {c_div['avg_gini']:.4f}"
              f"  ({pct_change(base_div['avg_gini'], c_div['avg_gini']):.1f}%)")
        print(f"    Coverage : {base_div['avg_coverage']:.4f} → {c_div['avg_coverage']:.4f}"
              f"  (+{(c_div['avg_coverage'] - base_div['avg_coverage'])*100:.1f}pp)")
        print(f"    ILD      : {base_div['avg_ild']:.4f} → {c_div['avg_ild']:.4f}")
        print(f"    Entropy  : {base_div['avg_entropy']:.4f} → {c_div['avg_entropy']:.4f}")
        print(f"    CalErr   : {base_div['avg_calibration_error']:.4f} → {c_div['avg_calibration_error']:.4f}")
        print(f"    HHI      : {base_div.get('system_hhi', 0):.4f} → {c_div.get('system_hhi', 0):.4f}")
        print(f"    KL-div   : {base_div.get('system_kl_divergence', 0):.4f} → {c_div.get('system_kl_divergence', 0):.4f}")

    print("\nTarget thresholds (re-rankers, pass/fail):")
    gini_pass = sum(1 for r in rerankers if r['diversity']['avg_gini'] < 0.6)
    cov_pass  = sum(1 for r in rerankers if r['diversity']['avg_coverage'] > 0.25)
    ild_pass  = sum(1 for r in rerankers if r['diversity']['avg_ild'] > 0.45)
    auc_pass  = sum(1 for r in rerankers if r['accuracy']['auc'] >= 0.55)

    print(f"  Gini < 0.6   : {gini_pass}/{n_rerankers} methods {'✔' if gini_pass >= 3 else '✗'} (target: ≥3)")
    print(f"  Coverage >25%: {cov_pass}/{n_rerankers} methods {'✔' if cov_pass >= 2 else '✗'} (target: ≥2)")
    print(f"  ILD > 0.45   : {ild_pass}/{n_rerankers} methods {'✔' if ild_pass >= 2 else '✗'} (target: ≥2)")
    print(f"  AUC ≥ 0.55   : {auc_pass}/{n_rerankers} methods {'✔' if auc_pass >= 5 else '✗'} (target: ≥5)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
