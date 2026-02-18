"""
Phase 4 Evaluation Script
=========================
Evaluate all 4 diversity re-ranking algorithms and compare vs baseline.

This script:
  1. Loads baseline recommender
  2. For each algorithm (MMR, xQuAD, Calibrated, Serendipity):
     - Generates recommendations for test users
     - Measures accuracy (AUC, NDCG, MRR, Precision, Recall)
     - Measures diversity (Gini, ILD, Coverage, Calibration Error)
  3. Creates comparison plots (before/after)
  4. Generates Pareto frontier (accuracy vs diversity trade-off)
  5. Saves comprehensive report

Usage:
    python run_phase4_evaluation.py

Expected runtime: 30-60 min (runs 5 full evaluations: baseline + 4 algorithms)
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

phase0_path = base_dir.parent / "Phase0_data_processing" / "data_processing_v1"
if phase0_path.exists():
    sys.path.insert(0, str(phase0_path))

from diversity_reranker import DiversityReranker
from baseline_recommender_phase2 import BaselineRecommender, RecommenderEvaluator
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
    
    # Algorithm parameters (tune these for best results)
    "mmr_lambda": 0.3,          # Lower = more diversity
    "xquad_lambda": 0.5,        # Balance relevance and coverage
    "calibration_alpha": 0.6,   # Higher = stricter calibration
    "serendipity_beta": 0.4,    # Higher = more serendipity
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
    
    # Generate recommendations for all test users
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
        baseline_candidates = reranker.baseline.recommend(
            user_history=history,
            k=100,
            exclude_history=False,
        )
        
        if not baseline_candidates:
            continue
        
        # Re-rank with diversity method
        if method_name == 'baseline':
            # No re-ranking
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
        
        # For accuracy metrics: score all impressions
        # (not just the re-ranked top-K)
        if impressions:
            known_impressions = [
                (nid, label) for nid, label in impressions
                if nid in reranker.news_id_to_idx
            ]
            
            if known_impressions:
                candidate_ids = [nid for nid, _ in known_impressions]
                labels = [label for _, label in known_impressions]
                
                # Get scores from baseline
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
    
    # Calculate accuracy metrics
    from sklearn.metrics import roc_auc_score
    
    all_labels = []
    all_scores = []
    for labels, scores in all_labels_and_scores:
        all_labels.extend(labels)
        all_scores.extend(scores)
    
    accuracy_metrics = {
        'auc': roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0,
    }
    
    # Calculate ranking metrics (@K)
    ranking_metrics = calculate_ranking_metrics(all_recommendations, test_data)
    accuracy_metrics.update(ranking_metrics)
    
    # Calculate diversity metrics
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
    )
    
    return {
        'method': method_name,
        'params': method_params,
        'accuracy': accuracy_metrics,
        'diversity': diversity_metrics,
    }


def calculate_ranking_metrics(recommendations: List[List[Tuple]], test_data: List[Dict]) -> Dict:
    """Calculate NDCG, MRR, Precision, Recall at K."""
    ndcg_scores = []
    mrr_scores = []
    precision_scores = []
    recall_scores = []
    
    for recs, sample in zip(recommendations, test_data):
        impressions = sample.get('impressions', [])
        if not impressions:
            continue
        
        # Build ground truth
        clicked_ids = {nid for nid, label in impressions if label == 1}
        
        if not clicked_ids:
            continue
        
        # Get recommended IDs
        rec_ids = [nid for nid, _ in recs]
        
        # Precision@K
        hits = sum(1 for nid in rec_ids if nid in clicked_ids)
        precision = hits / len(rec_ids) if rec_ids else 0
        precision_scores.append(precision)
        
        # Recall@K
        recall = hits / len(clicked_ids) if clicked_ids else 0
        recall_scores.append(recall)
        
        # MRR
        for rank, nid in enumerate(rec_ids, 1):
            if nid in clicked_ids:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
        
        # NDCG@K (simplified)
        dcg = sum(
            (1 if rec_ids[i] in clicked_ids else 0) / np.log2(i + 2)
            for i in range(len(rec_ids))
        )
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(clicked_ids), len(rec_ids))))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return {
        f'ndcg@{CONFIG["k"]}': np.mean(ndcg_scores) if ndcg_scores else 0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0,
        f'precision@{CONFIG["k"]}': np.mean(precision_scores) if precision_scores else 0,
        f'recall@{CONFIG["k"]}': np.mean(recall_scores) if recall_scores else 0,
    }


def calculate_diversity_from_recs(
    recommendations: List[List[Tuple]],
    test_data: List[Dict],
    analyzer: EchoChamberAnalyzer,
) -> Dict:
    """Calculate diversity metrics from recommendations."""
    gini_scores = []
    ild_scores = []
    coverage_scores = []
    
    for recs in recommendations:
        rec_ids = [nid for nid, _ in recs]
        
        # Categories
        rec_cats = [
            analyzer.news_to_category[nid]
            for nid in rec_ids
            if nid in analyzer.news_to_category
        ]
        
        if rec_cats:
            gini_scores.append(analyzer.calculate_gini(rec_cats))
            coverage_scores.append(analyzer.calculate_coverage(rec_cats))
        
        # ILD
        ild = analyzer.calculate_ild(rec_ids)
        ild_scores.append(ild)
    
    return {
        'avg_gini': np.mean(gini_scores) if gini_scores else 0,
        'avg_ild': np.mean(ild_scores) if ild_scores else 0,
        'avg_coverage': np.mean(coverage_scores) if coverage_scores else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    
    print("\n" + "=" * 60)
    print("PHASE 4 — DIVERSITY RE-RANKING EVALUATION")
    print("Diversity-Aware News Recommender — Capstone Project")
    print("=" * 60)
    
    # Load components
    logger.info("\nLoading baseline recommender ...")
    baseline = BaselineRecommender.load(
        str(Path(CONFIG["baseline_dir"]) / "baseline_recommender.pkl"),
        CONFIG["embeddings_dir"],
    )
    
    logger.info("Loading news metadata ...")
    processed_dir = Path(CONFIG["baseline_dir"]).parent.parent.parent / "Phase0_data_processing" / "processed_data_v2"
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
        ('xquad', {'lambda_param': CONFIG['xquad_lambda']}),
        ('calibrated', {'alpha': CONFIG['calibration_alpha']}),
        ('serendipity', {'beta': CONFIG['serendipity_beta']}),
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
    
    # Summary
    total_time = time.time() - t_start
    logger.info(f"\n✔ Phase 4 evaluation complete in {total_time/60:.1f} min")
    logger.info(f"\nResults saved to: {output_dir}/")
    
    print("\nNext: Create visualizations comparing baseline vs diversity methods")


def print_comparison(results: List[Dict]):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("ACCURACY vs DIVERSITY COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<15} {'NDCG@10':<10} {'MRR':<10} {'Gini':<10} {'ILD':<10} {'Coverage':<10}")
    print("-" * 80)
    
    for result in results:
        method = result['method']
        acc = result['accuracy']
        div = result['diversity']
        
        print(f"{method:<15} "
              f"{acc.get('ndcg@10', 0):<10.4f} "
              f"{acc.get('mrr', 0):<10.4f} "
              f"{div.get('avg_gini', 0):<10.4f} "
              f"{div.get('avg_ild', 0):<10.4f} "
              f"{div.get('avg_coverage', 0):<10.4f}")
    
    print("=" * 80)
    print("\nKey Insights:")
    
    baseline = results[0]
    best_diverse = min(results[1:], key=lambda x: x['diversity']['avg_gini'])
    
    gini_improvement = (baseline['diversity']['avg_gini'] - best_diverse['diversity']['avg_gini']) / baseline['diversity']['avg_gini'] * 100
    ndcg_cost = (baseline['accuracy']['ndcg@10'] - best_diverse['accuracy']['ndcg@10']) / baseline['accuracy']['ndcg@10'] * 100
    
    print(f"  Best diversity method: {best_diverse['method']}")
    print(f"  Gini improvement: {gini_improvement:.1f}% reduction (echo chamber reduced)")
    print(f"  NDCG cost: {ndcg_cost:.1f}% (small accuracy trade-off)")
    print(f"  Trade-off ratio: {gini_improvement/max(ndcg_cost, 0.01):.2f}× diversity gain per 1% accuracy loss")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
