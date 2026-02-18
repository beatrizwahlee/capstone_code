"""
Phase 2: Baseline Recommender System
=====================================
Diversity-Aware News Recommender System — Capstone Project

This is the accuracy-optimized baseline that Phase 4 diversity methods
will be compared against.

Architecture:
  Input:  User click history (list of news IDs)
  
  Step 1: Load Phase 1 embeddings from disk (~2 seconds)
  
  Step 2: Build user profile using AttentionUserProfiler
          (recency-weighted, content-aware aggregation)
  
  Step 3: Retrieve top-100 candidates from FAISS index (<1 ms)
  
  Step 4: Hybrid scoring:
          score = 0.6 × content_sim + 0.2 × popularity + 0.2 × recency
  
  Step 5: Return top-K recommendations

Evaluation:
  - AUC (Area Under ROC)
  - MRR (Mean Reciprocal Rank)
  - Precision@K, Recall@K, F1@K, NDCG@K for K ∈ {5, 10, 20}
  - Diversity metrics (Gini, ILD, Coverage) for comparison

Expected Performance:
  AUC:        0.65–0.70
  NDCG@10:    0.35–0.40
  MRR:        0.30–0.35

Usage:
    from baseline_recommender_phase2 import BaselineRecommender
    
    # One-time load
    recommender = BaselineRecommender.from_embeddings('./embeddings')
    
    # At recommendation time
    recs = recommender.recommend(
        user_history=['N123', 'N456', ...],
        k=10,
        exclude_history=True,
    )
    # Returns: [('N789', 0.856), ('N012', 0.832), ...]
"""

import json
import logging
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load Phase 1 outputs
# ---------------------------------------------------------------------------

def load_phase1_embeddings(embeddings_dir: str):
    """
    Load all Phase 1 outputs from disk.
    
    Returns:
        dict with keys: final_embeddings, news_ids, news_id_to_idx,
                        category_layer, retriever
    """
    from nlp_encoder import NewsEncoderPhase1
    
    logger.info(f"Loading Phase 1 embeddings from {embeddings_dir} ...")
    encoder = NewsEncoderPhase1.load(embeddings_dir)
    
    return {
        'final_embeddings': encoder.final_embeddings,
        'news_ids': encoder.news_ids,
        'news_id_to_idx': encoder.news_id_to_idx,
        'category_layer': encoder.category_layer,
        'retriever': encoder.retriever,
        'encoder': encoder,  # keep full encoder for save/load
    }


# ---------------------------------------------------------------------------
# Baseline Recommender
# ---------------------------------------------------------------------------

class BaselineRecommender:
    """
    Accuracy-optimized baseline recommender.
    
    Uses Phase 1 embeddings + hybrid scoring (content + popularity + recency).
    """
    
    def __init__(
        self,
        embeddings_dict: dict,
        content_weight: float = 0.6,
        popularity_weight: float = 0.2,
        recency_weight: float = 0.2,
        decay_lambda: float = 0.3,
    ):
        """
        Args:
            embeddings_dict:    Output from load_phase1_embeddings()
            content_weight:     Weight for content similarity (cosine)
            popularity_weight:  Weight for popularity score
            recency_weight:     Weight for recency score
            decay_lambda:       Recency decay rate for user profiler
        """
        self.encoder = embeddings_dict['encoder']
        self.final_embeddings = embeddings_dict['final_embeddings']
        self.news_ids = embeddings_dict['news_ids']
        self.news_id_to_idx = embeddings_dict['news_id_to_idx']
        self.retriever = embeddings_dict['retriever']
        
        # Hybrid scoring weights
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.recency_weight = recency_weight
        
        # User profiler
        from nlp_encoder import AttentionUserProfiler
        self.profiler = AttentionUserProfiler(decay_lambda=decay_lambda)
        
        # Popularity scores (will be computed from training data)
        self.popularity_scores: Dict[str, float] = {}
        
        # Recency scores (days since publication — needs news metadata)
        self.recency_scores: Dict[str, float] = {}
        
        # News metadata (for diversity metrics)
        self.news_metadata: Optional[pd.DataFrame] = None
    
    @classmethod
    def from_embeddings(cls, embeddings_dir: str, **kwargs) -> "BaselineRecommender":
        """Load Phase 1 embeddings and create recommender."""
        embeddings_dict = load_phase1_embeddings(embeddings_dir)
        return cls(embeddings_dict, **kwargs)
    
    def fit_popularity(
        self,
        train_interactions: List[Dict],
        news_df: Optional[pd.DataFrame] = None,
    ):
        """
        Compute popularity scores from training click data.
        
        Args:
            train_interactions: List of dicts with keys:
                                'news_id', 'label' (1=click, 0=no-click)
            news_df:            Optional news metadata (for recency scores)
        """
        logger.info("Computing popularity scores from training data ...")
        
        click_counts = Counter()
        impression_counts = Counter()
        
        for interaction in train_interactions:
            news_id = interaction['news_id']
            label = interaction['label']
            
            impression_counts[news_id] += 1
            if label == 1:
                click_counts[news_id] += 1
        
        # CTR-based popularity
        max_clicks = max(click_counts.values()) if click_counts else 1
        self.popularity_scores = {}
        
        for news_id in self.news_ids:
            clicks = click_counts.get(news_id, 0)
            impressions = impression_counts.get(news_id, 1)
            
            # Combine raw click count with CTR
            ctr = clicks / impressions if impressions > 0 else 0
            raw_pop = clicks / max_clicks
            
            # Weighted combination
            self.popularity_scores[news_id] = 0.7 * raw_pop + 0.3 * ctr
        
        logger.info(f"  Popularity scores computed for {len(self.popularity_scores):,} articles")
        logger.info(f"  Articles with clicks: {len(click_counts):,}")
        
        # Store news metadata if provided
        if news_df is not None:
            self.news_metadata = news_df
            logger.info(f"  News metadata loaded: {len(news_df):,} articles")
    
    def recommend(
        self,
        user_history: List[str],
        k: int = 10,
        exclude_history: bool = True,
        candidates: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            user_history:      List of clicked news IDs (chronological)
            k:                 Number of recommendations
            exclude_history:   Remove history items from recommendations
            candidates:        Optional pre-filtered candidate list
                               (if None, retrieves top-100 from FAISS)
        
        Returns:
            List of (news_id, score) tuples, sorted by score descending
        """
        if not user_history:
            # Cold start: return most popular items
            return self._cold_start_recommend(k)
        
        # Build user profile
        user_profile = self.profiler.build_profile(
            history_ids=user_history,
            news_embeddings=self.final_embeddings,
            news_id_to_idx=self.news_id_to_idx,
            max_history=50,
        )
        
        # Retrieve candidates
        if candidates is None:
            # Use FAISS to get top-100 by content similarity
            exclude_ids = user_history if exclude_history else None
            candidate_ids, content_scores = self.retriever.retrieve(
                query_vector=user_profile,
                k=100,
                exclude_ids=exclude_ids,
            )
        else:
            candidate_ids = [
                cid for cid in candidates
                if cid in self.news_id_to_idx  # Filter unknown articles
                and ((not exclude_history) or (cid not in user_history))
            ]
            
            if not candidate_ids:
                # All candidates filtered out — return empty
                return []
            
            # Score them
            candidate_embeddings = np.array([
                self.final_embeddings[self.news_id_to_idx[cid]]
                for cid in candidate_ids
            ])
            content_scores = (candidate_embeddings @ user_profile).astype(np.float32)
        
        # Hybrid scoring
        final_scores = []
        for i, news_id in enumerate(candidate_ids):
            content_sim = float(content_scores[i])
            popularity = self.popularity_scores.get(news_id, 0.0)
            recency = self.recency_scores.get(news_id, 0.5)  # default mid-range
            
            score = (
                self.content_weight * content_sim +
                self.popularity_weight * popularity +
                self.recency_weight * recency
            )
            final_scores.append((news_id, score))
        
        # Sort and return top-K
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:k]
    
    def _cold_start_recommend(self, k: int) -> List[Tuple[str, float]]:
        """Fallback for users with no history: return most popular."""
        items = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        return items
    
    def save(self, filepath: str):
        """Save recommender state to disk."""
        state = {
            'content_weight': self.content_weight,
            'popularity_weight': self.popularity_weight,
            'recency_weight': self.recency_weight,
            'popularity_scores': self.popularity_scores,
            'recency_scores': self.recency_scores,
            'profiler_lambda': self.profiler.decay_lambda,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Recommender saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, embeddings_dir: str) -> "BaselineRecommender":
        """Load recommender state from disk."""
        embeddings_dict = load_phase1_embeddings(embeddings_dir)
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        obj = cls(
            embeddings_dict,
            content_weight=state['content_weight'],
            popularity_weight=state['popularity_weight'],
            recency_weight=state['recency_weight'],
            decay_lambda=state['profiler_lambda'],
        )
        obj.popularity_scores = state['popularity_scores']
        obj.recency_scores = state.get('recency_scores', {})
        
        logger.info(f"Recommender loaded from {filepath}")
        return obj


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class RecommenderEvaluator:
    """
    Comprehensive evaluation for the baseline recommender.
    """
    
    def __init__(self, recommender: BaselineRecommender, news_df: pd.DataFrame):
        self.recommender = recommender
        self.news_df = news_df
        self.news_categories = dict(zip(news_df['news_id'], news_df['category']))
    
    def evaluate(
        self,
        test_data: List[Dict],
        k_values: List[int] = [5, 10, 20],
    ) -> Dict:
        """
        Evaluate on test impressions.
        
        Args:
            test_data:  List of dicts with keys:
                        - user_id
                        - history (list of news IDs)
                        - impressions (list of (news_id, label) tuples)
            k_values:   List of K values for ranking metrics
        
        Returns:
            Dict with all evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_data):,} test impressions ...")
        t0 = time.time()
        
        all_labels = []
        all_scores = []
        
        metrics_at_k = defaultdict(list)
        
        for sample in test_data:
            history = sample['history']
            impressions = sample['impressions']
            
            if not impressions:
                continue
            
            # Filter impressions to only known articles
            known_impressions = [
                (nid, label) for nid, label in impressions
                if nid in self.recommender.news_id_to_idx
            ]
            
            if not known_impressions:
                # Skip this impression — no known candidates
                continue
            
            candidate_ids = [nid for nid, _ in known_impressions]
            labels = np.array([label for _, label in known_impressions])
            
            # Get recommendations (returns ALL candidates, scored)
            recs = self.recommender.recommend(
                user_history=history,
                k=len(candidate_ids),
                candidates=candidate_ids,
                exclude_history=False,  # Don't exclude for eval
            )
            
            # Build score array in the same order as impressions
            rec_dict = dict(recs)
            scores = np.array([rec_dict.get(nid, 0.0) for nid in candidate_ids])
            
            # Collect for AUC
            all_labels.extend(labels)
            all_scores.extend(scores)
            
            # Per-impression metrics at each K
            for k in k_values:
                top_k_indices = np.argsort(scores)[-k:][::-1]
                top_k_labels = labels[top_k_indices]
                
                precision = top_k_labels.sum() / k if k > 0 else 0
                recall = top_k_labels.sum() / labels.sum() if labels.sum() > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                ndcg = self._ndcg(labels, scores, k)
                
                metrics_at_k[f'precision@{k}'].append(precision)
                metrics_at_k[f'recall@{k}'].append(recall)
                metrics_at_k[f'f1@{k}'].append(f1)
                metrics_at_k[f'ndcg@{k}'].append(ndcg)
            
            # MRR
            sorted_indices = np.argsort(scores)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if labels[idx] == 1:
                    metrics_at_k['mrr'].append(1.0 / rank)
                    break
            else:
                metrics_at_k['mrr'].append(0.0)
        
        # Aggregate metrics
        results = {
            'auc': roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0,
            'mrr': np.mean(metrics_at_k['mrr']),
        }
        
        for k in k_values:
            results[f'precision@{k}'] = np.mean(metrics_at_k[f'precision@{k}'])
            results[f'recall@{k}'] = np.mean(metrics_at_k[f'recall@{k}'])
            results[f'f1@{k}'] = np.mean(metrics_at_k[f'f1@{k}'])
            results[f'ndcg@{k}'] = np.mean(metrics_at_k[f'ndcg@{k}'])
        
        elapsed = time.time() - t0
        logger.info(f"Evaluation complete in {elapsed:.1f}s")
        
        return results
    
    def _ndcg(self, labels: np.ndarray, scores: np.ndarray, k: int) -> float:
        """Compute NDCG@K."""
        sorted_indices = np.argsort(scores)[-k:][::-1]
        dcg = sum(
            labels[idx] / np.log2(rank + 2)
            for rank, idx in enumerate(sorted_indices)
        )
        
        ideal_labels = sorted(labels, reverse=True)[:k]
        idcg = sum(
            label / np.log2(rank + 2)
            for rank, label in enumerate(ideal_labels)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_diversity_metrics(
        self,
        recommendations: List[List[Tuple[str, float]]],
    ) -> Dict:
        """
        Calculate diversity metrics (for comparison with Phase 4).
        
        Args:
            recommendations: List of recommendation lists
                             (one per user, each is list of (news_id, score))
        
        Returns:
            Dict with diversity metrics
        """
        logger.info("Calculating diversity metrics ...")
        
        gini_scores = []
        coverage_scores = []
        
        all_categories = []
        
        for recs in recommendations:
            rec_ids = [nid for nid, _ in recs]
            categories = [
                self.news_categories.get(nid)
                for nid in rec_ids
                if nid in self.news_categories
            ]
            
            if categories:
                # Category diversity (unique / total)
                unique_cats = len(set(categories))
                coverage_scores.append(unique_cats)
                
                # Gini coefficient for this user's recommendations
                cat_counts = Counter(categories)
                gini = self._gini(list(cat_counts.values()))
                gini_scores.append(gini)
                
                all_categories.extend(categories)
        
        # Overall Gini across all recommendations
        overall_cat_counts = Counter(all_categories)
        overall_gini = self._gini(list(overall_cat_counts.values()))
        
        return {
            'avg_coverage': np.mean(coverage_scores),
            'avg_gini': np.mean(gini_scores),
            'overall_gini': overall_gini,
            'unique_categories_recommended': len(set(all_categories)),
        }
    
    def _gini(self, values: List[float]) -> float:
        """Compute Gini coefficient."""
        if not values:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n


# ---------------------------------------------------------------------------
# Convenience function for quick testing
# ---------------------------------------------------------------------------

def quick_test():
    """Quick test to verify the baseline recommender loads and runs."""
    logger.info("=" * 60)
    logger.info("Quick Test — Baseline Recommender")
    logger.info("=" * 60)
    
    # Load
    recommender = BaselineRecommender.from_embeddings('./embeddings')
    
    # Mock user history
    mock_history = recommender.news_ids[:10]
    
    logger.info(f"\nMock user history: {len(mock_history)} articles")
    
    # Generate recommendations
    recs = recommender.recommend(mock_history, k=10)
    
    logger.info(f"\nTop-10 recommendations:")
    for rank, (news_id, score) in enumerate(recs, 1):
        logger.info(f"  {rank}. {news_id}  (score: {score:.4f})")
    
    logger.info("\n✔ Baseline recommender is working!")


if __name__ == "__main__":
    quick_test()
