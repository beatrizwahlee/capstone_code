"""
Phase 4: Diversity-Aware Re-ranking Algorithms
===============================================
Diversity-Aware News Recommender System — Capstone Project

This module implements 4 state-of-the-art diversity re-ranking algorithms
that fix the echo chamber problem identified in Phase 3.

The Pipeline:
  1. Baseline retrieves top-100 candidates (Phase 2)
  2. Re-ranking algorithm selects diverse top-K from those 100
  3. Evaluation compares accuracy + diversity vs baseline

Algorithms Implemented:
-----------------------

1. MMR (Maximal Marginal Relevance)
   Carbonell & Goldstein, 1998
   
   Balances relevance with dissimilarity to already-selected items.
   At each step, picks item that maximizes:
       λ × relevance - (1-λ) × max_similarity_to_selected
   
   Parameters: λ ∈ [0, 1]
     λ=1.0 → pure relevance (baseline)
     λ=0.5 → balanced
     λ=0.3 → high diversity

2. xQuAD (eXplicit Query Aspect Diversification)
   Santos et al., 2010
   
   Ensures category coverage proportional to user interests.
   Models user's category preferences from history, then selects
   items to cover those categories proportionally.
   
   Parameters: λ ∈ [0, 1]

3. Calibrated Re-ranking
   Steck, 2018
   
   Matches recommended category distribution to user's historical
   distribution. Minimizes KL divergence between P(cat|history)
   and P(cat|recommendations).
   
   Parameters: α ∈ [0, 1] (calibration strength)

4. Serendipity-Aware
   Ge et al., 2010 + custom
   
   Promotes items that are:
     - Relevant (high content similarity)
     - Unexpected (low popularity, distant from user's usual categories)
     - Novel (dissimilar to recent history)
   
   Parameters: β (serendipity weight)

Usage:
    from diversity_reranker import DiversityReranker
    
    reranker = DiversityReranker(
        baseline_recommender=recommender,
        embeddings=final_embeddings,
        news_id_to_idx=news_id_to_idx,
        news_categories=news_categories,
    )
    
    # Get top-100 from baseline
    candidates = recommender.recommend(user_history, k=100)
    
    # Re-rank with MMR
    diverse_recs = reranker.mmr_rerank(candidates, user_history, k=10, lambda_param=0.3)
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiversityReranker:
    """
    Implements 4 diversity-aware re-ranking algorithms.
    """
    
    def __init__(
        self,
        baseline_recommender,
        embeddings: np.ndarray,
        news_id_to_idx: Dict[str, int],
        news_categories: Dict[str, str],
        popularity_scores: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            baseline_recommender:  Trained BaselineRecommender from Phase 2
            embeddings:            Final 532-dim embeddings from Phase 1
            news_id_to_idx:        news_id → row index mapping
            news_categories:       news_id → category mapping
            popularity_scores:     news_id → popularity score (optional)
        """
        self.baseline = baseline_recommender
        self.embeddings = embeddings
        self.news_id_to_idx = news_id_to_idx
        self.news_categories = news_categories
        self.popularity_scores = popularity_scores or {}
        
        # Get all unique categories
        self.all_categories = sorted(set(news_categories.values()))
    
    # -----------------------------------------------------------------------
    # Algorithm 1: MMR (Maximal Marginal Relevance)
    # -----------------------------------------------------------------------
    
    def mmr_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        lambda_param: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        MMR re-ranking: balance relevance with diversity.
        
        At each step, select item i that maximizes:
            MMR(i) = λ × relevance(i) - (1-λ) × max_similarity(i, selected)
        
        Args:
            candidates:    List of (news_id, relevance_score) from baseline
            user_history:  User's click history
            k:             Number of items to select
            lambda_param:  Trade-off parameter
                           1.0 = pure relevance (baseline)
                           0.5 = balanced
                           0.0 = pure diversity
        
        Returns:
            Re-ranked list of (news_id, mmr_score) tuples
        """
        if not candidates:
            return []
        
        # Extract candidate IDs and scores
        candidate_ids = [nid for nid, _ in candidates]
        relevance_scores = {nid: score for nid, score in candidates}
        
        # Filter to known IDs
        candidate_ids = [nid for nid in candidate_ids if nid in self.news_id_to_idx]
        
        if not candidate_ids:
            return []
        
        # Get embeddings
        candidate_embs = {
            nid: self.embeddings[self.news_id_to_idx[nid]]
            for nid in candidate_ids
        }
        
        # Greedy selection
        selected = []
        remaining = set(candidate_ids)
        
        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None
            
            for nid in remaining:
                # Relevance component
                relevance = relevance_scores.get(nid, 0.0)
                
                # Diversity component (max similarity to already selected)
                if selected:
                    similarities = [
                        float(candidate_embs[nid] @ candidate_embs[sel_nid])
                        for sel_nid in selected
                        if sel_nid in candidate_embs
                    ]
                    max_sim = max(similarities) if similarities else 0.0
                else:
                    max_sim = 0.0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = nid
            
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
        
        # Return with scores
        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]
    
    # -----------------------------------------------------------------------
    # Algorithm 2: xQuAD (eXplicit Query Aspect Diversification)
    # -----------------------------------------------------------------------
    
    def xquad_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        lambda_param: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        xQuAD re-ranking: ensure category coverage proportional to user interests.
        
        Models P(category|user) from history, then selects items to cover
        those categories proportionally.
        
        Args:
            candidates:    List of (news_id, relevance_score) from baseline
            user_history:  User's click history
            k:             Number of items to select
            lambda_param:  Trade-off parameter (relevance vs coverage)
        
        Returns:
            Re-ranked list of (news_id, score) tuples
        """
        if not candidates:
            return []
        
        candidate_ids = [nid for nid, _ in candidates]
        relevance_scores = {nid: score for nid, score in candidates}
        
        # Build user's category distribution from history
        history_cats = [
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        ]
        
        if not history_cats:
            # No history categories — fall back to MMR
            return self.mmr_rerank(candidates, user_history, k, lambda_param)
        
        # P(category|user) from history
        cat_counts = Counter(history_cats)
        total = sum(cat_counts.values())
        p_cat_user = {cat: count / total for cat, count in cat_counts.items()}
        
        # Greedy selection
        selected = []
        remaining = set(candidate_ids)
        
        # Track category coverage
        selected_cats = Counter()
        
        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None
            
            for nid in remaining:
                if nid not in self.news_categories:
                    continue
                
                cat = self.news_categories[nid]
                
                # Relevance component
                relevance = relevance_scores.get(nid, 0.0)
                
                # Coverage component: P(cat|user) × (1 - current_coverage_of_cat)
                p_cat = p_cat_user.get(cat, 0.0)
                
                if selected:
                    # How much of this category's "quota" have we filled?
                    current_coverage = selected_cats.get(cat, 0) / len(selected)
                    coverage_gap = max(0, p_cat - current_coverage)
                else:
                    coverage_gap = p_cat
                
                # xQuAD score
                xquad_score = lambda_param * relevance + (1 - lambda_param) * coverage_gap
                
                if xquad_score > best_score:
                    best_score = xquad_score
                    best_item = nid
            
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
                selected_cats[self.news_categories[best_item]] += 1
        
        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]
    
    # -----------------------------------------------------------------------
    # Algorithm 3: Calibrated Re-ranking
    # -----------------------------------------------------------------------
    
    def calibrated_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        alpha: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Calibrated re-ranking: match recommended distribution to history.
        
        Minimizes KL divergence between:
          P(category|history) and P(category|recommendations)
        
        Args:
            candidates:    List of (news_id, relevance_score) from baseline
            user_history:  User's click history
            k:             Number of items to select
            alpha:         Calibration strength (0=ignore, 1=strict match)
        
        Returns:
            Re-ranked list of (news_id, score) tuples
        """
        if not candidates:
            return []
        
        candidate_ids = [nid for nid, _ in candidates]
        relevance_scores = {nid: score for nid, score in candidates}
        
        # Build target distribution from history
        history_cats = [
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        ]
        
        if not history_cats:
            return self.mmr_rerank(candidates, user_history, k, 0.5)
        
        # Target distribution (smoothed)
        cat_counts = Counter(history_cats)
        total = sum(cat_counts.values())
        target_dist = {
            cat: (cat_counts.get(cat, 0) + 0.01) / (total + 0.01 * len(self.all_categories))
            for cat in self.all_categories
        }
        
        # Greedy selection to minimize KL divergence
        selected = []
        remaining = set(candidate_ids)
        
        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None
            
            for nid in remaining:
                if nid not in self.news_categories:
                    continue
                
                # Simulate adding this item
                test_selected = selected + [nid]
                test_cats = [
                    self.news_categories[s]
                    for s in test_selected
                    if s in self.news_categories
                ]
                
                # Current distribution (smoothed)
                test_counts = Counter(test_cats)
                test_total = sum(test_counts.values())
                current_dist = {
                    cat: (test_counts.get(cat, 0) + 0.01) / (test_total + 0.01 * len(self.all_categories))
                    for cat in self.all_categories
                }
                
                # KL divergence: D_KL(target || current)
                kl_div = sum(
                    target_dist[cat] * np.log(target_dist[cat] / current_dist[cat])
                    for cat in self.all_categories
                    if target_dist[cat] > 0
                )
                
                # Score: balance relevance with calibration
                relevance = relevance_scores.get(nid, 0.0)
                calibration_score = -kl_div  # Negative because lower KL is better
                
                score = (1 - alpha) * relevance + alpha * calibration_score
                
                if score > best_score:
                    best_score = score
                    best_item = nid
            
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
        
        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]
    
    # -----------------------------------------------------------------------
    # Algorithm 4: Serendipity-Aware Re-ranking
    # -----------------------------------------------------------------------
    
    def serendipity_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        beta: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """
        Serendipity-aware re-ranking: promote unexpected-but-relevant items.
        
        Serendipitous items are:
          - Relevant (high baseline score)
          - Unexpected (low popularity, different category)
          - Novel (distant from recent history)
        
        Args:
            candidates:    List of (news_id, relevance_score) from baseline
            user_history:  User's click history
            k:             Number of items to select
            beta:          Serendipity weight
        
        Returns:
            Re-ranked list of (news_id, score) tuples
        """
        if not candidates:
            return []
        
        candidate_ids = [nid for nid, _ in candidates]
        relevance_scores = {nid: score for nid, score in candidates}
        
        # User's typical categories
        history_cats = [
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        ]
        typical_cats = set(history_cats) if history_cats else set()
        
        # Get recent history embeddings (for novelty)
        recent_history = user_history[-10:] if len(user_history) > 10 else user_history
        recent_embs = [
            self.embeddings[self.news_id_to_idx[nid]]
            for nid in recent_history
            if nid in self.news_id_to_idx
        ]
        
        # Score each candidate
        serendipity_scores = {}
        
        for nid in candidate_ids:
            if nid not in self.news_id_to_idx or nid not in self.news_categories:
                continue
            
            # Relevance (baseline score)
            relevance = relevance_scores.get(nid, 0.0)
            
            # Unexpectedness = low popularity + different category
            popularity = self.popularity_scores.get(nid, 0.5)
            unexpectedness_pop = 1.0 - popularity
            
            cat = self.news_categories[nid]
            unexpectedness_cat = 0.0 if cat in typical_cats else 1.0
            
            unexpectedness = 0.5 * unexpectedness_pop + 0.5 * unexpectedness_cat
            
            # Novelty = distance from recent history
            if recent_embs:
                candidate_emb = self.embeddings[self.news_id_to_idx[nid]]
                similarities = [
                    float(candidate_emb @ hist_emb)
                    for hist_emb in recent_embs
                ]
                avg_sim = np.mean(similarities)
                novelty = 1.0 - avg_sim
            else:
                novelty = 0.5
            
            # Serendipity score
            serendipity = 0.4 * unexpectedness + 0.6 * novelty
            
            # Combined score
            final_score = (1 - beta) * relevance + beta * serendipity
            serendipity_scores[nid] = final_score
        
        # Sort by serendipity score
        ranked = sorted(serendipity_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:k]
    
    # -----------------------------------------------------------------------
    # Unified Interface
    # -----------------------------------------------------------------------
    
    def rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        method: str = 'mmr',
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        Unified re-ranking interface.
        
        Args:
            candidates:    Baseline candidates
            user_history:  User's click history
            k:             Number to select
            method:        'mmr', 'xquad', 'calibrated', 'serendipity'
            **kwargs:      Method-specific parameters
        
        Returns:
            Re-ranked list
        """
        if method == 'mmr':
            return self.mmr_rerank(candidates, user_history, k, **kwargs)
        elif method == 'xquad':
            return self.xquad_rerank(candidates, user_history, k, **kwargs)
        elif method == 'calibrated':
            return self.calibrated_rerank(candidates, user_history, k, **kwargs)
        elif method == 'serendipity':
            return self.serendipity_rerank(candidates, user_history, k, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    logger.info("Diversity Re-ranking Algorithms — Phase 4")
    logger.info("4 algorithms: MMR, xQuAD, Calibrated, Serendipity")
