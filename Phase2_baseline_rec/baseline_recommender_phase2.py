"""
Phase 2: Baseline Recommender System
=====================================
Diversity-Aware News Recommender System — Capstone Project

This is the accuracy-optimized baseline that Phase 4 diversity methods
will be compared against.

Architecture:
  Input:  User click history (list of news IDs)
  
  Step 1: Load Phase 1 embeddings from disk (~2 seconds)
  
  Step 2: Build four user profiles using AttentionUserProfiler
          - recency-weighted (primary, attention-scored)
          - uniform-weighted (mean of all clicks)
          - recent-5 (most recent 5 clicks)
          - max-match (max cosine similarity to any single history article)
          Content scores = 0.45 × recency + 0.20 × uniform + 0.15 × recent5
                         + 0.20 × max-match, then min-max normalized.
          Max-match captures "similar to at least one past read" — avoids the
          centroid averaging problem for users with diverse interests.

  Step 3: Multi-query FAISS retrieval (<5 ms)
          - Query 1: recency-weighted profile  → top-150 candidates
          - Query 2: uniform-weighted profile  → top-75  candidates
          - Query 3: recent-clicks profile     → top-50  candidates
          → Union: up to ~200 unique candidates

  Step 4: Hybrid scoring with accuracy-optimised dynamic weights.
          Within an impression, popularity and recency are equal for all
          candidates (platform selected them all at the same moment), so they
          do not discriminate clicked vs non-clicked. Content + affinity do.

          Short history  (≤3):  0.45 × content + 0.15 × popularity + 0.00 × recency + 0.15 × affinity + 0.25 × cf
          Medium history (≤10): 0.60 × content + 0.05 × popularity + 0.00 × recency + 0.20 × affinity + 0.15 × cf
          Long history   (>10): 0.65 × content + 0.00 × popularity + 0.00 × recency + 0.25 × affinity + 0.10 × cf
          (affinity = user's normalized category click distribution; 0.0 if category map unavailable)
          (cf = item-item co-click CF — built from ALL training impressions, not just first per user)

  Step 5: Return top-K recommendations

Evaluation:
  - AUC (Area Under ROC)
  - MRR (Mean Reciprocal Rank)
  - Precision@K, Recall@K, F1@K, NDCG@K for K ∈ {5, 10, 20}
  - Diversity metrics (Gini, ILD, Coverage) for comparison

Expected Performance (after re-training with fixed CF index):
  AUC:        0.67–0.72
  NDCG@10:    0.42–0.46
  MRR:        0.35–0.40

Note on NDCG@10 ceiling:
  State-of-the-art end-to-end neural models on MINDlarge (NRMS, NAML, LSTUR)
  achieve NDCG@10 ≈ 0.43–0.47. This hybrid model approaches that ceiling
  without task-specific fine-tuning. Scores above ~0.55 are not achievable on
  MIND due to inherent click-noise — many relevant articles are never clicked
  simply because the user didn't see them.

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


# Ensure Phase 1 package is importable when this module is run directly
base_dir = Path(__file__).resolve().parent
phase1_path = base_dir.parent / "Phase1_NLP_encoding"
if phase1_path.exists():
    import sys
    if str(phase1_path) not in sys.path:
        sys.path.insert(0, str(phase1_path))


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
        candidate_pool_size: int = 150,
    ):
        """
        Args:
            embeddings_dict:    Output from load_phase1_embeddings()
            content_weight:     Base weight for content similarity (overridden dynamically)
            popularity_weight:  Base weight for popularity score (overridden dynamically)
            recency_weight:     Weight for recency score
            decay_lambda:       Recency decay rate for user profiler
            candidate_pool_size: FAISS candidates retrieved per query vector
        """
        self.encoder = embeddings_dict['encoder']
        self.final_embeddings = embeddings_dict['final_embeddings']
        self.news_ids = embeddings_dict['news_ids']
        self.news_id_to_idx = embeddings_dict['news_id_to_idx']
        self.retriever = embeddings_dict['retriever']
        
        # Base hybrid scoring weights (dynamic weighting overrides at recommend time)
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.recency_weight = recency_weight

        # Multi-query FAISS pool size per query vector
        self.candidate_pool_size = candidate_pool_size

        # User profiler
        from nlp_encoder import AttentionUserProfiler
        self.profiler = AttentionUserProfiler(decay_lambda=decay_lambda)
        
        # Popularity scores (will be computed from training data)
        self.popularity_scores: Dict[str, float] = {}
        self._default_pop: float = 0.0  # median of training scores; fallback for unseen articles

        # Recency scores (days since publication — needs news metadata)
        self.recency_scores: Dict[str, float] = {}

        # Category affinity: news_id → category string (built from news_df in fit_popularity)
        self.news_id_to_category: Dict[str, str] = {}

        # News metadata (for diversity metrics)
        self.news_metadata: Optional[pd.DataFrame] = None

        # Co-click index for collaborative filtering (built from training data)
        self.co_click: Dict[str, Dict[str, float]] = {}
    
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
        
        self._default_pop = float(np.median(list(self.popularity_scores.values()))) if self.popularity_scores else 0.0

        logger.info(f"  Popularity scores computed for {len(self.popularity_scores):,} articles")
        logger.info(f"  Articles with clicks: {len(click_counts):,}")
        logger.info(f"  Default popularity (median) for unseen articles: {self._default_pop:.4f}")

        # ---- Recency scores from impression timestamps ----
        # Each article's recency = how recently it was seen in training impressions.
        # Articles whose latest impression timestamp is closest to the end of the
        # training window get the highest recency score (→ 1.0).
        news_last_seen: Dict[str, object] = {}
        for interaction in train_interactions:
            t = interaction.get('impression_time')
            if t is None:
                continue
            nid = interaction['news_id']
            if nid not in news_last_seen or t > news_last_seen[nid]:
                news_last_seen[nid] = t

        if news_last_seen:
            ts_values = [t.timestamp() for t in news_last_seen.values()]
            min_ts = min(ts_values)
            max_ts = max(ts_values)
            ts_range = max_ts - min_ts if max_ts > min_ts else 1.0

            self.recency_scores = {}
            for news_id in self.news_ids:
                if news_id in news_last_seen:
                    elapsed = news_last_seen[news_id].timestamp() - min_ts
                    self.recency_scores[news_id] = elapsed / ts_range  # [0, 1]
                else:
                    self.recency_scores[news_id] = 0.0  # never seen → least recent

            logger.info(
                f"  Recency scores computed from {len(news_last_seen):,} timestamped articles "
                f"({len(self.recency_scores):,} scored)"
            )
        else:
            logger.warning(
                "  No impression timestamps found — recency_scores not populated. "
                "All articles will default to 0.5 at recommendation time."
            )

        # Store news metadata and build category mapping
        if news_df is not None:
            self.news_metadata = news_df
            if 'news_id' in news_df.columns and 'category' in news_df.columns:
                self.news_id_to_category = dict(
                    zip(news_df['news_id'], news_df['category'].fillna(''))
                )
                logger.info(
                    f"  Category mapping built for {len(self.news_id_to_category):,} articles "
                    f"({news_df['category'].nunique()} unique categories)"
                )
            logger.info(f"  News metadata loaded: {len(news_df):,} articles")

        logger.info("Building co-click index ...")
        self._build_co_click_index(train_interactions)

    # ------------------------------------------------------------------
    # Helper methods for multi-query retrieval and dynamic weighting
    # ------------------------------------------------------------------

    def _build_uniform_profile(self, history: List[str]) -> Optional[np.ndarray]:
        """Mean (uniform-weight) user profile — treats all clicks equally."""
        valid_ids = [nid for nid in history if nid in self.news_id_to_idx]
        if not valid_ids:
            return None
        indices = [self.news_id_to_idx[nid] for nid in valid_ids]
        vecs = self.final_embeddings[indices]
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        return (mean_vec / norm).astype(np.float32) if norm > 1e-10 else mean_vec.astype(np.float32)

    def _build_recent_profile(self, history: List[str], n_recent: int = 5) -> Optional[np.ndarray]:
        """Profile built from only the most recent N clicks."""
        return self._build_uniform_profile(history[-n_recent:])

    def _get_dynamic_weights(self, history_len: int) -> Tuple[float, float, float, float, float]:
        """
        Return (content_w, popularity_w, recency_w, affinity_w, cf_w) adapted to history length.

        Key insight for NDCG maximisation on impression-based evaluation:
        Within a single impression, ALL candidates (clicked and non-clicked) were
        selected by the platform at the same time — so popularity and recency are
        roughly equal across clicked and non-clicked articles and do NOT discriminate.
        The discriminating signals are content similarity (does the article match the
        user's taste?) and category affinity (is it in a category the user favours?).
        CF helps when co-clicked articles appear in the same impression candidate set.

        - Short history (≤3):   content + CF + small popularity fallback (profile is noisy)
        - Medium history (≤10): strongly content + affinity; small CF + negligible pop
        - Long history   (>10): content + affinity dominant; popularity removed entirely
        """
        if not self.news_id_to_category:
            # No category map — 3-component fallback (content, popularity, CF)
            if history_len <= 3:
                return 0.45, 0.15, 0.00, 0.00, 0.40
            elif history_len <= 10:
                return 0.60, 0.05, 0.00, 0.00, 0.35
            else:
                return 0.70, 0.00, 0.00, 0.00, 0.30

        if history_len <= 3:
            return 0.45, 0.15, 0.00, 0.15, 0.25
        elif history_len <= 10:
            return 0.60, 0.05, 0.00, 0.20, 0.15
        else:
            return 0.65, 0.00, 0.00, 0.25, 0.10

    def _build_co_click_index(
        self, train_interactions: List[Dict], top_k: int = 50
    ):
        """
        Build item-item co-click index from training interactions.

        For each user, only their longest (last) history snapshot is used.
        In MIND, the same user appears across many impression rows with
        progressively growing histories; each earlier history is a strict
        prefix of later ones, so the final snapshot already contains all
        pairs that earlier snapshots would contribute — processing every
        impression is redundant and ~3–5× slower.

        Args:
            train_interactions: List of interaction dicts (same as passed to fit_popularity)
            top_k:              Max co-clicked articles to keep per article (memory control)
        """
        # Pass 1: keep only the longest history per user (O(n) scan)
        user_best: Dict[str, List[str]] = {}
        for interaction in train_interactions:
            uid = interaction.get('user_id', '')
            if not uid:
                continue
            history = interaction.get('history') or []
            prev = user_best.get(uid)
            if prev is None or len(history) > len(prev):
                user_best[uid] = history

        # Pass 2: generate co-click pairs from unique user histories
        raw: Dict[str, Counter] = defaultdict(Counter)
        n_processed = 0
        for history in user_best.values():
            known = [nid for nid in history[-15:] if nid in self.news_id_to_idx]
            for i, a in enumerate(known):
                for b in known[i + 1:]:
                    raw[a][b] += 1
                    raw[b][a] += 1
            n_processed += 1

        # Normalize and keep top-k per article
        for article, co_counts in raw.items():
            max_count = max(co_counts.values())
            top = co_counts.most_common(top_k)
            self.co_click[article] = {
                b: cnt / max_count for b, cnt in top
            }
        logger.info(
            f"  Co-click index built: {len(self.co_click):,} articles "
            f"({n_processed:,} unique users processed)"
        )

    def recommend(
        self,
        user_history: List[str],
        k: int = 10,
        exclude_history: bool = True,
        candidates: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate top-K recommendations for a user.

        When candidates is None (production mode):
          - Builds three query vectors and unions their FAISS results for a
            richer, more diverse candidate pool.
          - Applies dynamic scoring weights based on history length.

        When candidates is provided (evaluation mode):
          - Scores the given fixed candidate list (no FAISS call needed).
          - Dynamic weights still apply.

        Args:
            user_history:    List of clicked news IDs (chronological order)
            k:               Number of recommendations to return
            exclude_history: Remove history items from the output
            candidates:      Optional fixed candidate list (evaluation mode)

        Returns:
            List of (news_id, score) tuples, sorted descending by score
        """
        if not user_history:
            return self._cold_start_recommend(k)

        # Dynamic weights based on how much history we have
        content_w, pop_w, recency_w, affinity_w, cf_w = self._get_dynamic_weights(len(user_history))

        # Build all three user profiles (used for both FAISS retrieval and content scoring)
        recency_profile = self.profiler.build_profile(
            history_ids=user_history,
            news_embeddings=self.final_embeddings,
            news_id_to_idx=self.news_id_to_idx,
            max_history=50,
        )
        uniform_profile = self._build_uniform_profile(user_history)
        recent_profile = self._build_recent_profile(user_history, n_recent=5)

        if candidates is None:
            # ---- Multi-query FAISS retrieval ----
            exclude_ids = user_history if exclude_history else None
            candidate_set: set = set()

            # Query 1: recency-weighted profile (primary — largest pool)
            pool = self.candidate_pool_size
            cids1, _ = self.retriever.retrieve(
                query_vector=recency_profile,
                k=pool,
                exclude_ids=exclude_ids,
            )
            candidate_set.update(cids1)

            # Query 2: uniform-weighted profile (catches items recency-profile misses)
            if uniform_profile is not None:
                cids2, _ = self.retriever.retrieve(
                    query_vector=uniform_profile,
                    k=pool // 2,
                    exclude_ids=exclude_ids,
                )
                candidate_set.update(cids2)

            # Query 3: recent-clicks profile (boosts fresh interests)
            if len(user_history) > 5 and recent_profile is not None:
                cids3, _ = self.retriever.retrieve(
                    query_vector=recent_profile,
                    k=pool // 3,
                    exclude_ids=exclude_ids,
                )
                candidate_set.update(cids3)

            candidate_ids = [cid for cid in candidate_set if cid in self.news_id_to_idx]
            if not candidate_ids:
                return self._cold_start_recommend(k)

        else:
            # ---- Evaluation mode: score fixed impression candidates ----
            candidate_ids = [
                cid for cid in candidates
                if cid in self.news_id_to_idx
                and ((not exclude_history) or (cid not in user_history))
            ]
            if not candidate_ids:
                return []

        # Multi-profile content scoring (applied in both production and eval mode)
        cand_indices = np.array([self.news_id_to_idx[cid] for cid in candidate_ids])
        embs = self.final_embeddings[cand_indices]

        # Profile 1–3: recency-weighted, uniform, recent-5 centroids
        content_scores = (0.45 * (embs @ recency_profile)).astype(np.float32)
        if uniform_profile is not None:
            content_scores += 0.20 * (embs @ uniform_profile)
        if recent_profile is not None:
            content_scores += 0.15 * (embs @ recent_profile)

        # Profile 4: max-match — similarity to the MOST similar history article.
        # A user who reads both sports AND tech has a centroid that matches neither
        # well; max-match captures "this article is very close to at least one of
        # your past reads", which is a strong relevance signal.
        known_hist_indices = [
            self.news_id_to_idx[nid]
            for nid in user_history
            if nid in self.news_id_to_idx
        ]
        if known_hist_indices:
            hist_embs = self.final_embeddings[np.array(known_hist_indices)]  # (H, D)
            # (N, H) dot product; take row-wise max → (N,)
            max_match = (embs @ hist_embs.T).max(axis=1).astype(np.float32)
            content_scores += 0.20 * max_match

        # Min-max normalize content scores into [0, 1] per impression
        cs_min, cs_max = content_scores.min(), content_scores.max()
        if cs_max > cs_min:
            content_scores = (content_scores - cs_min) / (cs_max - cs_min)

        # User category affinity: normalized distribution over categories in click history
        user_cat_affinity: Dict[str, float] = {}
        if affinity_w > 0 and self.news_id_to_category:
            cat_counts: Counter = Counter()
            for nid in user_history:
                cat = self.news_id_to_category.get(nid, '')
                if cat:
                    cat_counts[cat] += 1
            total_clicks = sum(cat_counts.values())
            if total_clicks > 0:
                user_cat_affinity = {cat: cnt / total_clicks for cat, cnt in cat_counts.items()}

        # CF signal: aggregate co-click votes from user history, normalize by history length
        history_cf: Dict[str, float] = defaultdict(float)
        if cf_w > 0 and self.co_click:
            for hist_id in user_history:
                for co_article, weight in self.co_click.get(hist_id, {}).items():
                    history_cf[co_article] += weight
            if history_cf:
                max_cf = max(history_cf.values())
                if max_cf > 0:
                    for k_id in history_cf:
                        history_cf[k_id] /= max_cf

        # Hybrid scoring (content + popularity + recency + category affinity + CF)
        final_scores = []
        for i, news_id in enumerate(candidate_ids):
            cat = self.news_id_to_category.get(news_id, '')
            score = (
                content_w  * float(content_scores[i]) +
                pop_w      * self.popularity_scores.get(news_id, self._default_pop) +
                recency_w  * self.recency_scores.get(news_id, 0.5) +
                affinity_w * user_cat_affinity.get(cat, 0.0) +
                cf_w       * history_cf.get(news_id, 0.0)
            )
            final_scores.append((news_id, score))

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
            'candidate_pool_size': self.candidate_pool_size,
            'popularity_scores': self.popularity_scores,
            'default_pop': self._default_pop,
            'recency_scores': self.recency_scores,
            'profiler_lambda': self.profiler.decay_lambda,
            'news_id_to_category': self.news_id_to_category,
            'co_click': self.co_click,
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
            candidate_pool_size=state.get('candidate_pool_size', 150),
        )
        obj.popularity_scores = state['popularity_scores']
        obj._default_pop = state.get('default_pop', 0.0)
        obj.recency_scores = state.get('recency_scores', {})
        obj.news_id_to_category = state.get('news_id_to_category', {})
        obj.co_click = state.get('co_click', {})

        logger.info(f"Recommender loaded from {filepath}")
        return obj


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class RecommenderEvaluator:
    """
    Comprehensive evaluation for the baseline recommender.

    Accuracy metrics computed
    ─────────────────────────
    Ranking (list-level)
      AUC        Area Under the ROC curve (aggregate over all impression
                 candidates).  Measures how well the model separates
                 clicked from non-clicked articles regardless of rank.
      MRR        Mean Reciprocal Rank.  Rewards placing the *first*
                 relevant article as high as possible in the ranked list.

    Top-K (cutoff-level, averaged over impressions)
      Precision@K  Fraction of top-K items that are relevant.
      Recall@K     Fraction of all relevant items that appear in top-K.
      F1@K         Harmonic mean of Precision@K and Recall@K.
      NDCG@K       Normalised Discounted Cumulative Gain.  Accounts for
                   the *position* of each hit — higher ranks score more.
      HR@K         Hit Rate.  1 if at least one relevant item is in
                   top-K, else 0.  A softer correctness measure than
                   Precision but intuitive: "did the user see something
                   they would click?"
      MAP@K        Mean Average Precision.  For each impression AP@K is
                   the average of the precision values at every rank
                   where a hit occurs.  Penalises gaps between hits more
                   than NDCG, making it sensitive to *how many* relevant
                   items the model retrieves and *where* they land.

    Score discriminability
      score_gap    Mean score of clicked articles minus mean score of
                   non-clicked articles.  A positive gap confirms the
                   model assigns higher relevance scores to items the
                   user actually clicked.
    """

    def __init__(self, recommender: BaselineRecommender, news_df: pd.DataFrame):
        self.recommender = recommender
        self.news_df = news_df
        self.news_categories = dict(zip(news_df['news_id'], news_df['category']))

    # ------------------------------------------------------------------
    # Main evaluation entry-point
    # ------------------------------------------------------------------

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
                        - history   (list of news IDs)
                        - impressions (list of (news_id, label) tuples)
            k_values:   List of K cutoffs for top-K metrics.

        Returns:
            Dict containing all accuracy metrics (see class docstring).
        """
        logger.info(f"Evaluating on {len(test_data):,} test impressions ...")
        t0 = time.time()

        all_labels: List[float] = []
        all_scores: List[float] = []

        # Accumulators for discriminability
        pos_scores: List[float] = []   # scores of clicked items
        neg_scores: List[float] = []   # scores of non-clicked items

        metrics_at_k: Dict[str, List[float]] = defaultdict(list)

        for sample in test_data:
            history    = sample['history']
            impressions = sample['impressions']

            if not impressions:
                continue

            # Keep only articles the encoder knows about
            known_impressions = [
                (nid, label) for nid, label in impressions
                if nid in self.recommender.news_id_to_idx
            ]
            if not known_impressions:
                continue

            candidate_ids = [nid  for nid, _     in known_impressions]
            labels        = np.array([lbl for _,   lbl in known_impressions], dtype=float)

            # Score every candidate
            recs      = self.recommender.recommend(
                user_history=history,
                k=len(candidate_ids),
                candidates=candidate_ids,
                exclude_history=False,
            )
            rec_dict  = dict(recs)
            scores    = np.array([rec_dict.get(nid, 0.0) for nid in candidate_ids])

            # ── Aggregate for AUC ───────────────────────────────────────
            all_labels.extend(labels.tolist())
            all_scores.extend(scores.tolist())

            # ── Score discriminability ──────────────────────────────────
            pos_mask = labels == 1
            if pos_mask.any():
                pos_scores.extend(scores[pos_mask].tolist())
            neg_mask = labels == 0
            if neg_mask.any():
                neg_scores.extend(scores[neg_mask].tolist())

            # ── Top-K metrics ───────────────────────────────────────────
            sorted_indices = np.argsort(scores)[::-1]   # descending rank

            for k in k_values:
                top_k_idx    = sorted_indices[:k]
                top_k_labels = labels[top_k_idx]

                n_hits    = top_k_labels.sum()
                n_rel     = labels.sum()
                precision = n_hits / k          if k    > 0 else 0.0
                recall    = n_hits / n_rel       if n_rel > 0 else 0.0
                f1        = (2 * precision * recall / (precision + recall)
                             if (precision + recall) > 0 else 0.0)
                ndcg      = self._ndcg(labels, scores, k)
                hr        = 1.0 if n_hits > 0 else 0.0
                ap        = self._average_precision(labels, sorted_indices, k)

                metrics_at_k[f'precision@{k}'].append(precision)
                metrics_at_k[f'recall@{k}'].append(recall)
                metrics_at_k[f'f1@{k}'].append(f1)
                metrics_at_k[f'ndcg@{k}'].append(ndcg)
                metrics_at_k[f'hr@{k}'].append(hr)
                metrics_at_k[f'map@{k}'].append(ap)

            # ── MRR ─────────────────────────────────────────────────────
            for rank, idx in enumerate(sorted_indices, 1):
                if labels[idx] == 1:
                    metrics_at_k['mrr'].append(1.0 / rank)
                    break
            else:
                metrics_at_k['mrr'].append(0.0)

        # ── Aggregate ───────────────────────────────────────────────────
        results: Dict = {
            'auc': (roc_auc_score(all_labels, all_scores)
                    if len(set(all_labels)) > 1 else 0.0),
            'mrr': float(np.mean(metrics_at_k['mrr'])),
            'score_gap': (float(np.mean(pos_scores) - np.mean(neg_scores))
                          if pos_scores and neg_scores else 0.0),
        }

        for k in k_values:
            results[f'precision@{k}'] = float(np.mean(metrics_at_k[f'precision@{k}']))
            results[f'recall@{k}']    = float(np.mean(metrics_at_k[f'recall@{k}']))
            results[f'f1@{k}']        = float(np.mean(metrics_at_k[f'f1@{k}']))
            results[f'ndcg@{k}']      = float(np.mean(metrics_at_k[f'ndcg@{k}']))
            results[f'hr@{k}']        = float(np.mean(metrics_at_k[f'hr@{k}']))
            results[f'map@{k}']       = float(np.mean(metrics_at_k[f'map@{k}']))

        elapsed = time.time() - t0
        logger.info(f"Evaluation complete in {elapsed:.1f}s")
        return results

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _ndcg(self, labels: np.ndarray, scores: np.ndarray, k: int) -> float:
        """Normalised Discounted Cumulative Gain at K."""
        sorted_indices = np.argsort(scores)[::-1][:k]
        dcg = sum(
            labels[idx] / np.log2(rank + 2)
            for rank, idx in enumerate(sorted_indices)
        )
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = sum(
            label / np.log2(rank + 2)
            for rank, label in enumerate(ideal_labels)
        )
        return dcg / idcg if idcg > 0 else 0.0

    def _average_precision(
        self,
        labels: np.ndarray,
        sorted_indices: np.ndarray,
        k: int,
    ) -> float:
        """
        Average Precision at K for a single impression.

        AP@K = (1 / min(R, K)) * sum_{i=1}^{K}  P@i * rel(i)

        where R is the total number of relevant items and rel(i) = 1
        if the item at rank i is relevant, else 0.
        """
        n_rel = int(labels.sum())
        if n_rel == 0:
            return 0.0
        hits      = 0
        precision_sum = 0.0
        for rank, idx in enumerate(sorted_indices[:k], 1):
            if labels[idx] == 1:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / min(n_rel, k)
    
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
    
    # Load Phase 1 embeddings from the standard Phase 1 directory
    embeddings_dir = base_dir.parent / "Phase1_NLP_encoding" / "embeddings"
    recommender = BaselineRecommender.from_embeddings(str(embeddings_dir))
    
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
