"""
Phase 4: Diversity-Aware Re-ranking Algorithms
===============================================
Diversity-Aware News Recommender System — Capstone Project

This module implements 6 diversity re-ranking algorithms that address
echo chambers identified in Phase 3. The key structural fix is candidate
pool augmentation: injecting articles from absent categories before
re-ranking, so algorithms have diverse material to work with.

Root Cause Fix:
---------------
Phase 2 FAISS retrieves top-100 candidates from the user's own profile
vector. For a sports-only user this returns ~100 sports articles.
Re-rankers cannot create diversity from a homogeneous pool.

Solution: _inject_diverse_candidates() appends popular articles from
categories absent in the current pool, expanding coverage to all 18
MIND categories before any re-ranking step.

Algorithms:
-----------
1. MMR  — Maximal Marginal Relevance with category saturation penalty
2. xQuAD — Category coverage with exploration blend (uniform prior)
3. Calibrated — KL-divergence minimisation with diversity smoothing
4. Serendipity — Greedy selection with 4 continuous components
5. Bounded Greedy — Hard per-category cap (strongest Gini reduction)
6. Max Coverage — Reserves slots for category coverage, fills rest by relevance
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiversityReranker:
    """
    Implements 6 diversity-aware re-ranking algorithms with candidate
    pool augmentation to break structural echo chambers.
    """

    def __init__(
        self,
        baseline_recommender,
        embeddings: np.ndarray,
        news_id_to_idx: Dict[str, int],
        news_categories: Dict[str, str],
        popularity_scores: Optional[Dict[str, float]] = None,
        corpus_category_dist: Optional[Dict[str, float]] = None,
        corpus_subcategory_dist: Optional[Dict[str, float]] = None,
        news_subcategories: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            baseline_recommender:    Trained BaselineRecommender from Phase 2
            embeddings:              Final embeddings from Phase 1
            news_id_to_idx:          news_id → row index mapping
            news_categories:         news_id → category mapping
            popularity_scores:       news_id → popularity score (optional)
            corpus_category_dist:    category → proportion in full corpus (for system fairness)
            corpus_subcategory_dist: subcategory → proportion in full corpus (for system fairness)
            news_subcategories:      news_id → subcategory mapping (for system fairness)
        """
        self.baseline = baseline_recommender
        self.embeddings = embeddings
        self.news_id_to_idx = news_id_to_idx
        self.news_categories = news_categories
        self.popularity_scores = popularity_scores or {}
        self.corpus_category_dist = corpus_category_dist or {}
        self.corpus_subcategory_dist = corpus_subcategory_dist or {}
        self.news_subcategories = news_subcategories or {}

        # All unique categories
        self.all_categories = sorted(set(news_categories.values()))
        self.n_categories = len(self.all_categories)

        # All unique subcategories (used for system-level fairness scoring)
        self.all_subcategories = sorted(set(news_subcategories.values())) if news_subcategories else []
        self.n_subcategories = len(self.all_subcategories)

        # Lazy-built category pool (built on first call)
        self._category_pool: Optional[Dict[str, List[Tuple[str, float]]]] = None

    # -----------------------------------------------------------------------
    # Pool Augmentation (structural fix)
    # -----------------------------------------------------------------------

    def _build_category_pool(self) -> None:
        """
        Build {category → [(news_id, popularity), ...]} sorted by popularity desc.
        Called once lazily; stored as self._category_pool.
        """
        pool: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for news_id, cat in self.news_categories.items():
            if news_id not in self.news_id_to_idx:
                continue
            pop = self.popularity_scores.get(news_id, 0.0)
            pool[cat].append((news_id, pop))

        # Sort each category bucket by popularity descending
        for cat in pool:
            pool[cat].sort(key=lambda x: x[1], reverse=True)

        self._category_pool = dict(pool)
        logger.debug(
            f"Category pool built: {len(self._category_pool)} categories, "
            f"{sum(len(v) for v in self._category_pool.values())} articles total"
        )

    def _inject_diverse_candidates(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        inject_per_cat: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Append popular articles from categories absent in the current pool.

        For each category not represented in the top-100 candidates, we
        inject the top `inject_per_cat` popular articles (excluding history
        and duplicates). Injected articles get a discounted score so real
        candidates are preferred by relevance-aware algorithms.

        Args:
            candidates:      Current candidate list (news_id, score)
            user_history:    User's click history (to exclude)
            inject_per_cat:  Articles to inject per absent category

        Returns:
            Augmented candidate list
        """
        if self._category_pool is None:
            self._build_category_pool()

        existing_ids = set(nid for nid, _ in candidates)
        history_set = set(user_history)
        excluded = existing_ids | history_set

        # Categories already present in pool
        pool_cats = {
            self.news_categories[nid]
            for nid, _ in candidates
            if nid in self.news_categories
        }

        # Set injected articles at the 30th percentile of the natural score range
        # so they are genuinely competitive when diversity/fairness weights are high,
        # but still below most natural candidates on pure relevance.
        # Previously used min_score * 0.5 which put injected articles below the
        # normalization floor, giving them norm_rel = 0 and making them impossible
        # to select regardless of fairness/diversity weights.
        scores = [s for _, s in candidates if s > 0]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            injected_score = min_score + 0.3 * (max_score - min_score)
        else:
            injected_score = 0.01

        augmented = list(candidates)

        for cat in self.all_categories:
            if cat in pool_cats:
                continue  # Category already represented

            bucket = self._category_pool.get(cat, [])
            added = 0
            for news_id, _pop in bucket:
                if news_id in excluded:
                    continue
                augmented.append((news_id, injected_score))
                excluded.add(news_id)
                added += 1
                if added >= inject_per_cat:
                    break

        return augmented

    # -----------------------------------------------------------------------
    # Algorithm 1: MMR with Category Saturation Penalty
    # -----------------------------------------------------------------------

    def mmr_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        lambda_param: float = 0.3,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        MMR with category saturation penalty.

        MMR(i) = λ × relevance(i)
                 − (1−λ) × [0.7 × max_emb_sim(i, selected)
                            + 0.3 × cat_share(i, selected)]

        The category saturation term explicitly penalises repeating a
        category beyond its fair share, even when embedding similarity
        is low (e.g. different sports articles can be embedding-distant).

        Args:
            candidates:         (news_id, relevance_score) list
            user_history:       User's click history
            k:                  Items to select
            lambda_param:       0 = pure diversity, 1 = pure relevance
            inject_candidates:  Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, original_relevance_score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        candidate_ids = [nid for nid, _ in candidates if nid in self.news_id_to_idx]
        relevance_scores = {nid: score for nid, score in candidates}

        if not candidate_ids:
            return []

        candidate_embs = {
            nid: self.embeddings[self.news_id_to_idx[nid]]
            for nid in candidate_ids
        }

        selected: List[str] = []
        selected_cats: Counter = Counter()
        remaining = set(candidate_ids)

        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None

            n_selected = max(1, len(selected))

            for nid in remaining:
                relevance = relevance_scores.get(nid, 0.0)
                cat = self.news_categories.get(nid, "")

                if selected:
                    # Embedding similarity penalty
                    emb_sims = [
                        float(candidate_embs[nid] @ candidate_embs[sel])
                        for sel in selected
                        if sel in candidate_embs
                    ]
                    max_emb_sim = max(emb_sims) if emb_sims else 0.0

                    # Category saturation penalty
                    cat_share = selected_cats.get(cat, 0) / n_selected

                    diversity_penalty = 0.7 * max_emb_sim + 0.3 * cat_share
                else:
                    diversity_penalty = 0.0

                score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

                if score > best_score:
                    best_score = score
                    best_item = nid

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
                best_cat = self.news_categories.get(best_item, "")
                if best_cat:
                    selected_cats[best_cat] += 1

        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

    # -----------------------------------------------------------------------
    # Algorithm 2: xQuAD with Exploration Blend
    # -----------------------------------------------------------------------

    def xquad_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        lambda_param: float = 0.5,
        explore_weight: float = 0.3,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        xQuAD with uniform exploration prior.

        target(cat) = (1 − explore_weight) × history_dist(cat)
                    + explore_weight × (1 / n_categories)

        Categories absent from user history still get a minimum quota
        (~1.7% at 18 categories with explore_weight=0.3), so the algorithm
        actively selects from them instead of ignoring them entirely.

        Args:
            candidates:       (news_id, relevance_score) list
            user_history:     User's click history
            k:                Items to select
            lambda_param:     Relevance vs coverage trade-off
            explore_weight:   Blend weight for uniform prior (0=pure history)
            inject_candidates: Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        candidate_ids = [nid for nid, _ in candidates if nid in self.news_categories]
        relevance_scores = {nid: score for nid, score in candidates}

        if not candidate_ids:
            return self.mmr_rerank(candidates, user_history, k, lambda_param,
                                   inject_candidates=False)

        # Build blended target distribution
        history_cats = [
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        ]

        if history_cats:
            cat_counts = Counter(history_cats)
            total = sum(cat_counts.values())
            history_dist = {cat: cat_counts.get(cat, 0) / total
                            for cat in self.all_categories}
        else:
            history_dist = {cat: 1.0 / self.n_categories
                            for cat in self.all_categories}

        uniform = 1.0 / self.n_categories
        target_dist = {
            cat: (1 - explore_weight) * history_dist.get(cat, 0.0)
                 + explore_weight * uniform
            for cat in self.all_categories
        }

        # Greedy selection
        selected: List[str] = []
        selected_cats: Counter = Counter()
        remaining = set(candidate_ids)
        coverage_floor = 0.005  # Minimum coverage gap to avoid stalling

        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None

            n_selected = max(1, len(selected))

            for nid in remaining:
                cat = self.news_categories.get(nid, "")
                relevance = relevance_scores.get(nid, 0.0)

                p_cat = target_dist.get(cat, 0.0)
                current_coverage = selected_cats.get(cat, 0) / n_selected
                coverage_gap = max(coverage_floor, p_cat - current_coverage)

                score = lambda_param * relevance + (1 - lambda_param) * coverage_gap

                if score > best_score:
                    best_score = score
                    best_item = nid

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
                best_cat = self.news_categories.get(best_item, "")
                if best_cat:
                    selected_cats[best_cat] += 1

        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

    # -----------------------------------------------------------------------
    # Algorithm 3: Calibrated with Diversity Smoothing
    # -----------------------------------------------------------------------

    def calibrated_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        alpha: float = 0.5,
        diversity_weight: float = 0.3,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Calibrated re-ranking with diversity smoothing.

        target(cat) = (1 − diversity_weight) × smoothed_history(cat)
                    + diversity_weight × (1 / n_categories)

        Minimises KL divergence against this blended target, pulling
        recommendations toward global diversity while respecting preferences.

        Args:
            candidates:        (news_id, relevance_score) list
            user_history:      User's click history
            k:                 Items to select
            alpha:             Calibration strength (0=pure relevance)
            diversity_weight:  Blend weight for uniform distribution
            inject_candidates: Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        candidate_ids = [nid for nid, _ in candidates if nid in self.news_categories]
        relevance_scores = {nid: score for nid, score in candidates}

        if not candidate_ids:
            return self.mmr_rerank(candidates, user_history, k, 0.5,
                                   inject_candidates=False)

        # Build smoothed history distribution
        history_cats = [
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        ]

        if history_cats:
            cat_counts = Counter(history_cats)
            total = sum(cat_counts.values())
            smoothing = 0.01
            denom = total + smoothing * self.n_categories
            smoothed_hist = {
                cat: (cat_counts.get(cat, 0) + smoothing) / denom
                for cat in self.all_categories
            }
        else:
            smoothed_hist = {cat: 1.0 / self.n_categories
                             for cat in self.all_categories}

        uniform = 1.0 / self.n_categories
        target_dist = {
            cat: (1 - diversity_weight) * smoothed_hist.get(cat, 0.0)
                 + diversity_weight * uniform
            for cat in self.all_categories
        }

        # Normalise target
        total_target = sum(target_dist.values())
        target_dist = {cat: v / total_target for cat, v in target_dist.items()}

        # Greedy KL-minimisation
        selected: List[str] = []
        remaining = set(candidate_ids)
        smoothing = 0.01
        denom_base = smoothing * self.n_categories

        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None

            test_cats_base = [
                self.news_categories[s]
                for s in selected
                if s in self.news_categories
            ]

            for nid in remaining:
                test_cats = test_cats_base + [self.news_categories[nid]]
                test_counts = Counter(test_cats)
                test_total = len(test_cats)

                current_dist = {
                    cat: (test_counts.get(cat, 0) + smoothing) / (test_total + denom_base)
                    for cat in self.all_categories
                }

                kl_div = sum(
                    target_dist[cat] * np.log(target_dist[cat] / current_dist[cat])
                    for cat in self.all_categories
                    if target_dist[cat] > 0 and current_dist[cat] > 0
                )

                relevance = relevance_scores.get(nid, 0.0)
                score = (1 - alpha) * relevance + alpha * (-kl_div)

                if score > best_score:
                    best_score = score
                    best_item = nid

            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)

        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

    # -----------------------------------------------------------------------
    # Algorithm 4: Serendipity (Greedy + Continuous Components)
    # -----------------------------------------------------------------------

    def serendipity_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        beta: float = 0.4,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Serendipity re-ranking: greedy selection with four continuous components.

        Four serendipity components:
          1. Unexpectedness  : 1 − cosine_sim(article, user_centroid)
          2. Popularity novelty: 1 − (pop / max_pop)  (niche = high score)
          3. Intra-list diversity: 1 − max_cos_sim(article, selected_embs)
          4. Category saturation: −cat_count_in_selected / max(1, total_selected)

        serendipity(i) = 0.35×unexpect + 0.25×pop_novelty + 0.25×intra_div
                       − 0.15×cat_saturation
        final(i)       = (1−β)×relevance + β×serendipity(i)

        Greedy selection at each step (prevents niche-cluster formation).

        Args:
            candidates:        (news_id, relevance_score) list
            user_history:      User's click history
            k:                 Items to select
            beta:              Serendipity weight
            inject_candidates: Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, final_score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        candidate_ids = [nid for nid, _ in candidates if nid in self.news_id_to_idx]
        relevance_scores = {nid: score for nid, score in candidates}

        if not candidate_ids:
            return []

        # Pre-compute user centroid from history embeddings
        history_embs = [
            self.embeddings[self.news_id_to_idx[nid]]
            for nid in user_history
            if nid in self.news_id_to_idx
        ]
        user_centroid = np.mean(history_embs, axis=0) if history_embs else None

        # Pre-compute candidate embeddings
        candidate_embs = {
            nid: self.embeddings[self.news_id_to_idx[nid]]
            for nid in candidate_ids
        }

        # Max popularity for normalisation
        pops = [self.popularity_scores.get(nid, 0.0) for nid in candidate_ids]
        max_pop = max(pops) if pops else 1.0
        if max_pop == 0:
            max_pop = 1.0

        # Greedy selection
        selected: List[str] = []
        selected_embs: List[np.ndarray] = []
        selected_cats: Counter = Counter()
        remaining = set(candidate_ids)

        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None
            n_selected = max(1, len(selected))

            for nid in remaining:
                emb = candidate_embs[nid]
                relevance = relevance_scores.get(nid, 0.0)
                cat = self.news_categories.get(nid, "")

                # 1. Unexpectedness: distance from user mean taste
                if user_centroid is not None:
                    cos_sim = float(emb @ user_centroid)
                    unexpectedness = 1.0 - cos_sim
                else:
                    unexpectedness = 0.5

                # 2. Popularity novelty: niche articles score higher
                pop = self.popularity_scores.get(nid, 0.0)
                pop_novelty = 1.0 - (pop / max_pop)

                # 3. Intra-list diversity: distance from already selected
                if selected_embs:
                    sims = [float(emb @ sel_emb) for sel_emb in selected_embs]
                    intra_div = 1.0 - max(sims)
                else:
                    intra_div = 1.0

                # 4. Category saturation penalty
                cat_saturation = selected_cats.get(cat, 0) / n_selected

                serendipity = (
                    0.35 * unexpectedness
                    + 0.25 * pop_novelty
                    + 0.25 * intra_div
                    - 0.15 * cat_saturation
                )

                final_score = (1 - beta) * relevance + beta * serendipity

                if final_score > best_score:
                    best_score = final_score
                    best_item = nid

            if best_item:
                selected.append(best_item)
                selected_embs.append(candidate_embs[best_item])
                remaining.remove(best_item)
                best_cat = self.news_categories.get(best_item, "")
                if best_cat:
                    selected_cats[best_cat] += 1

        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

    # -----------------------------------------------------------------------
    # Algorithm 5: Bounded Greedy (hard per-category cap)
    # -----------------------------------------------------------------------

    def bounded_greedy_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        max_per_category: int = 2,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Bounded greedy re-ranking: hard per-category cap.

        Strongest diversity guarantee. Sorts candidates by relevance desc,
        then greedily selects the next candidate where the category count
        has not yet reached max_per_category.

        Default max_per_category=2 for k=10 guarantees ≥5 distinct categories.

        Args:
            candidates:        (news_id, relevance_score) list
            user_history:      User's click history
            k:                 Items to select
            max_per_category:  Hard cap per category
            inject_candidates: Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, relevance_score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        # Sort by relevance descending
        sorted_candidates = sorted(
            [(nid, score) for nid, score in candidates if nid in self.news_categories],
            key=lambda x: x[1],
            reverse=True,
        )

        selected: List[Tuple[str, float]] = []
        cat_counts: Counter = Counter()

        for nid, score in sorted_candidates:
            if len(selected) >= k:
                break

            cat = self.news_categories.get(nid, "")
            if cat_counts.get(cat, 0) < max_per_category:
                selected.append((nid, score))
                if cat:
                    cat_counts[cat] += 1

        # Fallback: if strict cap left fewer than k items, relax it
        if len(selected) < k:
            selected_ids = {nid for nid, _ in selected}
            for nid, score in sorted_candidates:
                if len(selected) >= k:
                    break
                if nid not in selected_ids:
                    selected.append((nid, score))

        return selected[:k]

    # -----------------------------------------------------------------------
    # Algorithm 6: Max Coverage
    # -----------------------------------------------------------------------

    def max_coverage_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        coverage_budget: float = 0.6,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Max Coverage re-ranking: reserve slots for category coverage.

        Phase 1 — Coverage slots (coverage_budget × k, default 6 of 10):
          - Build per_cat_best = {cat → highest-relevance candidate}
          - Priority: (1) categories not in user history first,
                      (2) least-seen history categories
          - Pick best candidate per category in priority order

        Phase 2 — Relevance slots (remaining slots):
          - Fill with highest remaining relevance candidates

        Guarantees up to 6 distinct categories covered.

        Args:
            candidates:        (news_id, relevance_score) list
            user_history:      User's click history
            k:                 Items to select
            coverage_budget:   Fraction of slots reserved for coverage (0–1)
            inject_candidates: Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        valid_candidates = [
            (nid, score) for nid, score in candidates
            if nid in self.news_categories
        ]

        if not valid_candidates:
            return candidates[:k]

        relevance_scores = {nid: score for nid, score in valid_candidates}

        # Build per-category best candidate (by relevance)
        per_cat_best: Dict[str, Tuple[str, float]] = {}
        for nid, score in sorted(valid_candidates, key=lambda x: x[1], reverse=True):
            cat = self.news_categories[nid]
            if cat not in per_cat_best:
                per_cat_best[cat] = (nid, score)

        # User history category counts (for priority ordering)
        history_cat_counts = Counter(
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        )

        # Priority: unseen categories first, then least-seen
        def category_priority(cat: str) -> Tuple[int, int]:
            count = history_cat_counts.get(cat, 0)
            return (0 if count == 0 else 1, count)

        sorted_cats = sorted(per_cat_best.keys(), key=category_priority)

        # Phase 1: fill coverage slots
        n_coverage = int(coverage_budget * k)
        selected: List[Tuple[str, float]] = []
        selected_ids: set = set()

        for cat in sorted_cats:
            if len(selected) >= n_coverage:
                break
            nid, score = per_cat_best[cat]
            selected.append((nid, score))
            selected_ids.add(nid)

        # Phase 2: fill remaining slots with highest relevance
        n_remaining = k - len(selected)
        if n_remaining > 0:
            leftover = sorted(
                [(nid, score) for nid, score in valid_candidates
                 if nid not in selected_ids],
                key=lambda x: x[1],
                reverse=True,
            )
            selected.extend(leftover[:n_remaining])

        return selected[:k]

    # -----------------------------------------------------------------------
    # Algorithm 7: Composite (Diversity + Calibration + Serendipity + Fairness)
    # -----------------------------------------------------------------------

    def composite_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        w_relevance: float = 0.40,
        w_diversity: float = 0.15,
        w_calibration: float = 0.15,
        w_serendipity: float = 0.15,
        w_fairness: float = 0.15,
        explore_weight: float = 0.30,
        inject_candidates: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Composite re-ranker explicitly addressing all four diversity dimensions:

          1. Diversity (embedding):
               1 - max_cosine_sim(article, already_selected)
               Prevents embedding-similar articles from clustering.

          2. Calibration (user distribution):
               (clip((target_prop(cat) - current_prop(cat)) × n_categories, −1, 1) + 1) / 2
               Target = (1-explore_weight)×user_history_dist + explore_weight×uniform.
               Bidirectional: rewards under-represented and penalises over-represented
               categories, actively discouraging category repetition beyond its fair share.

          3. Serendipity (unexpectedness):
               clip(1 − cosine_sim(article, user_centroid), 0, 1)
               Rewards articles distant from the user's mean taste embedding.
               Wider effective range than (1−cos)/2 for typical news embeddings.

          4. Fairness (system-level corpus calibration):
               max(0, corpus_subcat_prop(subcat) - current_rec_subcat_prop(subcat))
                   × n_subcategories  ∈ [0, 1]
               Rewards articles from subcategories under-represented in the current
               recommendations relative to their proportion in the full corpus.
               This is system-level fairness — the target is the corpus distribution,
               not the user's history. Falls back to category-level if subcategory
               distributions are unavailable.

        Final score:
            score(i) = w_rel  × norm_relevance(i)
                     + w_div  × diversity_score(i)
                     + w_cal  × calibration_score(i)
                     + w_ser  × serendipity_score(i)
                     + w_fair × fairness_score(i)

        Greedy selection. Pool augmented with absent-category articles.

        Args:
            candidates:      (news_id, relevance_score) list
            user_history:    User's click history
            k:               Items to select
            w_relevance:     Weight for relevance component
            w_diversity:     Weight for embedding diversity component
            w_calibration:   Weight for calibration component
            w_serendipity:   Weight for serendipity component
            w_fairness:      Weight for popularity fairness component
            explore_weight:  Uniform prior blend in calibration target (0=pure history)
            inject_candidates: Augment pool with absent-category articles

        Returns:
            Re-ranked (news_id, original_relevance_score) list
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        candidate_ids = [nid for nid, _ in candidates if nid in self.news_id_to_idx]
        relevance_scores = {nid: score for nid, score in candidates}

        if not candidate_ids:
            return []

        # --- Pre-compute user profile statistics ---

        # User category distribution (for calibration target)
        history_cats = [
            self.news_categories[nid]
            for nid in user_history
            if nid in self.news_categories
        ]
        if history_cats:
            cat_counts = Counter(history_cats)
            total = sum(cat_counts.values())
            smoothing = 0.01
            denom = total + smoothing * self.n_categories
            smoothed_hist = {
                cat: (cat_counts.get(cat, 0) + smoothing) / denom
                for cat in self.all_categories
            }
        else:
            smoothed_hist = {cat: 1.0 / self.n_categories for cat in self.all_categories}

        uniform = 1.0 / self.n_categories
        target_dist = {
            cat: (1 - explore_weight) * smoothed_hist[cat] + explore_weight * uniform
            for cat in self.all_categories
        }
        t_total = sum(target_dist.values())
        target_dist = {cat: v / t_total for cat, v in target_dist.items()}

        # User embedding centroid (for serendipity)
        history_embs = [
            self.embeddings[self.news_id_to_idx[nid]]
            for nid in user_history
            if nid in self.news_id_to_idx
        ]
        user_centroid = np.mean(history_embs, axis=0) if history_embs else None

        # Pre-compute candidate embeddings
        candidate_embs = {
            nid: self.embeddings[self.news_id_to_idx[nid]]
            for nid in candidate_ids
        }

        # Normalise relevance to [0, 1] so all components share the same scale
        rel_vals = [relevance_scores.get(nid, 0.0) for nid in candidate_ids]
        rel_min, rel_max = min(rel_vals), max(rel_vals)
        rel_range = max(rel_max - rel_min, 1e-9)
        norm_rel = {
            nid: (relevance_scores.get(nid, 0.0) - rel_min) / rel_range
            for nid in candidate_ids
        }

        # --- Greedy composite selection ---
        selected: List[str] = []
        selected_embs: List[np.ndarray] = []
        selected_cats: Counter = Counter()
        selected_subcats: Counter = Counter()
        remaining = set(candidate_ids)

        for _ in range(min(k, len(candidate_ids))):
            best_score = -np.inf
            best_item = None
            n_selected = max(1, len(selected))

            for nid in remaining:
                cat = self.news_categories.get(nid, "")
                emb = candidate_embs[nid]

                # 1. Relevance (normalised to [0, 1])
                rel = norm_rel.get(nid, 0.0)

                # 2. Diversity: embedding distance from already-selected items
                if selected_embs:
                    max_sim = max(float(emb @ sel_emb) for sel_emb in selected_embs)
                else:
                    max_sim = 0.0
                div_score = min(1.0, max(0.0, 1.0 - max_sim))

                # 3. Calibration: bidirectional gap toward user-adjusted target distribution.
                # Positive (under-represented) → rewards selection; negative (over-represented)
                # → penalises selection. This actively discourages category repetition beyond
                # its target share, unlike the old one-directional clip at 0.
                current_prop = selected_cats.get(cat, 0) / n_selected
                target_prop = target_dist.get(cat, uniform)
                cal_gap = (target_prop - current_prop) * self.n_categories
                cal_score = (min(1.0, max(-1.0, cal_gap)) + 1.0) / 2.0  # Maps [-1,1] → [0,1]

                # 4. Serendipity: distance from user's mean embedding centroid.
                # Using 1 − cos (clipped to [0,1]) instead of (1 − cos)/2 to give
                # a wider score range for typical news embeddings (cos ~ 0.3–0.7).
                if user_centroid is not None:
                    cos_to_centroid = float(np.clip(emb @ user_centroid, -1.0, 1.0))
                    ser_score = min(1.0, max(0.0, 1.0 - cos_to_centroid))
                else:
                    ser_score = 0.5

                # 5. Fairness: scarcity-weighted minority boost.
                # Goal: give corpus-rare subcategories/categories a stronger push
                # than dominant ones (e.g. rare "travel/backpacking" >> common "news/politics").
                # Formula: gap_below_equal_share × scarcity_multiplier
                #   scarcity = uniform / corpus_prop  (rare→high multiplier, dominant→low)
                # This is supply-side fairness — target is equal representation, amplified
                # for how structurally under-served the item is in the corpus.
                # Distinct from calibration (which targets user history distribution).
                subcat = self.news_subcategories.get(nid, "")
                if self.corpus_subcategory_dist and subcat:
                    corpus_subcat_prop = self.corpus_subcategory_dist.get(subcat, 0.0)
                    if corpus_subcat_prop > 0:
                        uniform_sub = 1.0 / max(self.n_subcategories, 1)
                        cur_sub_prop = selected_subcats.get(subcat, 0) / n_selected
                        gap = max(0.0, uniform_sub - cur_sub_prop)
                        scarcity = uniform_sub / corpus_subcat_prop
                        fair_score = min(1.0, gap * scarcity / uniform_sub)
                    else:
                        fair_score = 0.0
                elif self.corpus_category_dist:
                    corpus_cat_prop = self.corpus_category_dist.get(cat, 0.0)
                    if corpus_cat_prop > 0:
                        uniform_cat = 1.0 / max(self.n_categories, 1)
                        cur_cat_prop = selected_cats.get(cat, 0) / n_selected
                        gap = max(0.0, uniform_cat - cur_cat_prop)
                        scarcity = uniform_cat / corpus_cat_prop
                        fair_score = min(1.0, gap * scarcity / uniform_cat)
                    else:
                        fair_score = 0.0
                else:
                    fair_score = 0.5

                score = (
                    w_relevance   * rel
                    + w_diversity   * div_score
                    + w_calibration * cal_score
                    + w_serendipity * ser_score
                    + w_fairness    * fair_score
                )

                if score > best_score:
                    best_score = score
                    best_item = nid

            if best_item:
                selected.append(best_item)
                selected_embs.append(candidate_embs[best_item])
                remaining.remove(best_item)
                best_cat = self.news_categories.get(best_item, "")
                if best_cat:
                    selected_cats[best_cat] += 1
                best_subcat = self.news_subcategories.get(best_item, "")
                if best_subcat:
                    selected_subcats[best_subcat] += 1

        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

    # -----------------------------------------------------------------------
    # Unified Interface
    # -----------------------------------------------------------------------

    def random_rerank(
        self,
        candidates: List[Tuple[str, float]],
        user_history: List[str],
        k: int = 10,
        inject_candidates: bool = True,
        seed: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Random re-ranker — purely random ordering of the candidate pool.

        Used as a lower-bound baseline to contextualise diversity and accuracy
        metrics for all other methods. Expected behaviour:
          - AUC ≈ 0.5  (random ranking gives no accuracy signal)
          - High diversity metrics (random selection samples broadly across
            categories, giving good coverage and low Gini)

        Args:
            candidates:        Baseline candidates (news_id, relevance_score)
            user_history:      User's click history (unused, kept for interface parity)
            k:                 Number of items to select
            inject_candidates: Augment pool with absent-category articles
            seed:              Optional RNG seed for reproducibility

        Returns:
            k randomly selected (news_id, original_relevance_score) pairs
        """
        if not candidates:
            return []

        if inject_candidates:
            candidates = self._inject_diverse_candidates(candidates, user_history)

        candidate_ids = [nid for nid, _ in candidates if nid in self.news_id_to_idx]
        relevance_scores = {nid: score for nid, score in candidates}

        if not candidate_ids:
            return []

        rng = np.random.default_rng(seed)
        shuffled = candidate_ids.copy()
        rng.shuffle(shuffled)
        selected = shuffled[:k]
        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

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
            method:        'random', 'mmr', 'xquad', 'calibrated', 'serendipity',
                           'bounded_greedy', 'max_coverage', 'composite'
            **kwargs:      Method-specific parameters

        Returns:
            Re-ranked list
        """
        dispatch = {
            'random': self.random_rerank,
            'mmr': self.mmr_rerank,
            'xquad': self.xquad_rerank,
            'calibrated': self.calibrated_rerank,
            'serendipity': self.serendipity_rerank,
            'bounded_greedy': self.bounded_greedy_rerank,
            'max_coverage': self.max_coverage_rerank,
            'composite': self.composite_rerank,
        }

        if method not in dispatch:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Choose from: {list(dispatch.keys())}"
            )

        return dispatch[method](candidates, user_history, k, **kwargs)


if __name__ == "__main__":
    logger.info("Diversity Re-ranking Algorithms — Phase 4")
    logger.info(
        "8 algorithms: Random, MMR, xQuAD, Calibrated, Serendipity, "
        "Bounded Greedy, Max Coverage, Composite"
    )
