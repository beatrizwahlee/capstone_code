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
    ):
        """
        Args:
            baseline_recommender:  Trained BaselineRecommender from Phase 2
            embeddings:            Final embeddings from Phase 1
            news_id_to_idx:        news_id → row index mapping
            news_categories:       news_id → category mapping
            popularity_scores:     news_id → popularity score (optional)
        """
        self.baseline = baseline_recommender
        self.embeddings = embeddings
        self.news_id_to_idx = news_id_to_idx
        self.news_categories = news_categories
        self.popularity_scores = popularity_scores or {}

        # All unique categories
        self.all_categories = sorted(set(news_categories.values()))
        self.n_categories = len(self.all_categories)

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

        # Minimum score so injected articles don't dominate
        scores = [s for _, s in candidates if s > 0]
        min_score = min(scores) if scores else 0.01
        injected_score = min_score * 0.5

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
               max(0, target_prop(cat) - current_prop(cat)) × n_categories
               Target = (1-explore_weight)×user_history_dist + explore_weight×uniform.
               Rewards filling categories under-represented relative to preferences.

          3. Serendipity (unexpectedness):
               (1 - cosine_sim(article, user_centroid)) / 2  ∈ [0, 1]
               Rewards articles distant from the user's mean taste embedding.

          4. Fairness (popularity miscalibration):
               1 - |pop_article - pop_user_mean| / max_pop  ∈ [0, 1]
               User-side fairness: prefers articles whose popularity level
               matches what the user historically reads (avoids forcing
               mainstream or niche regardless of user preference).

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

        # User's mean popularity (for fairness)
        history_pops = [
            self.popularity_scores.get(nid, 0.0)
            for nid in user_history
            if nid in self.popularity_scores
        ]
        user_pop_mean = float(np.mean(history_pops)) if history_pops else 0.0

        # Pre-compute candidate embeddings and popularity range
        candidate_embs = {
            nid: self.embeddings[self.news_id_to_idx[nid]]
            for nid in candidate_ids
        }
        all_pops = [self.popularity_scores.get(nid, 0.0) for nid in candidate_ids]
        max_pop = max(all_pops) if all_pops else 1.0
        if max_pop == 0:
            max_pop = 1.0

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

                # 3. Calibration: coverage gap toward user-adjusted target distribution
                current_prop = selected_cats.get(cat, 0) / n_selected
                target_prop = target_dist.get(cat, uniform)
                # Multiply by n_categories to normalise gap to [0, 1]
                cal_score = min(1.0, max(0.0, (target_prop - current_prop) * self.n_categories))

                # 4. Serendipity: distance from user's mean embedding centroid
                if user_centroid is not None:
                    cos_to_centroid = float(np.clip(emb @ user_centroid, -1.0, 1.0))
                    ser_score = (1.0 - cos_to_centroid) / 2.0  # Map [-1, 1] → [0, 1]
                else:
                    ser_score = 0.5

                # 5. Fairness: popularity level matching user's historical preference
                pop = self.popularity_scores.get(nid, 0.0)
                pop_diff = abs(pop - user_pop_mean) / max_pop
                fair_score = 1.0 - pop_diff  # 1.0 = perfect match, 0.0 = max mismatch

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

        return [(nid, relevance_scores.get(nid, 0.0)) for nid in selected]

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
            method:        'mmr', 'xquad', 'calibrated', 'serendipity',
                           'bounded_greedy', 'max_coverage', 'composite'
            **kwargs:      Method-specific parameters

        Returns:
            Re-ranked list
        """
        dispatch = {
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
        "7 algorithms: MMR, xQuAD, Calibrated, Serendipity, "
        "Bounded Greedy, Max Coverage, Composite"
    )
