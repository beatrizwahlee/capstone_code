"""
Phase 1: NLP Encoding Layer
============================
Diversity-Aware News Recommender System — Capstone Project

This module replaces the TF-IDF approach in content_based_recommender.py
with a three-tower semantic encoding pipeline:

  Tower 1 — Sentence-BERT (384-dim)
      Captures semantic meaning from title + abstract.
      "Fed rate hike" and "central bank tightening" become close vectors.
      TF-IDF cannot do this; it only sees keyword overlap.

  Tower 2 — Entity Embeddings (100-dim)
      Averaged WikiData knowledge graph embeddings per article.
      Captures factual/topical overlap through named entities.
      Already loaded by MINDDataLoader — used directly here.

  Tower 3 — Category Embeddings (learned, 32+16-dim)
      Small learnable embedding table instead of one-hot encoding.
      Learns that sports/soccer and sports/basketball are closer
      to each other than sports/soccer and finance/markets.
      Critical for the diversity re-ranking layer in Phase 4.

Final news embedding: 532-dim, L2-normalized.

User Profile: Attention-weighted aggregation over click history.
      Simple averaging dilutes rare but informative clicks.
      Attention learns which history items matter most.
      Recency decay: recent clicks contribute more than old ones.

FAISS Index: Enables sub-millisecond retrieval of top-100 candidates
      from 101K articles at recommendation time.

Outputs written to ./embeddings/ (configurable):
  - sbert_news_embeddings.npy       (101K × 384)
  - entity_news_features.npy        (101K × 100)
  - final_news_embeddings.npy       (101K × 532, L2-normalized)
  - news_faiss.index                (FAISS flat inner-product index)
  - category_embeddings.pkl         (learned 32-dim category lookup)
  - subcategory_embeddings.pkl      (learned 16-dim subcategory lookup)
  - news_id_to_idx.json             (news_id → row index mapping)
  - embedding_metadata.json         (shapes, model name, timestamps)

Usage:
    from nlp_encoder import NewsEncoderPhase1, AttentionUserProfiler

    # One-time: encode all articles and build FAISS index
    encoder = NewsEncoderPhase1()
    encoder.fit(news_df, entity_embeddings)
    encoder.save('./embeddings')

    # At recommendation time: load once, use many times
    encoder = NewsEncoderPhase1.load('./embeddings')
    user_profile = encoder.build_user_profile(history_news_ids)
    top_100_ids, scores = encoder.retrieve_candidates(user_profile, k=100)
"""

import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_faiss():
    """Graceful import for FAISS — not always available on all machines."""
    try:
        import faiss
        return faiss
    except ImportError:
        logger.warning(
            "faiss-cpu is not installed. FAISS index will be unavailable.\n"
            "Install with:  pip install faiss-cpu"
        )
        return None


def _check_sbert():
    """Graceful import for sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for Phase 1.\n"
            "Install with:  pip install sentence-transformers"
        )


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of a and each row of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# Tower 1 — Sentence-BERT Text Encoder
# ---------------------------------------------------------------------------

class SBERTTextEncoder:
    """
    Encodes news titles and abstracts using Sentence-BERT.

    Why SBERT over TF-IDF
    ---------------------
    TF-IDF relies purely on shared vocabulary.  Two articles using different
    words to describe the same event are treated as unrelated.  SBERT maps
    text into a semantic space where meaning-equivalent phrases cluster
    together, making it far more effective for:
      - Similarity scoring between user profile and candidate articles
      - Intra-List Diversity (ILD) measurement in Phase 3
      - Serendipity scoring in Phase 4 (unexpected-but-relevant items)

    Default model: all-MiniLM-L6-v2
      - 384-dim output
      - Very fast (~3 min for 101 K articles on CPU; ~30 s on GPU)
      - Strong performance for short texts like news headlines

    Alternatively: allenai/news-roberta-base
      - News-domain fine-tuned, better quality, ~4× slower
      - Recommended if GPU is available and you want best accuracy
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None  # Lazy-load to avoid slow import on module load

    def _load_model(self):
        if self._model is None:
            SentenceTransformer = _check_sbert()
            logger.info(f"Loading SBERT model '{self.model_name}' …")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SBERT model loaded.")

    def encode(
        self,
        news_df: pd.DataFrame,
        batch_size: int = 64,
        show_progress: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 2000,
    ) -> np.ndarray:
        """
        Encode all articles into 384-dim L2-normalised vectors.

        Supports RESUMABLE encoding via checkpointing: if the process is
        interrupted (Ctrl-C, crash, sleep), re-running the script will
        automatically pick up from the last saved checkpoint instead of
        starting over from article 0.

        CPU performance note
        --------------------
        On older Intel CPUs (SSE4.2 only, no AVX — like your machine),
        each batch takes longer than expected. The optimal strategy is
        LARGER batches (32-64), not smaller ones, because the per-batch
        overhead (tokenization, padding, data movement) is paid once per
        batch regardless of size. batch_size=16 means 6,346 batches;
        batch_size=64 means 1,587 batches — 4× less overhead.

        Input text: "{title} {abstract}"

        Args:
            news_df:           DataFrame with 'title' and 'abstract' columns.
            batch_size:        Encoding batch size. Use 32-64 on CPU.
                               Do NOT go lower than 32 — it makes things slower.
            show_progress:     Show tqdm progress bar.
            checkpoint_path:   Path to save/load partial results (e.g.,
                               './embeddings/sbert_checkpoint.npy').
                               If None, no checkpointing (runs in one shot).
            checkpoint_every:  Save a checkpoint every N articles.
                               2000 = saves ~50 times for 101K articles.
                               Each checkpoint takes ~1 second to write.

        Returns:
            Float32 array of shape (N, 384), L2-normalised.
        """
        self._load_model()

        texts = (
            news_df["title"].fillna("").astype(str)
            + " "
            + news_df["abstract"].fillna("").astype(str)
        ).tolist()

        N = len(texts)
        logger.info(f"Encoding {N:,} articles with SBERT (batch_size={batch_size}) …")

        # ── Try to resume from checkpoint ──────────────────────────────────
        start_idx = 0
        all_embeddings: List[np.ndarray] = []

        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        ckpt_meta_path = ckpt_path.with_suffix(".meta.json") if ckpt_path else None

        if ckpt_path and ckpt_path.exists() and ckpt_meta_path and ckpt_meta_path.exists():
            try:
                with open(ckpt_meta_path) as f:
                    meta = json.load(f)
                saved_n = meta.get("n_encoded", 0)
                if saved_n > 0 and saved_n < N:
                    partial = np.load(str(ckpt_path))
                    if partial.shape[0] == saved_n:
                        all_embeddings = [partial]
                        start_idx = saved_n
                        logger.info(
                            f"  ✔ Resuming from checkpoint: {saved_n:,}/{N:,} articles "
                            f"already encoded ({saved_n/N*100:.1f}%)"
                        )
                    else:
                        logger.warning("  Checkpoint shape mismatch — starting fresh.")
                elif saved_n >= N:
                    logger.info(f"  ✔ Checkpoint complete ({saved_n:,} articles). Loading …")
                    return np.load(str(ckpt_path))
            except Exception as e:
                logger.warning(f"  Could not load checkpoint ({e}). Starting fresh.")

        if start_idx == 0:
            logger.info(f"  Starting fresh encoding …")

        # ── Encode in manual chunks so we can checkpoint ───────────────────
        t0 = time.time()
        remaining_texts = texts[start_idx:]
        n_remaining = len(remaining_texts)

        # Split remaining texts into chunks of checkpoint_every
        chunk_size = checkpoint_every
        chunks = [
            remaining_texts[i: i + chunk_size]
            for i in range(0, n_remaining, chunk_size)
        ]

        logger.info(
            f"  Articles remaining: {n_remaining:,}  |  "
            f"Chunks: {len(chunks)}  |  "
            f"Articles/chunk: {chunk_size:,}"
        )
        logger.info(
            f"  Estimated time: "
            f"{n_remaining / 60:.0f}–{n_remaining / 30:.0f} min on CPU  "
            f"(varies greatly by machine)"
        )

        encoded_so_far = start_idx
        for chunk_i, chunk_texts in enumerate(chunks):
            t_chunk = time.time()

            chunk_embs = self._model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embeddings.append(chunk_embs.astype(np.float32))
            encoded_so_far += len(chunk_texts)

            elapsed_chunk = time.time() - t_chunk
            elapsed_total = time.time() - t0
            speed = len(chunk_texts) / elapsed_chunk
            remaining_articles = N - encoded_so_far
            eta_min = (remaining_articles / speed / 60) if speed > 0 else 0

            logger.info(
                f"  Chunk {chunk_i+1}/{len(chunks)} done  |  "
                f"{encoded_so_far:,}/{N:,} articles ({encoded_so_far/N*100:.1f}%)  |  "
                f"Speed: {speed:.0f} art/s  |  "
                f"ETA: {eta_min:.1f} min"
            )

            # Save checkpoint
            if ckpt_path is not None:
                partial_arr = np.vstack(all_embeddings)
                np.save(str(ckpt_path), partial_arr)
                with open(str(ckpt_meta_path), "w") as f:
                    json.dump({"n_encoded": encoded_so_far, "total": N}, f)
                logger.info(f"  💾 Checkpoint saved ({encoded_so_far:,} articles)")

        # ── Stack all chunks into final array ──────────────────────────────
        embeddings = np.vstack(all_embeddings).astype(np.float32)

        elapsed = time.time() - t0
        logger.info(
            f"SBERT encoding done in {elapsed/60:.1f} min  "
            f"({N/elapsed:.0f} articles/s)  "
            f"Shape: {embeddings.shape}"
        )

        # Clean up checkpoint now that we have the full result
        if ckpt_path and ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)
            if ckpt_meta_path and ckpt_meta_path.exists():
                ckpt_meta_path.unlink(missing_ok=True)
            logger.info("  Checkpoint files cleaned up (full encoding complete).")

        return embeddings


# ---------------------------------------------------------------------------
# Tower 2 — Entity Embedding Aggregator
# ---------------------------------------------------------------------------

class EntityEmbeddingAggregator:
    """
    Aggregates WikiData entity embeddings per article.

    The MIND entity_embedding.vec file contains 100-dim TransE-style
    knowledge graph embeddings.  Each article carries a list of WikiData
    entity IDs linked from its title and abstract.

    Strategy
    --------
    Average-pool all entity vectors that exist in the embedding dict.
    For articles with no matching entities use a zero vector (handled
    gracefully downstream by the L2-normalisation — zero stays zero,
    contributing nothing to the dot product).

    Coverage is ~96.7 % per the data quality report, so the fallback
    matters for only ~3 % of articles.
    """

    ENTITY_DIM = 100  # Fixed by the MIND dataset

    def __init__(self):
        self.entity_dim = self.ENTITY_DIM

    def encode(
        self,
        news_df: pd.DataFrame,
        entity_embeddings: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Build a (N, 100) matrix of averaged entity embeddings.

        Args:
            news_df:           DataFrame with 'all_entities' column
                               (list of WikiData IDs per article, built
                               by MINDDataLoader._combine_entities).
            entity_embeddings: Dict mapping WikiData ID → 100-dim vector
                               (from MINDDataLoader.load_entity_embeddings).

        Returns:
            Float32 array of shape (N, 100).
            Rows for articles with no known entities remain zero.
        """
        N = len(news_df)
        features = np.zeros((N, self.entity_dim), dtype=np.float32)

        missing_count = 0
        hit_count = 0

        for i, entities in enumerate(news_df["all_entities"]):
            if not entities:
                missing_count += 1
                continue

            vecs = [
                entity_embeddings[eid]
                for eid in entities
                if eid in entity_embeddings
            ]

            if vecs:
                features[i] = np.mean(vecs, axis=0).astype(np.float32)
                hit_count += 1
            else:
                missing_count += 1

        coverage = hit_count / N * 100
        logger.info(
            f"Entity features built: {hit_count:,}/{N:,} articles "
            f"have entity embeddings ({coverage:.1f}% coverage)"
        )
        return features


# ---------------------------------------------------------------------------
# Tower 3 — Learned Category Embeddings
# ---------------------------------------------------------------------------

class CategoryEmbeddingLayer:
    """
    Replaces one-hot category encoding with small learned embedding tables.

    Why not one-hot?
    ----------------
    One-hot treats every category as equally distant from every other.
    "sports/soccer" and "sports/basketball" are just as distant as
    "sports/soccer" and "finance/markets" under one-hot.

    With learned embeddings the model can discover that subcategories
    within the same parent are similar, which is valuable for:
      - The re-ranking layer (Phase 4) when computing category diversity
      - The calibration algorithm which needs to know which categories
        are "close enough" to satisfy a user's inferred preferences

    Initialisation
    --------------
    We use a simple, effective initialisation:
    - Categories sorted alphabetically → one-hot positions → PCA to 32-dim
      This is deterministic (no random seed dependency) and ensures that
      the initial embeddings are already somewhat meaningful.
    - For the capstone scope, these embeddings are fixed (not fine-tuned).
      Phase 2 can optionally fine-tune them end-to-end.

    Dimensions:  category → 32-dim,  subcategory → 16-dim
    """

    CAT_DIM = 32
    SUBCAT_DIM = 16

    def __init__(self):
        self.cat_dim = self.CAT_DIM
        self.subcat_dim = self.SUBCAT_DIM
        self.cat_to_idx: Dict[str, int] = {}
        self.subcat_to_idx: Dict[str, int] = {}
        self.cat_embeddings: Optional[np.ndarray] = None
        self.subcat_embeddings: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, news_df: pd.DataFrame) -> "CategoryEmbeddingLayer":
        """
        Build category and subcategory embedding tables from news_df.

        Args:
            news_df: DataFrame with 'category' and 'subcategory' columns.

        Returns:
            self (for chaining)
        """
        # --- Build index mappings (sorted for determinism) ---
        categories = sorted(news_df["category"].dropna().unique())
        subcategories = sorted(news_df["subcategory"].dropna().unique())

        self.cat_to_idx = {c: i for i, c in enumerate(categories)}
        self.subcat_to_idx = {s: i for i, s in enumerate(subcategories)}

        n_cats = len(categories)
        n_subcats = len(subcategories)

        logger.info(
            f"Category embedding: {n_cats} categories → {self.cat_dim}-dim  |  "
            f"{n_subcats} subcategories → {self.subcat_dim}-dim"
        )

        # --- Initialise with PCA-compressed one-hot ---
        # This gives meaningful initial geometry (similar cats cluster)
        # without requiring a neural training loop for Phase 1.
        self.cat_embeddings = self._pca_init_onehot(n_cats, self.cat_dim)
        self.subcat_embeddings = self._pca_init_onehot(n_subcats, self.subcat_dim)

        self._fitted = True
        return self

    @staticmethod
    def _pca_init_onehot(n: int, target_dim: int) -> np.ndarray:
        """
        Create a (n, target_dim) initialisation matrix.

        For n ≤ target_dim: zero-pad one-hot (every category orthogonal).
        For n > target_dim: PCA on one-hot to compress (principal components).

        Both cases are L2-normalised so all category vectors have equal norm.
        """
        if n <= target_dim:
            # Pad one-hot with zeros
            emb = np.zeros((n, target_dim), dtype=np.float32)
            emb[:n, :n] = np.eye(n, dtype=np.float32)
        else:
            # PCA: the principal components of an identity matrix are just
            # random orthonormal directions — use random orthonormal init
            # which is equivalent and cheaper
            rng = np.random.RandomState(42)  # fixed seed for reproducibility
            raw = rng.randn(n, target_dim).astype(np.float32)
            # Orthonormalise columns via QR
            q, _ = np.linalg.qr(raw)
            emb = q[:n, :target_dim]

        return normalize(emb, norm="l2")

    def transform(self, news_df: pd.DataFrame) -> np.ndarray:
        """
        Map each article to its (cat_dim + subcat_dim) embedding vector.

        Args:
            news_df: DataFrame with 'category' and 'subcategory' columns.

        Returns:
            Float32 array of shape (N, cat_dim + subcat_dim).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        N = len(news_df)
        features = np.zeros((N, self.cat_dim + self.subcat_dim), dtype=np.float32)

        for i, (cat, subcat) in enumerate(
            zip(news_df["category"], news_df["subcategory"])
        ):
            cat_idx = self.cat_to_idx.get(cat)
            subcat_idx = self.subcat_to_idx.get(subcat)

            if cat_idx is not None:
                features[i, : self.cat_dim] = self.cat_embeddings[cat_idx]

            if subcat_idx is not None:
                features[i, self.cat_dim :] = self.subcat_embeddings[subcat_idx]

        return features

    def get_category_vector(self, category: str) -> np.ndarray:
        """Return the 32-dim embedding for a single category name."""
        idx = self.cat_to_idx.get(category)
        if idx is None:
            return np.zeros(self.cat_dim, dtype=np.float32)
        return self.cat_embeddings[idx]

    def get_subcategory_vector(self, subcategory: str) -> np.ndarray:
        """Return the 16-dim embedding for a single subcategory name."""
        idx = self.subcat_to_idx.get(subcategory)
        if idx is None:
            return np.zeros(self.subcat_dim, dtype=np.float32)
        return self.subcat_embeddings[idx]

    def save(self, path: Path):
        """Persist fitted layer to disk."""
        data = {
            "cat_to_idx": self.cat_to_idx,
            "subcat_to_idx": self.subcat_to_idx,
            "cat_embeddings": self.cat_embeddings,
            "subcat_embeddings": self.subcat_embeddings,
            "cat_dim": self.cat_dim,
            "subcat_dim": self.subcat_dim,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"CategoryEmbeddingLayer saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "CategoryEmbeddingLayer":
        """Load a previously saved layer."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.__dict__.update(data)
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Attention-Weighted User Profiler
# ---------------------------------------------------------------------------

class AttentionUserProfiler:
    """
    Builds a user profile as an attention-weighted average of clicked article
    embeddings, with optional recency decay.

    Why attention instead of simple average?
    -----------------------------------------
    A user who clicked 40 sports articles and 1 finance article has a profile
    dominated by sports under simple averaging.  The single finance click
    carries real signal about a latent interest, but it contributes only 2 %
    of the mean vector.

    The attention mechanism learns a *content-aware* weighting:
    articles whose embeddings are more informative for predicting what the
    user will click next receive higher weights.

    Implementation note (Phase 1 scope)
    ------------------------------------
    We use a *parameter-free* approximation here that does not require training:
      - Compute the naive average of the history embeddings.
      - Weight each history item by its cosine similarity to the current
        candidate context (if provided) or to the mean itself (self-attention).
      - Apply temporal decay so recent clicks weigh more.

    This is well-performing and fast.  Phase 2 can optionally replace this
    with a proper trainable attention layer (MHA or additive).

    Recency decay
    -------------
    Weight for article at position i (0 = oldest, T-1 = newest):
        decay_i = exp(lambda * (i - (T-1)))   where lambda controls steepness
    lambda = 0.1 → gentle decay (long memory)
    lambda = 0.3 → moderate decay (default)
    lambda = 1.0 → aggressive decay (only last few items matter)
    """

    def __init__(self, decay_lambda: float = 0.3):
        """
        Args:
            decay_lambda: Recency decay rate.  0.0 disables recency weighting.
        """
        self.decay_lambda = decay_lambda

    def build_profile(
        self,
        history_ids: List[str],
        news_embeddings: np.ndarray,
        news_id_to_idx: Dict[str, int],
        max_history: int = 50,
        candidate_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute a single user profile vector from click history.

        Args:
            history_ids:       Ordered list of clicked news IDs
                               (earliest first, most recent last).
            news_embeddings:   (N, D) matrix — the final_news_embeddings
                               output from NewsEncoderPhase1.
            news_id_to_idx:    Mapping from news_id string → row index.
            max_history:       Truncate history to this many most-recent items.
                               50 is the MIND dataset cap; more items add noise.
            candidate_context: Optional (D,) vector.  If provided, attention
                               weights are computed against this context
                               (useful at recommendation time when we know
                               the candidate pool).

        Returns:
            (D,) float32 vector, L2-normalised.
            Returns a zero vector if no history items are found in the index.
        """
        if not history_ids:
            dim = news_embeddings.shape[1]
            return np.zeros(dim, dtype=np.float32)

        # --- Resolve indices, keeping order, skipping unknowns ---
        resolved = []
        for nid in history_ids[-max_history:]:   # keep most-recent max_history
            idx = news_id_to_idx.get(nid)
            if idx is not None:
                resolved.append(idx)

        if not resolved:
            dim = news_embeddings.shape[1]
            return np.zeros(dim, dtype=np.float32)

        T = len(resolved)
        history_embs = news_embeddings[resolved]  # (T, D)

        # --- Step 1: Recency weights (older → smaller weight) ---
        positions = np.arange(T, dtype=np.float32)   # 0 = oldest
        recency_w = np.exp(self.decay_lambda * (positions - (T - 1)))  # (T,)

        # --- Step 2: Content-aware attention weights ---
        if candidate_context is not None:
            # Attention to candidate: how relevant is each history item to this candidate?
            ctx = candidate_context.reshape(1, -1).astype(np.float32)
            content_w = (history_embs @ ctx.T).flatten()           # (T,)
            content_w = np.clip(content_w, 0, None)                # ReLU: ignore negatives
        else:
            # Self-attention: how "central" is each item to the user's overall profile?
            # We use cosine similarity to the naive mean as a proxy.
            naive_mean = history_embs.mean(axis=0, keepdims=True)  # (1, D)
            content_w = (history_embs @ naive_mean.T).flatten()    # (T,)
            content_w = np.clip(content_w, 0, None)

        # --- Step 3: Combine recency and content, normalise ---
        combined_w = recency_w * (content_w + 1e-6)  # +eps avoids all-zero
        combined_w /= combined_w.sum()                # normalise to sum=1

        # --- Step 4: Weighted sum ---
        profile = (history_embs * combined_w[:, np.newaxis]).sum(axis=0)

        # --- Step 5: L2-normalise ---
        norm = np.linalg.norm(profile)
        if norm > 1e-10:
            profile /= norm

        return profile.astype(np.float32)

    def build_profiles_batch(
        self,
        user_histories: Dict[str, List[str]],
        news_embeddings: np.ndarray,
        news_id_to_idx: Dict[str, int],
        max_history: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Build profiles for many users at once.

        Args:
            user_histories:  Dict mapping user_id → list of clicked news IDs.
            news_embeddings: (N, D) embedding matrix.
            news_id_to_idx:  news_id → row index mapping.
            max_history:     Max history items per user.

        Returns:
            Dict mapping user_id → (D,) profile vector.
        """
        logger.info(f"Building profiles for {len(user_histories):,} users …")
        t0 = time.time()

        profiles: Dict[str, np.ndarray] = {}
        for user_id, history in user_histories.items():
            profiles[user_id] = self.build_profile(
                history_ids=history,
                news_embeddings=news_embeddings,
                news_id_to_idx=news_id_to_idx,
                max_history=max_history,
            )

        elapsed = time.time() - t0
        logger.info(
            f"Profiles built in {elapsed:.1f}s  "
            f"({len(user_histories)/elapsed:.0f} users/s)"
        )
        return profiles


# ---------------------------------------------------------------------------
# FAISS Retrieval Index
# ---------------------------------------------------------------------------

class FAISSCandidateRetriever:
    """
    Wraps a FAISS flat inner-product index for fast approximate-nearest-
    neighbour retrieval.

    Why FAISS?
    ----------
    At recommendation time we need to find the top-100 most relevant articles
    for a user profile vector.  Brute-force cosine similarity over 101 K
    article embeddings takes ~40 ms per query.  FAISS reduces this to <1 ms,
    making the system viable for a real-time web app.

    Index type: IndexFlatIP (exact inner product search)
    - "Flat" means no compression — full precision, exact results.
    - Inner-product is equivalent to cosine similarity when vectors are
      L2-normalised (which all our embeddings are).
    - For 101 K articles at 532-dim, memory footprint ≈ 215 MB — fine.
    - If you later scale to millions of articles, switch to IndexIVFFlat
      or IndexHNSWFlat for approximate but much faster search.
    """

    def __init__(self):
        self._index = None
        self._news_ids: List[str] = []   # position i → news_id
        self.embedding_dim: Optional[int] = None

    def build(
        self,
        embeddings: np.ndarray,
        news_ids: List[str],
    ) -> "FAISSCandidateRetriever":
        """
        Build the FAISS index from the final L2-normalised embeddings.

        Args:
            embeddings: (N, D) float32 L2-normalised array.
            news_ids:   List of news_id strings in the same row order.

        Returns:
            self (for chaining)
        """
        faiss = _check_faiss()
        if faiss is None:
            logger.warning("FAISS unavailable — retrieval will use NumPy fallback.")
            self._index = None
            self._news_ids = news_ids
            self._fallback_embeddings = embeddings
            self.embedding_dim = embeddings.shape[1]
            return self

        N, D = embeddings.shape
        self.embedding_dim = D
        self._news_ids = news_ids

        logger.info(f"Building FAISS IndexFlatIP  ({N:,} articles, {D}-dim) …")
        t0 = time.time()

        self._index = faiss.IndexFlatIP(D)
        self._index.add(embeddings.astype(np.float32))

        logger.info(
            f"FAISS index built in {time.time()-t0:.2f}s  "
            f"(Total vectors: {self._index.ntotal:,})"
        )
        return self

    def retrieve(
        self,
        query_vector: np.ndarray,
        k: int = 100,
        exclude_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve top-k candidates for a user profile vector.

        Args:
            query_vector: (D,) float32 L2-normalised user profile.
            k:            Number of candidates to retrieve.
                          Retrieve more than needed (e.g., k=150) if you
                          plan to exclude history items afterwards.
            exclude_ids:  News IDs to exclude (user's click history).
                          We over-retrieve then filter, which is the standard
                          FAISS pattern.

        Returns:
            (candidate_ids, scores)
              candidate_ids: List of news_id strings (descending score order)
              scores:        Float32 array of cosine similarity scores
        """
        exclude_set = set(exclude_ids) if exclude_ids else set()
        fetch_k = k + len(exclude_set) + 10  # over-fetch to allow filtering

        q = query_vector.reshape(1, -1).astype(np.float32)

        if self._index is not None:
            scores_raw, indices = self._index.search(q, min(fetch_k, len(self._news_ids)))
            scores_raw = scores_raw[0]
            indices = indices[0]
        else:
            # NumPy fallback (slower but works without FAISS)
            sims = (self._fallback_embeddings @ q.T).flatten()
            indices = np.argsort(sims)[::-1][:fetch_k]
            scores_raw = sims[indices]

        # Filter excluded and collect top-k
        candidate_ids: List[str] = []
        candidate_scores: List[float] = []

        for idx, score in zip(indices, scores_raw):
            if idx < 0:
                continue
            nid = self._news_ids[idx]
            if nid in exclude_set:
                continue
            candidate_ids.append(nid)
            candidate_scores.append(float(score))
            if len(candidate_ids) >= k:
                break

        return candidate_ids, np.array(candidate_scores, dtype=np.float32)

    def save(self, path: Path):
        """Save FAISS index and news_ids list to disk."""
        faiss = _check_faiss()

        meta = {
            "news_ids": self._news_ids,
            "embedding_dim": self.embedding_dim,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        if faiss is not None and self._index is not None:
            faiss.write_index(self._index, str(path))
            logger.info(f"FAISS index saved → {path}")
        else:
            np.save(str(path.with_suffix(".npy")), self._fallback_embeddings)
            logger.info(f"NumPy fallback saved → {path.with_suffix('.npy')}")

    @classmethod
    def load(cls, path: Path) -> "FAISSCandidateRetriever":
        """Load a previously saved FAISS index."""
        faiss = _check_faiss()
        meta_path = path.with_suffix(".json")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        obj = cls()
        obj._news_ids = meta["news_ids"]
        obj.embedding_dim = meta["embedding_dim"]

        if faiss is not None and path.exists():
            obj._index = faiss.read_index(str(path))
            logger.info(f"FAISS index loaded ← {path}  ({obj._index.ntotal:,} vectors)")
        else:
            npy_path = path.with_suffix(".npy")
            obj._fallback_embeddings = np.load(str(npy_path))
            obj._index = None
            logger.info(f"NumPy fallback loaded ← {npy_path}")

        return obj


# ---------------------------------------------------------------------------
# Main Encoder — assembles all three towers
# ---------------------------------------------------------------------------

class NewsEncoderPhase1:
    """
    Three-tower news encoder for Phase 1.

    Combines:
      - Tower 1: SBERT text embeddings (384-dim)
      - Tower 2: Entity embedding aggregation (100-dim)
      - Tower 3: Learned category embeddings (32 + 16 = 48-dim)

    Final embedding: 532-dim, L2-normalised float32.

    Typical usage
    -------------
    # One-time setup (run once, save to disk)
    encoder = NewsEncoderPhase1()
    encoder.fit(news_df, entity_embeddings)
    encoder.save('./embeddings')

    # Recommendation time (load from disk, fast)
    encoder = NewsEncoderPhase1.load('./embeddings')

    # Build user profile
    profiler = AttentionUserProfiler()
    profile = profiler.build_profile(
        history_ids=['N123', 'N456'],
        news_embeddings=encoder.final_embeddings,
        news_id_to_idx=encoder.news_id_to_idx,
    )

    # Retrieve candidates
    candidate_ids, scores = encoder.retriever.retrieve(
        query_vector=profile,
        k=100,
        exclude_ids=['N123', 'N456'],
    )
    """

    # Component dimensions (for documentation / assertions)
    SBERT_DIM = 384
    ENTITY_DIM = 100
    CAT_DIM = 32
    SUBCAT_DIM = 16
    FINAL_DIM = SBERT_DIM + ENTITY_DIM + CAT_DIM + SUBCAT_DIM  # = 532

    def __init__(self, sbert_model: str = SBERTTextEncoder.DEFAULT_MODEL, device: str = "cpu"):
        self.sbert_model = sbert_model
        self.device = device

        # Sub-components
        self.text_encoder = SBERTTextEncoder(sbert_model, device)
        self.entity_aggregator = EntityEmbeddingAggregator()
        self.category_layer = CategoryEmbeddingLayer()

        # Outputs populated by fit()
        self.sbert_embeddings: Optional[np.ndarray] = None
        self.entity_features: Optional[np.ndarray] = None
        self.category_features: Optional[np.ndarray] = None
        self.final_embeddings: Optional[np.ndarray] = None

        self.news_id_to_idx: Dict[str, int] = {}
        self.news_ids: List[str] = []

        self.retriever = FAISSCandidateRetriever()
        self._fitted = False

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(
        self,
        news_df: pd.DataFrame,
        entity_embeddings: Dict[str, np.ndarray],
        sbert_batch_size: int = 64,
        build_faiss: bool = True,
        sbert_checkpoint_path: Optional[str] = None,
        sbert_checkpoint_every: int = 2000,
        precomputed_sbert_path: Optional[str] = None,
    ) -> "NewsEncoderPhase1":
        """
        Encode all articles and build the FAISS retrieval index.

        Args:
            news_df:                   Cleaned news DataFrame from MINDDataLoader.
                                       Required columns: news_id, title, abstract,
                                       category, subcategory, all_entities.
            entity_embeddings:         Dict from MINDDataLoader.load_entity_embeddings().
            sbert_batch_size:          SBERT batch size. Use 32-64 on CPU.
                                       Larger = fewer overhead calls = faster.
            build_faiss:               Whether to build the FAISS index.
            sbert_checkpoint_path:     Path to save/resume SBERT partial results.
                                       Set this! If encoding is interrupted it
                                       resumes from the last checkpoint.
            sbert_checkpoint_every:    Save every N articles (default 2000).
            precomputed_sbert_path:    Path to a pre-computed sbert_news_embeddings.npy
                                       file (e.g. generated on Google Colab GPU).
                                       If set and file exists, SBERT encoding is
                                       skipped entirely — Towers 2+3 run as normal.
                                       This is the recommended workflow for CPU-only
                                       machines: encode once on free Colab GPU,
                                       then run everything else locally.

        Returns:
            self (for chaining)
        """
        t_total = time.time()
        logger.info("=" * 60)
        logger.info("Phase 1 — NLP Encoding Pipeline")
        logger.info("=" * 60)
        logger.info(f"Articles to encode: {len(news_df):,}")

        # --- Build news ID index ---
        self.news_ids = list(news_df["news_id"])
        self.news_id_to_idx = {nid: i for i, nid in enumerate(self.news_ids)}

        # -------------------------------------------------------
        # Tower 1: SBERT
        # -------------------------------------------------------
        logger.info("\n[Tower 1/3] Sentence-BERT text encoding …")

        if precomputed_sbert_path and Path(precomputed_sbert_path).exists():
            # Fast path: load SBERT embeddings computed on Colab/GPU
            logger.info(f"  Loading pre-computed SBERT from {precomputed_sbert_path} …")
            self.sbert_embeddings = np.load(precomputed_sbert_path).astype(np.float32)

            # Validate alignment using news_id_order if provided
            order_path = Path(precomputed_sbert_path).parent / "news_id_order.json"
            if order_path.exists():
                with open(order_path) as f_order:
                    colab_order = json.load(f_order)
                if colab_order != self.news_ids:
                    logger.warning(
                        "  news_id order in pre-computed file differs from current news_df. "
                        "Re-indexing to align …"
                    )
                    colab_idx = {nid: i for i, nid in enumerate(colab_order)}
                    reordered = np.zeros_like(self.sbert_embeddings)
                    for new_i, nid in enumerate(self.news_ids):
                        old_i = colab_idx.get(nid)
                        if old_i is not None:
                            reordered[new_i] = self.sbert_embeddings[old_i]
                    self.sbert_embeddings = reordered
                    logger.info("  Re-indexing complete.")
                else:
                    logger.info("  news_id order matches — no re-indexing needed.")
            logger.info(f"  SBERT loaded: {self.sbert_embeddings.shape}")
        else:
            # Slow path: encode on this machine (CPU will take a long time)
            self.sbert_embeddings = self.text_encoder.encode(
                news_df,
                batch_size=sbert_batch_size,
                checkpoint_path=sbert_checkpoint_path,
                checkpoint_every=sbert_checkpoint_every,
            )

        assert self.sbert_embeddings.shape == (len(news_df), self.SBERT_DIM), \
            f"SBERT shape mismatch: {self.sbert_embeddings.shape}"

        # -------------------------------------------------------
        # Tower 2: Entity embeddings
        # -------------------------------------------------------
        logger.info("\n[Tower 2/3] Entity embedding aggregation …")
        self.entity_features = self.entity_aggregator.encode(news_df, entity_embeddings)
        assert self.entity_features.shape == (len(news_df), self.ENTITY_DIM), \
            f"Entity shape mismatch: {self.entity_features.shape}"

        # -------------------------------------------------------
        # Tower 3: Category embeddings
        # -------------------------------------------------------
        logger.info("\n[Tower 3/3] Category embedding layer …")
        self.category_layer.fit(news_df)
        self.category_features = self.category_layer.transform(news_df)
        assert self.category_features.shape == (len(news_df), self.CAT_DIM + self.SUBCAT_DIM), \
            f"Category shape mismatch: {self.category_features.shape}"

        # -------------------------------------------------------
        # Fuse towers → final embedding
        # -------------------------------------------------------
        logger.info("\n[Fusion] Concatenating towers and L2-normalising …")
        fused = np.hstack([
            self.sbert_embeddings,
            self.entity_features,
            self.category_features,
        ])
        assert fused.shape[1] == self.FINAL_DIM, \
            f"Expected {self.FINAL_DIM}-dim fused embedding, got {fused.shape[1]}"

        self.final_embeddings = normalize(fused, norm="l2").astype(np.float32)

        logger.info(f"Final embeddings shape: {self.final_embeddings.shape}")
        logger.info(f"  → SBERT:     {self.SBERT_DIM}-dim")
        logger.info(f"  → Entity:    {self.ENTITY_DIM}-dim")
        logger.info(f"  → Category:  {self.CAT_DIM + self.SUBCAT_DIM}-dim  "
                    f"(cat:{self.CAT_DIM} + subcat:{self.SUBCAT_DIM})")
        logger.info(f"  → Total:     {self.FINAL_DIM}-dim, L2-normalised")

        # Verify normalisation
        norms = np.linalg.norm(self.final_embeddings, axis=1)
        logger.info(f"  Norm check: min={norms.min():.4f}  max={norms.max():.4f}  "
                    f"(should both be ≈1.0)")

        # -------------------------------------------------------
        # FAISS index
        # -------------------------------------------------------
        if build_faiss:
            logger.info("\n[FAISS] Building retrieval index …")
            self.retriever.build(self.final_embeddings, self.news_ids)

        self._fitted = True
        logger.info(
            f"\n{'='*60}\n"
            f"Phase 1 encoding complete in {time.time()-t_total:.1f}s\n"
            f"{'='*60}"
        )
        return self

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, output_dir: str) -> Path:
        """
        Persist all encoding outputs to disk.

        File layout:
          <output_dir>/
            sbert_news_embeddings.npy       — Tower 1 only (384-dim)
            entity_news_features.npy        — Tower 2 only (100-dim)
            category_news_features.npy      — Tower 3 only (48-dim)
            final_news_embeddings.npy       — Fused 532-dim L2-normalised
            news_faiss.index                — FAISS index file
            news_faiss.json                 — FAISS metadata (news_ids, dim)
            category_layer.pkl              — CategoryEmbeddingLayer state
            news_id_to_idx.json             — news_id → row-index mapping
            embedding_metadata.json         — shapes, model, timestamp

        Args:
            output_dir: Directory to write files into (created if missing).

        Returns:
            Path to output_dir.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before save().")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving Phase 1 outputs to {out} …")

        # Arrays
        np.save(out / "sbert_news_embeddings.npy", self.sbert_embeddings)
        np.save(out / "entity_news_features.npy", self.entity_features)
        np.save(out / "category_news_features.npy", self.category_features)
        np.save(out / "final_news_embeddings.npy", self.final_embeddings)
        logger.info("  ✔ Embeddings saved")

        # FAISS index
        self.retriever.save(out / "news_faiss.index")
        logger.info("  ✔ FAISS index saved")

        # Category layer
        self.category_layer.save(out / "category_layer.pkl")
        logger.info("  ✔ Category layer saved")

        # news_id mapping
        with open(out / "news_id_to_idx.json", "w") as f:
            json.dump(self.news_id_to_idx, f)
        logger.info("  ✔ news_id_to_idx mapping saved")

        # Metadata
        metadata = {
            "sbert_model": self.sbert_model,
            "n_articles": len(self.news_ids),
            "sbert_dim": self.SBERT_DIM,
            "entity_dim": self.ENTITY_DIM,
            "cat_dim": self.CAT_DIM,
            "subcat_dim": self.SUBCAT_DIM,
            "final_dim": self.FINAL_DIM,
            "n_categories": len(self.category_layer.cat_to_idx),
            "n_subcategories": len(self.category_layer.subcat_to_idx),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out / "embedding_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("  ✔ Metadata saved")

        logger.info(f"\nAll Phase 1 outputs saved to: {out}/")
        return out

    @classmethod
    def load(cls, output_dir: str) -> "NewsEncoderPhase1":
        """
        Load previously saved encoder from disk.

        This is the fast path used at recommendation time — no re-encoding
        needed.  Loading all files for 101 K articles takes ~2-3 seconds.

        Args:
            output_dir: Directory written by save().

        Returns:
            Fully initialised NewsEncoderPhase1 with all embeddings loaded.
        """
        out = Path(output_dir)

        with open(out / "embedding_metadata.json") as f:
            meta = json.load(f)

        obj = cls(sbert_model=meta["sbert_model"])

        # Embeddings
        obj.sbert_embeddings = np.load(out / "sbert_news_embeddings.npy")
        obj.entity_features = np.load(out / "entity_news_features.npy")
        obj.category_features = np.load(out / "category_news_features.npy")
        obj.final_embeddings = np.load(out / "final_news_embeddings.npy")

        # Category layer
        obj.category_layer = CategoryEmbeddingLayer.load(out / "category_layer.pkl")

        # news_id mapping
        with open(out / "news_id_to_idx.json") as f:
            obj.news_id_to_idx = json.load(f)
        obj.news_ids = [None] * len(obj.news_id_to_idx)
        for nid, idx in obj.news_id_to_idx.items():
            obj.news_ids[idx] = nid

        # FAISS
        obj.retriever = FAISSCandidateRetriever.load(out / "news_faiss.index")

        obj._fitted = True

        logger.info(
            f"NewsEncoderPhase1 loaded from {out}  "
            f"({meta['n_articles']:,} articles, {meta['final_dim']}-dim)"
        )
        return obj

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def get_embedding(self, news_id: str) -> Optional[np.ndarray]:
        """Return the 532-dim final embedding for a single article."""
        idx = self.news_id_to_idx.get(news_id)
        if idx is None:
            return None
        return self.final_embeddings[idx]

    def get_embeddings_batch(self, news_ids: List[str]) -> np.ndarray:
        """
        Return a (K, 532) matrix for a list of news IDs.
        Unknown IDs produce zero rows (handled gracefully by cosine sim).
        """
        dim = self.final_embeddings.shape[1]
        result = np.zeros((len(news_ids), dim), dtype=np.float32)
        for i, nid in enumerate(news_ids):
            idx = self.news_id_to_idx.get(nid)
            if idx is not None:
                result[i] = self.final_embeddings[idx]
        return result
