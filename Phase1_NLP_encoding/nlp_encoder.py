"""
Phase 1: NLP Encoding Layer  (v2 – improved)
=============================================
Diversity-Aware News Recommender System – Capstone Project

Improvements over v1
--------------------
  [FIX 1] SBERT model upgraded to 'all-mpnet-base-v2' (768-dim, default) with
          support for 'allenai/news-roberta-base' (news-domain, best quality).
          Controlled via the SBERT_MODEL constant / constructor arg.

  [FIX 2] SBERT input changed from  "{title} {abstract}"
                                  to  "{title}. {title}. {abstract}"
          Repeating the title ensures that articles with no/short abstracts
          still receive a meaningful embedding, instead of being penalised
          compared to articles with rich abstracts.

  [FIX 3] Entity embeddings now use IDF-weighted aggregation instead of
          a plain average.  Rare, specific entities (athletes, politicians,
          company names) receive higher weight; generic entities (country
          names, years) are down-weighted.  This makes the entity tower
          carry genuine topical signal rather than background noise.

  [FIX 4] Per-tower L2 normalisation BEFORE weighted fusion.
          In v1, raw (unequal-magnitude) towers were concatenated and then
          L2-normalised – meaning SBERT (384-dim) dominated by sheer size.
          Now each tower is independently L2-normalised and then fused with
          explicit weights:
              SBERT   0.60   (semantic content)
              Entity  0.30   (topical specificity)
              Category 0.10  (structural signal)
          The resulting 916-dim vector is then L2-normalised for cosine search.

Three-tower architecture
------------------------
  Tower 1 – Sentence-BERT (768-dim, all-mpnet-base-v2)
      Captures semantic meaning from title + abstract.
      "Fed rate hike" and "central bank tightening" become close vectors.

  Tower 2 – Entity Embeddings (100-dim, IDF-weighted)
      Averaged WikiData knowledge graph embeddings per article.
      IDF weighting ensures rare/specific entities dominate.

  Tower 3 – Learned Category Embeddings (32+16-dim)
      Small learnable embedding table instead of one-hot encoding.
      Learns that sports/soccer and sports/basketball are closer
      to each other than sports/soccer and finance/markets.

Final embedding: 916-dim, L2-normalized.
"""

import json
import logging
import math
import pickle
import time
from collections import Counter
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
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for Phase 1.\n"
            "Install with:  pip install sentence-transformers"
        )


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# Tower 1 – Sentence-BERT Text Encoder  [FIX 1 + FIX 2]
# ---------------------------------------------------------------------------

class SBERTTextEncoder:
    """
    Encodes news titles and abstracts using Sentence-BERT.

    FIX 1 – Model choice
    --------------------
    Default: 'all-mpnet-base-v2' (768-dim, strong general quality)
    Better:  'allenai/news-roberta-base' (768-dim, news-domain fine-tuned,
             ~4× slower on GPU but significantly better retrieval quality)

    To use the news-domain model, change DEFAULT_MODEL below or pass
    model_name='allenai/news-roberta-base' to the constructor.  You will
    also need to update SBERT_DIM and FINAL_DIM in NewsEncoderPhase1.

    FIX 2 – Input text construction
    --------------------------------
    v1: "{title} {abstract}"
    v2: "{title}. {title}. {abstract}"

    Repeating the title twice before the abstract gives the title 3× the
    influence of the abstract in the SBERT token window.  This matters
    because:
      - Abstracts in MIND are often empty or very short (median ~20 words)
      - The title is always present and is the most information-dense field
      - Without this fix, articles with no abstract receive a "diluted"
        encoding compared to articles with long abstracts
    """

    # Change to 'allenai/news-roberta-base' for best quality (needs GPU run)
    DEFAULT_MODEL = "all-mpnet-base-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            SentenceTransformer = _check_sbert()
            logger.info(f"Loading SBERT model '{self.model_name}' …")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SBERT model loaded.")

    @staticmethod
    def _build_input_texts(news_df: pd.DataFrame) -> List[str]:
        """
        FIX 2: Build input strings with title-emphasis weighting.

        Formula: "{title}. {title}. {abstract}"

        Why repeat title twice?
        - Gives title ~3× weight of abstract in SBERT's token attention
        - Articles with empty abstracts → "{title}. {title}. " (still good)
        - Articles with rich abstracts → title anchors the semantic direction
          before the abstract provides supporting detail
        """
        titles = news_df["title"].fillna("").astype(str)
        abstracts = news_df["abstract"].fillna("").astype(str)
        return (titles + ". " + titles + ". " + abstracts).tolist()

    def encode(
        self,
        news_df: pd.DataFrame,
        batch_size: int = 64,
        show_progress: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 2000,
    ) -> np.ndarray:
        """
        Encode all articles into (N, SBERT_DIM) L2-normalised float32 vectors.

        Args:
            news_df:           DataFrame with 'title' and 'abstract' columns.
            batch_size:        Encoding batch size. Use 32-64 on CPU, 256 on GPU.
            show_progress:     Show tqdm progress bar.
            checkpoint_path:   Path to save/load partial results.
            checkpoint_every:  Save a checkpoint every N articles.

        Returns:
            Float32 array of shape (N, dim), L2-normalised per row.
        """
        self._load_model()

        # FIX 2: use title-weighted input construction
        texts = self._build_input_texts(news_df)
        N = len(texts)
        logger.info(f"Encoding {N:,} articles with SBERT (batch_size={batch_size}) …")
        logger.info(f"  Input format: 'title. title. abstract'  (title-emphasis weighting)")

        # -- Resume from checkpoint --
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
                            f"({saved_n/N*100:.1f}%)"
                        )
                elif saved_n >= N:
                    logger.info(f"  ✔ Checkpoint complete ({saved_n:,} articles). Loading …")
                    return np.load(str(ckpt_path))
            except Exception as e:
                logger.warning(f"  Could not load checkpoint ({e}). Starting fresh.")

        # -- Encode in chunks --
        t0 = time.time()
        remaining_texts = texts[start_idx:]
        n_remaining = len(remaining_texts)
        chunks = [
            remaining_texts[i: i + checkpoint_every]
            for i in range(0, n_remaining, checkpoint_every)
        ]

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
            speed = len(chunk_texts) / elapsed_chunk
            remaining_articles = N - encoded_so_far
            eta_min = (remaining_articles / speed / 60) if speed > 0 else 0

            logger.info(
                f"  Chunk {chunk_i+1}/{len(chunks)} done  |  "
                f"{encoded_so_far:,}/{N:,} ({encoded_so_far/N*100:.1f}%)  |  "
                f"Speed: {speed:.0f} art/s  |  ETA: {eta_min:.1f} min"
            )

            if ckpt_path is not None:
                partial_arr = np.vstack(all_embeddings)
                np.save(str(ckpt_path), partial_arr)
                with open(str(ckpt_meta_path), "w") as f:
                    json.dump({"n_encoded": encoded_so_far, "total": N}, f)
                logger.info(f"  💾 Checkpoint saved ({encoded_so_far:,} articles)")

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        elapsed = time.time() - t0
        logger.info(
            f"SBERT encoding done in {elapsed/60:.1f} min  "
            f"({N/elapsed:.0f} articles/s)  Shape: {embeddings.shape}"
        )

        if ckpt_path and ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)
            if ckpt_meta_path and ckpt_meta_path.exists():
                ckpt_meta_path.unlink(missing_ok=True)

        return embeddings


# ---------------------------------------------------------------------------
# Tower 2 – Entity Embedding Aggregator  [FIX 3: IDF weighting]
# ---------------------------------------------------------------------------

class EntityEmbeddingAggregator:
    """
    Aggregates WikiData entity embeddings per article using IDF weighting.

    FIX 3 – IDF-weighted aggregation
    ---------------------------------
    v1: simple average of all entity vectors per article.
    v2: IDF-weighted average.

    Intuition: "LeBron James" appears in very few articles → high IDF →
    high weight. "United States" appears in thousands → low IDF → low weight.

    The IDF weight for entity e is:
        IDF(e) = log((1 + N) / (1 + df(e))) + 1
    where N = total articles, df(e) = number of articles mentioning e.
    (Smooth variant that avoids zero weights for universal entities.)

    This makes the entity tower carry genuine topical signal (specific people,
    companies, places) rather than being dominated by background entities.

    Coverage is ~96.7% per the data quality report, so zero-vectors affect
    only ~3% of articles.
    """

    ENTITY_DIM = 100

    def __init__(self):
        self.entity_dim = self.ENTITY_DIM
        self.entity_idf: Dict[str, float] = {}   # computed during encode()

    def _compute_idf(self, news_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute IDF weights for all entities across the corpus.

        Returns:
            Dict mapping entity_id → IDF weight (float ≥ 1.0)
        """
        N = len(news_df)
        entity_doc_freq: Counter = Counter()

        for entities in news_df["all_entities"]:
            if entities:
                # Count each entity once per article (document frequency)
                for eid in set(entities):
                    entity_doc_freq[eid] += 1

        idf = {}
        for eid, df in entity_doc_freq.items():
            # Smooth IDF: log((1+N)/(1+df)) + 1
            idf[eid] = math.log((1 + N) / (1 + df)) + 1.0

        logger.info(
            f"  IDF computed for {len(idf):,} unique entities  "
            f"(N={N:,} articles)"
        )
        if idf:
            weights = list(idf.values())
            logger.info(
                f"  IDF range: [{min(weights):.2f}, {max(weights):.2f}]  "
                f"mean={sum(weights)/len(weights):.2f}"
            )
        return idf

    def encode(
        self,
        news_df: pd.DataFrame,
        entity_embeddings: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Build a (N, 100) matrix of IDF-weighted entity embeddings.

        Args:
            news_df:           DataFrame with 'all_entities' column.
            entity_embeddings: Dict mapping WikiData ID → 100-dim vector.

        Returns:
            Float32 array of shape (N, 100).
        """
        N = len(news_df)
        features = np.zeros((N, self.entity_dim), dtype=np.float32)

        # FIX 3: compute IDF weights before aggregating
        logger.info("  Computing entity IDF weights …")
        self.entity_idf = self._compute_idf(news_df)

        missing_count = 0
        hit_count = 0

        for i, entities in enumerate(news_df["all_entities"]):
            if not entities:
                missing_count += 1
                continue

            # Collect (vector, idf_weight) pairs for known entities
            weighted_vecs = []
            total_weight = 0.0

            for eid in entities:
                if eid in entity_embeddings:
                    idf_w = self.entity_idf.get(eid, 1.0)
                    weighted_vecs.append(entity_embeddings[eid] * idf_w)
                    total_weight += idf_w

            if weighted_vecs and total_weight > 0:
                # Weighted sum normalised by total weight → weighted average
                features[i] = (np.sum(weighted_vecs, axis=0) / total_weight).astype(np.float32)
                hit_count += 1
            else:
                missing_count += 1

        coverage = hit_count / N * 100
        logger.info(
            f"  Entity features built: {hit_count:,}/{N:,} articles "
            f"have entity embeddings ({coverage:.1f}% coverage)"
        )
        return features


# ---------------------------------------------------------------------------
# Tower 3 – Learned Category Embeddings  (unchanged from v1)
# ---------------------------------------------------------------------------

class CategoryEmbeddingLayer:
    """
    Replaces one-hot category encoding with small learned embedding tables.

    Initialised with PCA-compressed one-hot (deterministic, no training loop).
    Category: 32-dim,  Subcategory: 16-dim.
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

        self.cat_embeddings = self._pca_init_onehot(n_cats, self.cat_dim)
        self.subcat_embeddings = self._pca_init_onehot(n_subcats, self.subcat_dim)

        self._fitted = True
        return self

    @staticmethod
    def _pca_init_onehot(n: int, target_dim: int) -> np.ndarray:
        if n <= target_dim:
            emb = np.zeros((n, target_dim), dtype=np.float32)
            emb[:n, :n] = np.eye(n, dtype=np.float32)
        else:
            rng = np.random.RandomState(42)
            raw = rng.randn(n, target_dim).astype(np.float32)
            q, _ = np.linalg.qr(raw)
            emb = q[:n, :target_dim]
        return normalize(emb, norm="l2")

    def transform(self, news_df: pd.DataFrame) -> np.ndarray:
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
                features[i, :self.cat_dim] = self.cat_embeddings[cat_idx]
            if subcat_idx is not None:
                features[i, self.cat_dim:] = self.subcat_embeddings[subcat_idx]

        return features

    def get_category_vector(self, category: str) -> np.ndarray:
        idx = self.cat_to_idx.get(category)
        if idx is None:
            return np.zeros(self.cat_dim, dtype=np.float32)
        return self.cat_embeddings[idx]

    def get_subcategory_vector(self, subcategory: str) -> np.ndarray:
        idx = self.subcat_to_idx.get(subcategory)
        if idx is None:
            return np.zeros(self.subcat_dim, dtype=np.float32)
        return self.subcat_embeddings[idx]

    def save(self, path: Path):
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
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.__dict__.update(data)
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Attention-Weighted User Profiler  (unchanged from v1)
# ---------------------------------------------------------------------------

class AttentionUserProfiler:
    """
    Builds a user profile as an attention-weighted average of clicked article
    embeddings, with recency decay.

    Recency decay weight for article at position i (0=oldest, T-1=newest):
        decay_i = exp(lambda * (i - (T-1)))
    lambda=0.3 → moderate decay (default, recommended for news)
    """

    def __init__(self, decay_lambda: float = 0.3):
        self.decay_lambda = decay_lambda

    def build_profile(
        self,
        history_ids: List[str],
        news_embeddings: np.ndarray,
        news_id_to_idx: Dict[str, int],
        max_history: int = 50,
        candidate_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not history_ids:
            dim = news_embeddings.shape[1]
            return np.zeros(dim, dtype=np.float32)

        resolved = []
        for nid in history_ids[-max_history:]:
            idx = news_id_to_idx.get(nid)
            if idx is not None:
                resolved.append(idx)

        if not resolved:
            dim = news_embeddings.shape[1]
            return np.zeros(dim, dtype=np.float32)

        T = len(resolved)
        history_embs = news_embeddings[resolved]

        positions = np.arange(T, dtype=np.float32)
        recency_w = np.exp(self.decay_lambda * (positions - (T - 1)))

        if candidate_context is not None:
            ctx = candidate_context.reshape(1, -1).astype(np.float32)
            content_w = (history_embs @ ctx.T).flatten()
            content_w = np.clip(content_w, 0, None)
        else:
            naive_mean = history_embs.mean(axis=0, keepdims=True)
            content_w = (history_embs @ naive_mean.T).flatten()
            content_w = np.clip(content_w, 0, None)

        combined_w = recency_w * (content_w + 1e-6)
        combined_w /= combined_w.sum()

        profile = (history_embs * combined_w[:, np.newaxis]).sum(axis=0)

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
# FAISS Retrieval Index  (unchanged from v1)
# ---------------------------------------------------------------------------

class FAISSCandidateRetriever:
    """
    Wraps a FAISS flat inner-product index for fast candidate retrieval.
    Inner product = cosine similarity when vectors are L2-normalised.
    """

    def __init__(self):
        self._index = None
        self._news_ids: List[str] = []
        self.embedding_dim: Optional[int] = None

    def build(self, embeddings: np.ndarray, news_ids: List[str]) -> "FAISSCandidateRetriever":
        faiss = _check_faiss()
        if faiss is None:
            logger.warning("FAISS unavailable – retrieval will use NumPy fallback.")
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
        exclude_set = set(exclude_ids) if exclude_ids else set()
        fetch_k = k + len(exclude_set) + 10

        q = query_vector.reshape(1, -1).astype(np.float32)

        if self._index is not None:
            scores_raw, indices = self._index.search(q, min(fetch_k, len(self._news_ids)))
            scores_raw = scores_raw[0]
            indices = indices[0]
        else:
            sims = (self._fallback_embeddings @ q.T).flatten()
            indices = np.argsort(sims)[::-1][:fetch_k]
            scores_raw = sims[indices]

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
        faiss = _check_faiss()
        meta = {"news_ids": self._news_ids, "embedding_dim": self.embedding_dim}
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
# Main Encoder – assembles all three towers  [FIX 4: per-tower normalisation]
# ---------------------------------------------------------------------------

# Tower fusion weights — must sum to 1.0.
# Tuning guide:
#   Increase SBERT_WEIGHT   → more semantic matching, better for long abstracts
#   Increase ENTITY_WEIGHT  → more entity/topic matching, good for news events
#   Increase CAT_WEIGHT     → stronger category signal, better for cold start
TOWER_WEIGHTS = {
    "sbert":    0.60,
    "entity":   0.30,
    "category": 0.10,
}


class NewsEncoderPhase1:
    """
    Three-tower news encoder for Phase 1 (v2 – improved fusion).

    FIX 4 – Per-tower normalisation before weighted fusion
    -------------------------------------------------------
    v1: raw towers concatenated → single L2-norm of the full 532-dim vector.
        Problem: SBERT (384-dim) dominated by sheer dimension count.
        Entity/category signal was diluted by the imbalanced dimension ratio.

    v2: each tower is independently L2-normalised FIRST, then scaled by
        an explicit weight (TOWER_WEIGHTS), then concatenated, then the
        final 916-dim vector is L2-normalised.

        Result: each tower contributes in proportion to its weight, not
        its raw dimension count.  The entity and category towers now have
        meaningful, controllable influence on retrieval and similarity.

    Fusion formula:
        fused = [0.60 × sbert_normed | 0.30 × entity_normed | 0.10 × cat_normed]
        final = L2_normalize(fused)

    Component dimensions:
        Tower 1 SBERT:     768-dim  (all-mpnet-base-v2)
        Tower 2 Entity:    100-dim
        Tower 3 Category:   48-dim  (cat:32 + subcat:16)
        Total final:       916-dim, L2-normalised
    """

    SBERT_DIM = 768   # all-mpnet-base-v2
    ENTITY_DIM = 100  # fixed by MIND dataset (WikiData KG embeddings)
    CAT_DIM = 32
    SUBCAT_DIM = 16
    FINAL_DIM = SBERT_DIM + ENTITY_DIM + CAT_DIM + SUBCAT_DIM  # = 916

    def __init__(self, sbert_model: str = SBERTTextEncoder.DEFAULT_MODEL, device: str = "cpu"):
        self.sbert_model = sbert_model
        self.device = device

        self.text_encoder = SBERTTextEncoder(sbert_model, device)
        self.entity_aggregator = EntityEmbeddingAggregator()
        self.category_layer = CategoryEmbeddingLayer()

        self.sbert_embeddings: Optional[np.ndarray] = None
        self.entity_features: Optional[np.ndarray] = None
        self.category_features: Optional[np.ndarray] = None
        self.final_embeddings: Optional[np.ndarray] = None

        self.news_id_to_idx: Dict[str, int] = {}
        self.news_ids: List[str] = []

        self.retriever = FAISSCandidateRetriever()
        self._fitted = False

    @staticmethod
    def _fuse_towers(
        sbert: np.ndarray,
        entity: np.ndarray,
        category: np.ndarray,
        weights: Dict[str, float] = TOWER_WEIGHTS,
    ) -> np.ndarray:
        """
        FIX 4: Per-tower L2-normalisation then weighted concatenation.

        Steps:
          1. L2-normalise each tower independently (each row → unit vector)
          2. Scale each tower by its weight
          3. Concatenate → (N, 916)
          4. L2-normalise the final concatenated vector

        This ensures:
          - Each tower contributes proportionally to its weight, not its raw dim
          - The final vector is unit-norm for cosine similarity search (FAISS)
          - Tower magnitudes cannot dominate each other accidentally
        """
        # Step 1: per-tower L2 normalisation
        sbert_n   = normalize(sbert,    norm="l2")   # (N, 384)
        entity_n  = normalize(entity,   norm="l2")   # (N, 100)
        cat_n     = normalize(category, norm="l2")   # (N,  48)

        # Step 2 & 3: scale then concatenate
        fused = np.hstack([
            weights["sbert"]    * sbert_n,
            weights["entity"]   * entity_n,
            weights["category"] * cat_n,
        ])  # (N, 916)

        # Step 4: final L2 normalisation for cosine search
        return normalize(fused, norm="l2").astype(np.float32)

    def fit(
        self,
        news_df: pd.DataFrame,
        entity_embeddings: Dict[str, np.ndarray],
        sbert_batch_size: int = 64,
        build_faiss: bool = True,
        sbert_checkpoint_path: Optional[str] = None,
        sbert_checkpoint_every: int = 2000,
        precomputed_sbert_path: Optional[str] = None,
        tower_weights: Optional[Dict[str, float]] = None,
    ) -> "NewsEncoderPhase1":
        """
        Encode all articles and build the FAISS retrieval index.

        Args:
            news_df:                   Cleaned news DataFrame.
                                       Required columns: news_id, title, abstract,
                                       category, subcategory, all_entities.
            entity_embeddings:         Dict mapping WikiData ID → 100-dim vector.
            sbert_batch_size:          Encoding batch size (32-64 CPU, 256 GPU).
            build_faiss:               Whether to build the FAISS index.
            sbert_checkpoint_path:     Path to save/resume SBERT partial results.
            sbert_checkpoint_every:    Checkpoint every N articles.
            precomputed_sbert_path:    Path to pre-computed sbert_news_embeddings.npy
                                       (skip SBERT encoding if file exists).
            tower_weights:             Override TOWER_WEIGHTS dict if provided.

        Returns:
            self (for chaining)
        """
        t_total = time.time()
        weights = tower_weights or TOWER_WEIGHTS

        logger.info("=" * 60)
        logger.info("Phase 1 – NLP Encoding Pipeline  (v2 improved)")
        logger.info("=" * 60)
        logger.info(f"Articles to encode: {len(news_df):,}")
        logger.info(f"Tower weights: SBERT={weights['sbert']}  "
                    f"Entity={weights['entity']}  Category={weights['category']}")

        # -- Build news ID index --
        self.news_ids = list(news_df["news_id"])
        self.news_id_to_idx = {nid: i for i, nid in enumerate(self.news_ids)}

        # ---------------------------------------------------
        # Tower 1: SBERT
        # ---------------------------------------------------
        logger.info("\n[Tower 1/3] Sentence-BERT text encoding …")

        if precomputed_sbert_path and Path(precomputed_sbert_path).exists():
            logger.info(f"  Loading pre-computed SBERT from {precomputed_sbert_path} …")
            self.sbert_embeddings = np.load(precomputed_sbert_path).astype(np.float32)

            order_path = Path(precomputed_sbert_path).parent / "news_id_order.json"
            if order_path.exists():
                with open(order_path) as f_order:
                    colab_order = json.load(f_order)
                if colab_order != self.news_ids:
                    logger.warning("  news_id order differs – re-indexing …")
                    colab_idx = {nid: i for i, nid in enumerate(colab_order)}
                    reordered = np.zeros_like(self.sbert_embeddings)
                    for new_i, nid in enumerate(self.news_ids):
                        old_i = colab_idx.get(nid)
                        if old_i is not None:
                            reordered[new_i] = self.sbert_embeddings[old_i]
                    self.sbert_embeddings = reordered
                    logger.info("  Re-indexing complete.")
            logger.info(f"  SBERT loaded: {self.sbert_embeddings.shape}")
        else:
            self.sbert_embeddings = self.text_encoder.encode(
                news_df,
                batch_size=sbert_batch_size,
                checkpoint_path=sbert_checkpoint_path,
                checkpoint_every=sbert_checkpoint_every,
            )

        assert self.sbert_embeddings.shape == (len(news_df), self.SBERT_DIM), \
            f"SBERT shape mismatch: {self.sbert_embeddings.shape}"

        # ---------------------------------------------------
        # Tower 2: Entity embeddings (IDF-weighted)
        # ---------------------------------------------------
        logger.info("\n[Tower 2/3] Entity embedding aggregation (IDF-weighted) …")
        self.entity_features = self.entity_aggregator.encode(news_df, entity_embeddings)
        assert self.entity_features.shape == (len(news_df), self.ENTITY_DIM), \
            f"Entity shape mismatch: {self.entity_features.shape}"

        # ---------------------------------------------------
        # Tower 3: Category embeddings
        # ---------------------------------------------------
        logger.info("\n[Tower 3/3] Category embedding layer …")
        self.category_layer.fit(news_df)
        self.category_features = self.category_layer.transform(news_df)
        assert self.category_features.shape == (len(news_df), self.CAT_DIM + self.SUBCAT_DIM), \
            f"Category shape mismatch: {self.category_features.shape}"

        # ---------------------------------------------------
        # FIX 4: Per-tower normalisation + weighted fusion
        # ---------------------------------------------------
        logger.info("\n[Fusion] Per-tower L2-normalisation → weighted concat → L2-norm …")
        logger.info(f"  Weights: SBERT={weights['sbert']}  "
                    f"Entity={weights['entity']}  Category={weights['category']}")

        self.final_embeddings = self._fuse_towers(
            self.sbert_embeddings,
            self.entity_features,
            self.category_features,
            weights=weights,
        )

        assert self.final_embeddings.shape == (len(news_df), self.FINAL_DIM), \
            f"Expected {self.FINAL_DIM}-dim fused embedding, got {self.final_embeddings.shape[1]}"

        logger.info(f"Final embeddings shape: {self.final_embeddings.shape}")
        logger.info(f"  → SBERT:     {self.SBERT_DIM}-dim  (weight {weights['sbert']})")
        logger.info(f"  → Entity:    {self.ENTITY_DIM}-dim  (weight {weights['entity']})")
        logger.info(f"  → Category:  {self.CAT_DIM + self.SUBCAT_DIM}-dim  "
                    f"(weight {weights['category']})")
        logger.info(f"  → Total:     {self.FINAL_DIM}-dim, L2-normalised")

        norms = np.linalg.norm(self.final_embeddings, axis=1)
        logger.info(f"  Norm check: min={norms.min():.4f}  max={norms.max():.4f}  "
                    f"(should both be ≈1.0)")

        # ---------------------------------------------------
        # FAISS index (rebuilt on new embeddings)
        # ---------------------------------------------------
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
        if not self._fitted:
            raise RuntimeError("Call fit() before save().")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nSaving Phase 1 outputs to {out} …")

        np.save(out / "sbert_news_embeddings.npy", self.sbert_embeddings)
        np.save(out / "entity_news_features.npy", self.entity_features)
        np.save(out / "category_news_features.npy", self.category_features)
        np.save(out / "final_news_embeddings.npy", self.final_embeddings)
        logger.info("  ✔ Embeddings saved")

        self.retriever.save(out / "news_faiss.index")
        logger.info("  ✔ FAISS index saved")

        self.category_layer.save(out / "category_layer.pkl")
        logger.info("  ✔ Category layer saved")

        with open(out / "news_id_to_idx.json", "w") as f:
            json.dump(self.news_id_to_idx, f)
        logger.info("  ✔ news_id_to_idx mapping saved")

        # Also save news_id_order.json for alignment verification
        with open(out / "news_id_order.json", "w") as f:
            json.dump(self.news_ids, f)

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
            "tower_weights": TOWER_WEIGHTS,
            "encoder_version": "v2",
            "improvements": [
                "FIX1: model upgraded to all-mpnet-base-v2 (768-dim), news-roberta-base also supported",
                "FIX2: SBERT input = 'title. title. abstract' (title-emphasis weighting)",
                "FIX3: IDF-weighted entity aggregation (rare entities get higher weight)",
                "FIX4: per-tower L2-normalisation before weighted fusion",
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out / "embedding_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("  ✔ Metadata saved")

        logger.info(f"\nAll Phase 1 outputs saved to: {out}/")
        return out

    @classmethod
    def load(cls, output_dir: str) -> "NewsEncoderPhase1":
        out = Path(output_dir)

        with open(out / "embedding_metadata.json") as f:
            meta = json.load(f)

        obj = cls(sbert_model=meta["sbert_model"])

        obj.sbert_embeddings   = np.load(out / "sbert_news_embeddings.npy")
        obj.entity_features    = np.load(out / "entity_news_features.npy")
        obj.category_features  = np.load(out / "category_news_features.npy")
        obj.final_embeddings   = np.load(out / "final_news_embeddings.npy")

        obj.category_layer = CategoryEmbeddingLayer.load(out / "category_layer.pkl")

        with open(out / "news_id_to_idx.json") as f:
            obj.news_id_to_idx = json.load(f)
        obj.news_ids = [None] * len(obj.news_id_to_idx)
        for nid, idx in obj.news_id_to_idx.items():
            obj.news_ids[idx] = nid

        obj.retriever = FAISSCandidateRetriever.load(out / "news_faiss.index")
        obj._fitted = True

        version = meta.get("encoder_version", "v1")
        logger.info(
            f"NewsEncoderPhase1 ({version}) loaded from {out}  "
            f"({meta['n_articles']:,} articles, {meta['final_dim']}-dim)"
        )
        if version == "v1":
            logger.warning(
                "  ⚠ These embeddings were generated by v1 (unweighted fusion).\n"
                "  Re-run run_phase1_encoding.py to regenerate with v2 improvements."
            )
        return obj

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def get_embedding(self, news_id: str) -> Optional[np.ndarray]:
        idx = self.news_id_to_idx.get(news_id)
        if idx is None:
            return None
        return self.final_embeddings[idx]

    def get_embeddings_batch(self, news_ids: List[str]) -> np.ndarray:
        dim = self.final_embeddings.shape[1]
        result = np.zeros((len(news_ids), dim), dtype=np.float32)
        for i, nid in enumerate(news_ids):
            idx = self.news_id_to_idx.get(nid)
            if idx is not None:
                result[i] = self.final_embeddings[idx]
        return result

    def diagnose_tower_contributions(self) -> Dict:
        """
        Diagnostic utility: shows how much each tower contributes to the
        final embeddings after per-tower normalisation and weighting.

        Useful for validating FIX 4 and tuning TOWER_WEIGHTS.

        Returns:
            Dict with contribution stats per tower.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        w = TOWER_WEIGHTS
        sbert_n   = normalize(self.sbert_embeddings,  norm="l2") * w["sbert"]
        entity_n  = normalize(self.entity_features,   norm="l2") * w["entity"]
        cat_n     = normalize(self.category_features, norm="l2") * w["category"]

        sbert_norms  = np.linalg.norm(sbert_n,  axis=1).mean()
        entity_norms = np.linalg.norm(entity_n, axis=1).mean()
        cat_norms    = np.linalg.norm(cat_n,    axis=1).mean()
        total        = sbert_norms + entity_norms + cat_norms

        result = {
            "tower_weights_configured": w,
            "effective_contributions": {
                "sbert":    round(sbert_norms  / total * 100, 1),
                "entity":   round(entity_norms / total * 100, 1),
                "category": round(cat_norms    / total * 100, 1),
            },
            "note": (
                "Effective contributions should match configured weights. "
                "Large deviations indicate zero-vectors in entity/category towers."
            ),
        }

        logger.info("\n=== Tower Contribution Diagnostic ===")
        logger.info(f"  Configured weights: {w}")
        logger.info(f"  Effective contributions:")
        for k, v in result["effective_contributions"].items():
            logger.info(f"    {k:10s}: {v:.1f}%  (target: {w[k]*100:.0f}%)")

        return result
