"""
Phase 1 Run Script — NLP Encoding Pipeline
===========================================
Diversity-Aware News Recommender System — Capstone Project

This script drives the full Phase 1 encoding pipeline:
  1. Load cleaned MIND data (from Phase 0 / processed_data_v2)
  2. Encode all news articles with the three-tower encoder
  3. Build the FAISS retrieval index
  4. Run a sanity-check suite to verify embeddings make sense
  5. Save everything to ./embeddings/

Run time estimates (MINDlarge, 101 K articles):
  CPU only  : ~30-40 min  (SBERT is the bottleneck)
  GPU (T4)  : ~3-5  min

Run this script ONCE.  All downstream phases load embeddings from disk.

Usage:
    python run_phase1_encoding.py

    # Override paths via env vars or edit CONFIG below
    EMBEDDINGS_DIR=./my_embeddings python run_phase1_encoding.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration — edit these to match your directory layout
# ---------------------------------------------------------------------------
CONFIG = {
    # Where Phase 0 wrote its outputs
    "processed_data_dir":  "./processed_data_v2",

    # Raw MIND directories (needed to reload full news_df + entity embeddings)
    "train_data_dir":  "./MINDlarge_train",   # or ./MINDsmall_train

    # Where Phase 1 will write its outputs
    "embeddings_dir":  str(base_dir / "embeddings"),

    # SBERT model to use.
    # "all-MiniLM-L6-v2"       — fast, good quality (recommended for CPU)
    # "allenai/news-roberta-base" — news-domain fine-tuned, ~4× slower, best quality
    "sbert_model":  "all-MiniLM-L6-v2",

    # "cpu" or "cuda" — SBERT will use GPU if available and set to "cuda"
    "device":  "cpu",

    # SBERT encoding batch size.
    # ─────────────────────────────────────────────────────────────────────
    # IMPORTANT: Do NOT use a very small batch size on CPU.
    # Each batch has fixed tokenization overhead regardless of size.
    # batch_size=16  →  6,346 batches  →  slow (too much overhead per batch)
    # batch_size=32  →  3,173 batches  →  good default for most laptops
    # batch_size=64  →  1,587 batches  →  ideal for 16 GB+ RAM
    "sbert_batch_size":  32,

    # Checkpoint: save partial SBERT results every N articles.
    # If the script is interrupted, re-running it resumes from here
    # instead of starting over from article 0.
    # Set sbert_checkpoint_path to None to disable (not recommended on CPU).
    "sbert_checkpoint_path":   None,
    "sbert_checkpoint_every":  2000,

    # ── Colab pre-computed SBERT path ──────────────────────────────────────
    # RECOMMENDED WORKFLOW FOR CPU-ONLY MACHINES:
    #   1. Open phase1_sbert_colab.ipynb in Google Colab (free T4 GPU)
    #   2. Upload news_features_train.csv, run all cells (~3 min)
    #   3. Download sbert_news_embeddings.npy + news_id_order.json
    #   4. Put both files in ./embeddings/
    #   5. Set this path — the script will skip SBERT and run Towers 2+3 only
    #
    # Set to None to encode locally (only viable with a GPU or overnight run).
    "precomputed_sbert_path":  None,

    # Sample mode: encode only the first N articles (for fast testing).
    # Set to 5000 to test the full pipeline in ~5 min before the full run.
    # Set to None to encode all 101K articles (the real run).
    "sample_n":  None,

    # Max history length per user for profile building
    "max_history":  50,

    # Recency decay lambda for AttentionUserProfiler
    # 0.0 = no decay (all history equally weighted)
    # 0.3 = moderate decay (default, recommended)
    # 1.0 = aggressive decay (only last few clicks matter)
    "decay_lambda":  0.3,
}

# Allow env-var overrides for easy scripting
CONFIG["processed_data_dir"] = os.environ.get("PROCESSED_DIR", CONFIG["processed_data_dir"])
CONFIG["train_data_dir"]      = os.environ.get("TRAIN_DIR",     CONFIG["train_data_dir"])
CONFIG["embeddings_dir"]      = os.environ.get("EMBEDDINGS_DIR", CONFIG["embeddings_dir"])
CONFIG["sbert_checkpoint_path"] = os.environ.get(
    "SBERT_CHECKPOINT_PATH",
    str(Path(CONFIG["embeddings_dir"]) / "sbert_checkpoint.npy")
)
CONFIG["precomputed_sbert_path"] = os.environ.get(
    "PRECOMPUTED_SBERT_PATH",
    str(Path(CONFIG["embeddings_dir"]) / "sbert_news_embeddings.npy")
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(base_dir))

phase0_path = base_dir.parent / "Phase0_data_processing" / "data_processing_v1"
if phase0_path.exists():
    sys.path.insert(0, str(phase0_path))

from nlp_encoder import (
    NewsEncoderPhase1,
    AttentionUserProfiler,
)


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------

def load_data():
    """
    Load news_df and entity_embeddings.

    Tries processed_data_v2 first (Phase 0 output).
    Falls back to loading directly from the raw MIND directory.
    """
    logger.info("=" * 60)
    logger.info("Step 1/5 — Loading data")
    logger.info("=" * 60)

    processed_dir = Path(CONFIG["processed_data_dir"])
    raw_dir        = Path(CONFIG["train_data_dir"])

    # ---- news_df ----
    news_features_path = processed_dir / "news_features_train.csv"

    if news_features_path.exists():
        logger.info(f"Loading news features from {news_features_path} …")
        news_df = pd.read_csv(news_features_path)

        # all_entities column is stored as a string repr of a list — parse it
        if "all_entities" in news_df.columns:
            import ast
            news_df["all_entities"] = news_df["all_entities"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
            )
        else:
            news_df["all_entities"] = [[] for _ in range(len(news_df))]

        logger.info(f"  Loaded {len(news_df):,} articles from processed CSV")
    else:
        logger.warning(
            f"  Processed CSV not found at {news_features_path}\n"
            f"  Falling back to raw MIND directory: {raw_dir}"
        )
        # Import the data loader from Phase 0
        from mind_data_loader import MINDDataLoader
        loader = MINDDataLoader(str(raw_dir), dataset_type="train")
        loader.load_all_data()
        news_df = loader.news_df
        logger.info(f"  Loaded {len(news_df):,} articles from raw MIND data")

    # Ensure required columns exist
    for col in ["news_id", "title", "abstract", "category", "subcategory", "all_entities"]:
        if col not in news_df.columns:
            raise ValueError(
                f"Column '{col}' missing from news_df.  "
                f"Re-run Phase 0 (example_usage_v2.py) to regenerate processed data."
            )

    # Fill missing text fields
    news_df["title"]    = news_df["title"].fillna("").astype(str)
    news_df["abstract"] = news_df["abstract"].fillna("").astype(str)

    logger.info(f"  news_df shape: {news_df.shape}")
    logger.info(f"  Categories:    {news_df['category'].nunique()}")
    logger.info(f"  Subcategories: {news_df['subcategory'].nunique()}")

    # ---- entity_embeddings ----
    entity_vec_path = raw_dir / "entity_embedding.vec"

    if entity_vec_path.exists():
        logger.info(f"Loading entity embeddings from {entity_vec_path} …")
        entity_embeddings: dict = {}
        with open(entity_vec_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    entity_embeddings[parts[0]] = np.array(
                        [float(x) for x in parts[1:]], dtype=np.float32
                    )
        logger.info(f"  Loaded {len(entity_embeddings):,} entity embeddings")
    else:
        logger.warning(
            f"  entity_embedding.vec not found at {entity_vec_path}. "
            f"Entity tower will produce zero vectors."
        )
        entity_embeddings = {}

    return news_df, entity_embeddings


# ---------------------------------------------------------------------------
# Step 2 — Run Phase 1 encoding
# ---------------------------------------------------------------------------

def run_encoding(news_df: pd.DataFrame, entity_embeddings: dict) -> NewsEncoderPhase1:
    logger.info("\n" + "=" * 60)
    logger.info("Step 2/5 — Three-tower encoding")
    logger.info("=" * 60)

    # Apply sample_n if set (for fast pipeline testing)
    sample_n = CONFIG.get("sample_n")
    if sample_n and sample_n > 0:
        logger.info(f"  SAMPLE MODE: encoding only first {sample_n:,} articles")
        logger.info(f"  (Set sample_n=None in CONFIG for the full dataset)")
        news_df = news_df.head(sample_n).reset_index(drop=True)

    encoder = NewsEncoderPhase1(
        sbert_model=CONFIG["sbert_model"],
        device=CONFIG["device"],
    )

    # Create embeddings dir early so checkpoint path is valid
    Path(CONFIG["embeddings_dir"]).mkdir(parents=True, exist_ok=True)

    encoder.fit(
        news_df=news_df,
        entity_embeddings=entity_embeddings,
        sbert_batch_size=CONFIG["sbert_batch_size"],
        build_faiss=True,
        sbert_checkpoint_path=CONFIG.get("sbert_checkpoint_path"),
        sbert_checkpoint_every=CONFIG.get("sbert_checkpoint_every", 2000),
        precomputed_sbert_path=CONFIG.get("precomputed_sbert_path"),
    )

    return encoder


# ---------------------------------------------------------------------------
# Step 3 — Save
# ---------------------------------------------------------------------------

def save_outputs(encoder: NewsEncoderPhase1) -> Path:
    logger.info("\n" + "=" * 60)
    logger.info("Step 3/5 — Saving outputs")
    logger.info("=" * 60)

    out_path = encoder.save(CONFIG["embeddings_dir"])
    return out_path


# ---------------------------------------------------------------------------
# Step 4 — Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(encoder: NewsEncoderPhase1, news_df: pd.DataFrame):
    """
    Run a suite of checks to verify the embeddings are correct.

    These are not unit tests (no assertions that raise errors); they print
    results so you can visually verify sensibility.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Step 4/5 — Sanity checks")
    logger.info("=" * 60)

    embeddings = encoder.final_embeddings
    N, D = embeddings.shape

    print(f"\n--- Basic Checks ---")
    print(f"  Final embedding shape:  {N:,} × {D}")
    print(f"  Expected shape:         {N:,} × 532")
    assert D == 532, f"❌  Dimension mismatch! Got {D}, expected 532"
    print(f"  ✔ Dimension correct (532)")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Norm range: [{norms.min():.6f}, {norms.max():.6f}]  (should be ≈ 1.0)")
    assert norms.min() > 0.99 and norms.max() < 1.01, \
        f"❌  Embeddings not L2-normalised!  min={norms.min()}, max={norms.max()}"
    print(f"  ✔ All embeddings L2-normalised")

    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    print(f"  NaN values:  {nan_count}   (should be 0)")
    print(f"  Inf values:  {inf_count}   (should be 0)")
    assert nan_count == 0 and inf_count == 0, "❌  NaN or Inf found in embeddings!"
    print(f"  ✔ No NaN/Inf values")

    # ---- Semantic coherence: same-category articles should be closer ----
    print(f"\n--- Semantic Coherence Check ---")
    print("  (Articles in the same category should be more similar than cross-category)")

    # Pick two different categories with at least 5 articles each
    cat_counts = news_df["category"].value_counts()
    cats_ok = cat_counts[cat_counts >= 5].index.tolist()

    if len(cats_ok) >= 2:
        cat_a, cat_b = cats_ok[0], cats_ok[1]

        ids_a = news_df[news_df["category"] == cat_a]["news_id"].head(5).tolist()
        ids_b = news_df[news_df["category"] == cat_b]["news_id"].head(5).tolist()

        emb_a = np.array([embeddings[encoder.news_id_to_idx[nid]] for nid in ids_a])
        emb_b = np.array([embeddings[encoder.news_id_to_idx[nid]] for nid in ids_b])

        # Within-category similarities
        within_a = []
        for i in range(len(emb_a)):
            for j in range(i + 1, len(emb_a)):
                within_a.append(float(emb_a[i] @ emb_a[j]))
        avg_within = np.mean(within_a)

        # Cross-category similarities
        cross = []
        for ea in emb_a:
            for eb in emb_b:
                cross.append(float(ea @ eb))
        avg_cross = np.mean(cross)

        print(f"  Category A: {cat_a!r}")
        print(f"  Category B: {cat_b!r}")
        print(f"  Avg within-{cat_a} similarity: {avg_within:.4f}")
        print(f"  Avg cross-category similarity:  {avg_cross:.4f}")

        if avg_within > avg_cross:
            print(f"  ✔ Within-category similarity > cross-category  (expected ✓)")
        else:
            print(f"  ⚠ Within-category NOT higher than cross-category.")
            print(f"    This can happen if the two chosen categories are topically")
            print(f"    similar.  Try checking a sports vs. finance pair manually.")

    # ---- FAISS sanity: nearest neighbours of an article make sense ----
    print(f"\n--- FAISS Retrieval Check ---")
    sample_id = encoder.news_ids[0]
    sample_emb = encoder.final_embeddings[0]

    cand_ids, cand_scores = encoder.retriever.retrieve(
        query_vector=sample_emb,
        k=5,
        exclude_ids=[sample_id],
    )

    sample_row = news_df[news_df["news_id"] == sample_id].iloc[0]
    print(f"\n  Query article: [{sample_id}]  {sample_row['title'][:70]}")
    print(f"  Category: {sample_row['category']} / {sample_row['subcategory']}")
    print(f"\n  Top-5 nearest neighbours:")

    for rank, (nid, score) in enumerate(zip(cand_ids, cand_scores), 1):
        rows = news_df[news_df["news_id"] == nid]
        if len(rows):
            row = rows.iloc[0]
            print(f"  {rank}. [{nid}]  score={score:.4f}  "
                  f"{row['category']}/{row['subcategory']}")
            print(f"     {row['title'][:70]}")
        else:
            print(f"  {rank}. [{nid}]  score={score:.4f}  (not in news_df)")

    # ---- User profile check ----
    print(f"\n--- User Profile Check ---")

    # Grab a real user history from Phase 0 processed data
    sample_history_ids = []
    interactions_path = Path(CONFIG["processed_data_dir"]) / "sample_train_interactions.csv"
    if interactions_path.exists():
        import ast
        interactions_df = pd.read_csv(interactions_path, nrows=100)
        if "history" in interactions_df.columns:
            for _, row in interactions_df.iterrows():
                h = row["history"]
                if isinstance(h, str):
                    try:
                        h = ast.literal_eval(h)
                    except Exception:
                        h = []
                if h and len(h) >= 3:
                    sample_history_ids = h[:20]
                    break

    if not sample_history_ids:
        # Fall back to first 10 articles as a mock history
        sample_history_ids = encoder.news_ids[:10]
        logger.info("  (Using mock history — no real history found in processed data)")

    profiler = AttentionUserProfiler(decay_lambda=CONFIG["decay_lambda"])
    profile = profiler.build_profile(
        history_ids=sample_history_ids,
        news_embeddings=encoder.final_embeddings,
        news_id_to_idx=encoder.news_id_to_idx,
        max_history=CONFIG["max_history"],
    )

    profile_norm = np.linalg.norm(profile)
    print(f"  History length (truncated):  {min(len(sample_history_ids), CONFIG['max_history'])}")
    print(f"  Profile vector shape:         {profile.shape}")
    print(f"  Profile L2 norm:              {profile_norm:.6f}  (should be ≈ 1.0)")

    # Retrieve top-5 for this user profile
    cands, scores = encoder.retriever.retrieve(
        query_vector=profile,
        k=5,
        exclude_ids=sample_history_ids,
    )

    print(f"\n  Top-5 recommendations for this user profile:")
    for rank, (nid, score) in enumerate(zip(cands, scores), 1):
        rows = news_df[news_df["news_id"] == nid]
        if len(rows):
            row = rows.iloc[0]
            print(f"  {rank}. [{nid}]  score={score:.4f}  "
                  f"{row['category']}/{row['subcategory']}")
            print(f"     {row['title'][:70]}")
        else:
            print(f"  {rank}. [{nid}]  score={score:.4f}")

    print(f"\n✔ All sanity checks passed")


# ---------------------------------------------------------------------------
# Step 5 — Verify load-from-disk works correctly
# ---------------------------------------------------------------------------

def verify_load(embeddings_dir: str):
    logger.info("\n" + "=" * 60)
    logger.info("Step 5/5 — Verifying load-from-disk")
    logger.info("=" * 60)

    encoder_reloaded = NewsEncoderPhase1.load(embeddings_dir)

    orig_path = Path(embeddings_dir) / "final_news_embeddings.npy"
    orig = np.load(str(orig_path))

    assert np.allclose(encoder_reloaded.final_embeddings, orig, atol=1e-6), \
        "❌  Reloaded embeddings differ from saved embeddings!"

    print(f"  ✔ Reloaded embeddings are byte-identical to saved embeddings")
    print(f"  ✔ FAISS index: {encoder_reloaded.retriever._index.ntotal:,} vectors"
          if encoder_reloaded.retriever._index is not None
          else "  ✔ FAISS index loaded (NumPy fallback)")
    print(f"  ✔ news_id_to_idx: {len(encoder_reloaded.news_id_to_idx):,} entries")
    print(f"  ✔ Category layer: "
          f"{len(encoder_reloaded.category_layer.cat_to_idx)} categories, "
          f"{len(encoder_reloaded.category_layer.subcat_to_idx)} subcategories")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("\n" + "=" * 60)
    print("PHASE 1 — NLP ENCODING PIPELINE")
    print("Diversity-Aware News Recommender — Capstone Project")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k:25s}: {v}")

    # ---- Step 1: Load ----
    news_df, entity_embeddings = load_data()

    # ---- Step 2: Encode ----
    encoder = run_encoding(news_df, entity_embeddings)

    # ---- Step 3: Save ----
    out_path = save_outputs(encoder)

    # ---- Step 4: Sanity checks ----
    run_sanity_checks(encoder, news_df)

    # ---- Step 5: Verify reload ----
    verify_load(CONFIG["embeddings_dir"])

    # ---- Summary ----
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  Total time:          {total_time/60:.1f} min")
    print(f"  Output directory:    {out_path}/")
    print(f"\n  Files written:")
    for p in sorted(out_path.iterdir()):
        size_mb = p.stat().st_size / 1_048_576
        print(f"    {p.name:45s}  {size_mb:6.1f} MB")

    print(f"""
Next steps:
-----------
  Phase 2 — Baseline Recommender
    Load encoder once:
        from nlp_encoder import NewsEncoderPhase1, AttentionUserProfiler
        encoder = NewsEncoderPhase1.load('./embeddings')

    Build user profile at recommendation time:
        profiler = AttentionUserProfiler(decay_lambda=0.3)
        profile  = profiler.build_profile(history_ids, encoder.final_embeddings,
                                          encoder.news_id_to_idx)

    Retrieve top-100 candidates:
        cand_ids, scores = encoder.retriever.retrieve(profile, k=100,
                                                       exclude_ids=history_ids)
""")


if __name__ == "__main__":
    main()
