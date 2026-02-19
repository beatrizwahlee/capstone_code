"""
refuse_embeddings.py
====================
Quick-apply FIX 4 (per-tower normalisation + weighted fusion) to your
existing saved embeddings WITHOUT re-running SBERT or entity encoding.

This script:
  1. Loads the three raw tower files (already saved by your v1 run):
       - sbert_news_embeddings.npy    (N × 768)
       - entity_news_features.npy    (N × 100)
       - category_news_features.npy  (N × 48)
  2. Applies per-tower L2-normalisation
  3. Fuses with configurable weights (default: 0.60 / 0.30 / 0.10)
  4. L2-normalises the final vector
  5. Overwrites final_news_embeddings.npy and rebuilds the FAISS index

Run time: < 30 seconds (pure NumPy + FAISS, no model loading).

Usage:
    python refuse_embeddings.py

    # Custom weights:
    python refuse_embeddings.py --sbert 0.7 --entity 0.2 --category 0.1

    # Different embeddings directory:
    python refuse_embeddings.py --embeddings_dir ./my_embeddings
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default path — adjust if your structure is different
DEFAULT_EMBEDDINGS_DIR = Path(__file__).resolve().parent / "embeddings"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-fuse tower embeddings with per-tower normalisation (FIX 4)"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help="Directory containing the tower .npy files",
    )
    parser.add_argument("--sbert",    type=float, default=0.60, help="SBERT tower weight")
    parser.add_argument("--entity",   type=float, default=0.30, help="Entity tower weight")
    parser.add_argument("--category", type=float, default=0.10, help="Category tower weight")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print diagnostics without saving",
    )
    return parser.parse_args()


def load_towers(embeddings_dir: Path):
    """Load the three raw tower arrays from disk."""
    logger.info(f"Loading tower arrays from {embeddings_dir} …")

    sbert    = np.load(embeddings_dir / "sbert_news_embeddings.npy")
    entity   = np.load(embeddings_dir / "entity_news_features.npy")
    category = np.load(embeddings_dir / "category_news_features.npy")

    logger.info(f"  SBERT shape:    {sbert.shape}")
    logger.info(f"  Entity shape:   {entity.shape}")
    logger.info(f"  Category shape: {category.shape}")

    assert sbert.shape[0] == entity.shape[0] == category.shape[0], \
        "Row count mismatch between tower files!"

    return sbert, entity, category


def diagnose_before(sbert, entity, category):
    """Show how each tower contributes BEFORE the fix (v1 behaviour)."""
    logger.info("\n=== V1 Fusion Diagnosis (BEFORE fix) ===")
    logger.info("In v1, towers were concatenated raw then L2-normalised.")

    fused_v1 = np.hstack([sbert, entity, category])
    fused_v1_normed = normalize(fused_v1, norm="l2")

    N, D = fused_v1_normed.shape
    sbert_dim = sbert.shape[1]
    entity_dim = entity.shape[1]
    cat_dim = category.shape[1]

    # Measure each tower's actual contribution to the final vector
    sbert_contrib  = np.linalg.norm(fused_v1_normed[:, :sbert_dim],               axis=1).mean()
    entity_contrib = np.linalg.norm(fused_v1_normed[:, sbert_dim:sbert_dim+entity_dim], axis=1).mean()
    cat_contrib    = np.linalg.norm(fused_v1_normed[:, sbert_dim+entity_dim:],    axis=1).mean()
    total = sbert_contrib + entity_contrib + cat_contrib

    logger.info(f"  Effective contribution of each tower to final vector (should reflect DIM ratio):")
    logger.info(f"    SBERT    ({sbert_dim:3d}-dim): {sbert_contrib/total*100:.1f}%  "
                f"← dominated by dimension count ({sbert_dim}/{D}={sbert_dim/D*100:.0f}%)")
    logger.info(f"    Entity   ({entity_dim:3d}-dim): {entity_contrib/total*100:.1f}%")
    logger.info(f"    Category  ({cat_dim:2d}-dim): {cat_contrib/total*100:.1f}%")
    logger.info("  → Entity and Category were underrepresented relative to their value.")


def fuse_towers(sbert, entity, category, weights):
    """Apply FIX 4: per-tower L2-norm + weighted concat + final L2-norm."""
    logger.info(f"\n=== Applying FIX 4: per-tower normalisation + weighted fusion ===")
    logger.info(f"  Weights: SBERT={weights['sbert']}  "
                f"Entity={weights['entity']}  Category={weights['category']}")

    t0 = time.time()

    # Step 1: per-tower L2 normalisation
    sbert_n   = normalize(sbert,    norm="l2")
    entity_n  = normalize(entity,   norm="l2")
    cat_n     = normalize(category, norm="l2")

    # Step 2 & 3: weighted concatenation
    fused = np.hstack([
        weights["sbert"]    * sbert_n,
        weights["entity"]   * entity_n,
        weights["category"] * cat_n,
    ])

    # Step 4: final L2 normalisation
    final = normalize(fused, norm="l2").astype(np.float32)

    logger.info(f"  Fusion done in {time.time()-t0:.2f}s")
    logger.info(f"  Final shape: {final.shape}")

    # Verify norms
    norms = np.linalg.norm(final, axis=1)
    logger.info(f"  Norm check: min={norms.min():.6f}  max={norms.max():.6f}  (should both be ≈1.0)")

    return final


def diagnose_after(final, sbert, entity, category, weights):
    """Show each tower's actual contribution AFTER the fix."""
    logger.info("\n=== V2 Fusion Diagnosis (AFTER fix) ===")

    sbert_dim  = sbert.shape[1]
    entity_dim = entity.shape[1]
    cat_dim    = category.shape[1]

    sbert_contrib  = np.linalg.norm(final[:, :sbert_dim],                          axis=1).mean()
    entity_contrib = np.linalg.norm(final[:, sbert_dim:sbert_dim+entity_dim],      axis=1).mean()
    cat_contrib    = np.linalg.norm(final[:, sbert_dim+entity_dim:],               axis=1).mean()
    total = sbert_contrib + entity_contrib + cat_contrib

    logger.info("  Effective contribution of each tower (should now match weights):")
    logger.info(f"    SBERT:    {sbert_contrib/total*100:.1f}%  (target {weights['sbert']*100:.0f}%)")
    logger.info(f"    Entity:   {entity_contrib/total*100:.1f}%  (target {weights['entity']*100:.0f}%)")
    logger.info(f"    Category: {cat_contrib/total*100:.1f}%  (target {weights['category']*100:.0f}%)")


def rebuild_faiss(final: np.ndarray, embeddings_dir: Path):
    """Rebuild the FAISS index on the new embeddings."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed – skipping FAISS rebuild.")
        logger.warning("Install with: pip install faiss-cpu")
        return

    # Load news_ids for the index
    news_id_order_path = embeddings_dir / "news_id_order.json"
    news_id_to_idx_path = embeddings_dir / "news_id_to_idx.json"

    if news_id_order_path.exists():
        with open(news_id_order_path) as f:
            news_ids = json.load(f)
    elif news_id_to_idx_path.exists():
        with open(news_id_to_idx_path) as f:
            mapping = json.load(f)
        news_ids = [None] * len(mapping)
        for nid, idx in mapping.items():
            news_ids[idx] = nid
    else:
        logger.warning("news_id_order.json not found – cannot rebuild FAISS.")
        return

    logger.info(f"\nRebuilding FAISS index ({len(news_ids):,} articles, {final.shape[1]}-dim) …")
    t0 = time.time()

    N, D = final.shape
    index = faiss.IndexFlatIP(D)
    index.add(final.astype(np.float32))

    # Save index
    index_path = embeddings_dir / "news_faiss.index"
    faiss.write_index(index, str(index_path))

    meta = {"news_ids": news_ids, "embedding_dim": D}
    with open(embeddings_dir / "news_faiss.json", "w") as f:
        json.dump(meta, f)

    logger.info(f"  FAISS index rebuilt in {time.time()-t0:.2f}s → {index_path}")


def update_metadata(embeddings_dir: Path, weights: dict):
    """Update embedding_metadata.json to reflect v2 fusion."""
    meta_path = embeddings_dir / "embedding_metadata.json"
    if not meta_path.exists():
        return

    with open(meta_path) as f:
        meta = json.load(f)

    meta["encoder_version"] = "v2"
    meta["tower_weights"] = weights
    meta["fusion_fix_applied"] = True
    meta["fusion_fix_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["improvements"] = meta.get("improvements", []) + [
        "FIX4: per-tower L2-normalisation before weighted fusion (applied by refuse_embeddings.py)"
    ]

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata updated → {meta_path}")


def main():
    args = parse_args()
    weights = {
        "sbert":    args.sbert,
        "entity":   args.entity,
        "category": args.category,
    }

    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total:.3f} (not 1.0). Consider normalising them.")

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("RE-FUSE EMBEDDINGS – FIX 4 (per-tower normalisation)")
    print("=" * 60)
    print(f"Directory: {embeddings_dir}")
    print(f"Weights:   SBERT={weights['sbert']}  Entity={weights['entity']}  Category={weights['category']}")
    if args.dry_run:
        print("DRY RUN: diagnostics only, no files will be written")
    print("=" * 60 + "\n")

    # Load towers
    sbert, entity, category = load_towers(embeddings_dir)

    # Show v1 diagnosis
    diagnose_before(sbert, entity, category)

    # Apply fix
    final = fuse_towers(sbert, entity, category, weights)

    # Show v2 diagnosis
    diagnose_after(final, sbert, entity, category, weights)

    if args.dry_run:
        print("\nDry run complete. No files were modified.")
        return

    # Save
    logger.info("\nSaving updated files …")

    np.save(embeddings_dir / "final_news_embeddings.npy", final)
    logger.info(f"  ✔ final_news_embeddings.npy  ({final.nbytes/1e6:.0f} MB)")

    rebuild_faiss(final, embeddings_dir)
    update_metadata(embeddings_dir, weights)

    print("\n" + "=" * 60)
    print("✔ FIX 4 applied successfully.")
    print(f"  final_news_embeddings.npy updated with per-tower normalisation")
    print(f"  FAISS index rebuilt on new embeddings")
    print(f"  embedding_metadata.json updated to version v2")
    print("\nNext: re-run train_phase2_baseline.py to measure the improvement")
    print("=" * 60)


if __name__ == "__main__":
    main()
