"""
Phase 3 Analysis Runner
=======================
Run comprehensive echo chamber analysis on the baseline recommender.

This script:
  1. Loads Phase 2 baseline recommender
  2. Loads validation data
  3. Generates recommendations for all test users
  4. Measures 5 echo chamber metrics
  5. Segments users by diversity level
  6. Saves comprehensive report

Usage:
    python run_phase3_analysis.py

Expected runtime: 15-45 min depending on dataset size.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add paths for imports
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

phase1_path = base_dir.parent / "Phase1_NLP_encoding"
if phase1_path.exists():
    sys.path.insert(0, str(phase1_path))

phase2_path = base_dir.parent / "Phase2_baseline_rec"
if phase2_path.exists():
    sys.path.insert(0, str(phase2_path))

phase0_path = base_dir.parent / "Phase0_data_processing" / "data_processing_v1"
if phase0_path.exists():
    sys.path.insert(0, str(phase0_path))

from echo_chamber_analyzer import EchoChamberAnalyzer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Phase 2 baseline directory
    "baseline_dir": str(base_dir.parent / "Phase2_baseline_rec" / "outputs" / "baseline"),
    
    # Phase 1 embeddings directory
    "embeddings_dir": str(base_dir.parent / "Phase1_NLP_encoding" / "embeddings"),
    
    # Validation data
    "valid_data_dir": str(base_dir.parent / "MINDlarge_dev"),
    
    # Output directory
    "output_dir": str(base_dir / "outputs" / "echo_chamber_analysis"),
    
    # Analysis settings
    "k": 10,  # Number of recommendations per user
    "max_users": None,  # Set to e.g. 10000 for faster testing, None for full
}


# ---------------------------------------------------------------------------
# Load validation data
# ---------------------------------------------------------------------------

def load_validation_data(valid_dir: Path, max_users: int = None) -> List[Dict]:
    """
    Load validation data for echo chamber analysis.
    
    Returns:
        List of dicts with keys: user_id, history
    """
    logger.info("Loading validation data ...")
    
    from mind_data_loader import MINDDataLoader
    
    loader = MINDDataLoader(str(valid_dir), dataset_type='valid')
    loader.load_all_data()
    
    test_data = []
    for _, row in loader.behaviors_df.iterrows():
        if row['history']:  # Only users with history
            test_data.append({
                'user_id': row['user_id'],
                'history': row['history'],
            })
        
        if max_users and len(test_data) >= max_users:
            break
    
    logger.info(f"  Loaded {len(test_data):,} users with click history")
    return test_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    
    print("\n" + "=" * 60)
    print("PHASE 3 — ECHO CHAMBER ANALYSIS")
    print("Diversity-Aware News Recommender — Capstone Project")
    print("=" * 60)
    
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k:20s}: {v}")
    
    # Validate paths
    baseline_dir = Path(CONFIG["baseline_dir"])
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Run Phase 2 first: python train_phase2_baseline.py")
        sys.exit(1)
    
    embeddings_dir = Path(CONFIG["embeddings_dir"])
    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        logger.error("Run Phase 1 first")
        sys.exit(1)
    
    valid_dir = Path(CONFIG["valid_data_dir"])
    if not valid_dir.exists():
        logger.error(f"Validation directory not found: {valid_dir}")
        sys.exit(1)
    
    # ---- Step 1: Load analyzer ----
    logger.info("\nLoading baseline recommender ...")
    analyzer = EchoChamberAnalyzer.from_baseline(
        str(baseline_dir),
        str(embeddings_dir),
    )
    
    # ---- Step 2: Load validation data ----
    test_data = load_validation_data(valid_dir, CONFIG["max_users"])
    
    # ---- Step 3: Run analysis ----
    report = analyzer.analyze(
        test_data,
        k=CONFIG["k"],
        max_users=CONFIG["max_users"],
    )
    
    # ---- Step 4: Print and save ----
    analyzer.print_report(report)
    analyzer.save_report(CONFIG["output_dir"], report)
    
    # ---- Step 5: Generate visualizations ----
    logger.info("\nGenerating visualizations ...")
    try:
        from echo_chamber_visualizer import EchoChamberVisualizer
        
        viz = EchoChamberVisualizer(report, report['raw_metrics'])
        viz_dir = Path(CONFIG["output_dir"]) / "visualizations"
        viz.create_all_plots(str(viz_dir))
        
        logger.info(f"✔ Visualizations saved to {viz_dir}/")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")
        logger.warning("Install matplotlib and seaborn: pip install matplotlib seaborn")
    
    # ---- Summary ----
    total_time = time.time() - t_start
    
    print(f"\n✔ Phase 3 complete in {total_time/60:.1f} min")
    print(f"\nOutputs saved to: {CONFIG['output_dir']}/")
    print(f"  - echo_chamber_report.json        (summary statistics)")
    print(f"  - user_metrics.json               (per-user raw data)")
    print(f"  - visualizations/                 (6 plots in PNG + PDF)")
    print(f"     • gini_distribution.png        (concentration histogram)")
    print(f"     • ild_distribution.png         (diversity histogram)")
    print(f"     • segment_comparison.png       (filter bubble vs balanced vs diverse)")
    print(f"     • gini_vs_coverage.png         (scatter plot)")
    print(f"     • lorenz_curve.png             (inequality visualization)")
    print(f"     • summary_dashboard.png        (all metrics in one page)")
    
    print(f"\nKey Findings:")
    print(f"  Average Gini:     {report['overall']['avg_gini']:.4f}  {'(HIGH - echo chamber!)' if report['overall']['avg_gini'] > 0.5 else '(moderate)'}")
    print(f"  Average ILD:      {report['overall']['avg_ild']:.4f}  {'(LOW - not diverse!)' if report['overall']['avg_ild'] < 0.3 else '(moderate)'}")
    print(f"  Filter Bubble %:  {report['segments']['filter_bubble']['pct']:.1f}%")
    
    print(f"\nNext steps:")
    print(f"  Phase 4 — Diversity Re-ranking")
    print(f"    Implement MMR, xQuAD, Calibration, Fairness")
    print(f"    Target: Gini < 0.4, ILD > 0.4")
    print(f"  Use summary_dashboard.png in your thesis/presentation!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
