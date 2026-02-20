"""
Phase 4: Echo Chamber Comparison Visualizations
================================================
Mirrors every Phase 3 plot but with Baseline vs Reranked side-by-side.

Six plots generated
-------------------
1. Gini Distribution Comparison      — histogram overlay per user
2. ILD Distribution Comparison       — histogram overlay per user
3. Multi-Method Bar Comparison       — all 7 algorithms across 5 metrics
4. Lorenz Curve Comparison           — category inequality before/after
5. Coverage vs Gini Scatter          — each algorithm as a labeled point
6. Summary Dashboard                 — all key numbers in one figure

Data sources
------------
- Per-user distributions  : computed live from sample_train_interactions.csv
                             (28 users, baseline + MMR + best calibrated)
- Aggregate multi-method   : Phase 4 diversity_evaluation.json
                             (full evaluation set, all 7 algorithms)

Usage:
    python run_phase4_visualization.py
"""

import ast
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Style (matches Phase 3) ───────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "font.size":     11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})

BASE     = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE / "Phase4_reranker" / "outputs" / "diversity_evaluation" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "baseline":     "#e74c3c",   # red
    "mmr":          "#2ecc71",   # green
    "xquad":        "#3498db",   # blue
    "calibrated":   "#9b59b6",   # purple
    "serendipity":  "#f39c12",   # orange
    "bounded_greedy": "#1abc9c", # teal
    "max_coverage": "#e67e22",   # dark orange
    "composite":    "#34495e",   # dark grey
}
METHOD_LABELS = {
    "baseline":       "Baseline",
    "mmr":            "MMR",
    "xquad":          "xQuAD",
    "calibrated":     "Calibrated",
    "serendipity":    "Serendipity",
    "bounded_greedy": "Bounded Greedy",
    "max_coverage":   "Max Coverage",
    "composite":      "Composite",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load models and data
# ─────────────────────────────────────────────────────────────────────────────

def load_everything():
    for sub in ["Phase2_baseline_rec", "Phase3_echo_chambers",
                "Phase4_reranker", "Phase1_NLP_encoding"]:
        p = BASE / sub
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from baseline_recommender_phase2 import BaselineRecommender
    from diversity_reranker import DiversityReranker
    from echo_chamber_analyzer import EchoChamberAnalyzer

    logger.info("Loading baseline model …")
    baseline = BaselineRecommender.load(
        str(BASE / "Phase2_baseline_rec/outputs/baseline/baseline_recommender.pkl"),
        str(BASE / "Phase1_NLP_encoding/embeddings"),
    )
    news_df = pd.read_csv(BASE / "Phase0_data_processing/processed_data/news_features_train.csv")
    news_categories = dict(zip(news_df["news_id"], news_df["category"]))

    reranker = DiversityReranker(
        baseline_recommender=baseline,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
        news_categories=news_categories,
        popularity_scores=baseline.popularity_scores,
    )

    analyzer = EchoChamberAnalyzer(
        recommender=baseline,
        news_df=news_df,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
    )

    # Sample users
    df = pd.read_csv(BASE / "Phase0_data_processing/processed_data/sample_train_interactions.csv")
    users = []
    for (uid, _), grp in df.groupby(["user_id", "impression_id"]):
        raw = grp["history"].iloc[0]
        history = ast.literal_eval(raw) if isinstance(raw, str) else []
        if history:
            users.append({"user_id": uid, "history": history})

    # Phase 4 aggregate data
    p4_path = BASE / "Phase4_reranker/outputs/diversity_evaluation/diversity_evaluation.json"
    with open(p4_path) as f:
        p4_results = json.load(f)

    logger.info(f"Loaded {len(users)} sample users · {len(p4_results)} algorithms from Phase 4")
    return baseline, reranker, analyzer, news_df, users, p4_results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute per-user diversity metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_user_metrics(users, baseline, reranker, analyzer, methods):
    """
    Returns dict: method_name → list of per-user metric dicts
    {gini, ild, coverage, calibration_error, entropy, num_unique_cats}
    """
    results = {m: [] for m in methods}

    for user in users:
        history = user["history"]

        # Baseline candidates (top-100)
        candidates = baseline.recommend(user_history=history, k=100, exclude_history=True)
        if not candidates:
            continue

        for method in methods:
            if method == "baseline":
                recs = candidates[:10]
            elif method == "mmr":
                recs = reranker.rerank(candidates, history, k=10,
                                       method="mmr", lambda_param=0.5)
            elif method == "calibrated":
                recs = reranker.rerank(candidates, history, k=10,
                                       method="calibrated", alpha=0.6)
            else:
                continue

            rec_ids = [nid for nid, _ in recs]
            rec_cats = [analyzer.news_to_category.get(nid, "")
                        for nid in rec_ids if analyzer.news_to_category.get(nid)]
            hist_cats = [analyzer.news_to_category.get(nid, "")
                         for nid in history if analyzer.news_to_category.get(nid)]

            if not rec_cats:
                continue

            results[method].append({
                "gini":              analyzer.calculate_gini(rec_cats),
                "ild":               analyzer.calculate_ild(rec_ids),
                "coverage":          analyzer.calculate_coverage(rec_cats),
                "calibration_error": analyzer.calculate_calibration_error(rec_cats, hist_cats),
                "entropy":           analyzer.calculate_entropy(rec_cats),
                "num_unique_cats":   len(set(rec_cats)),
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name):
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    logger.info(f"  Saved: {name}.png / .pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – Gini Distribution Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_gini_distribution(per_user):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    method_list = [("baseline", "Baseline"), ("mmr", "MMR (λ=0.5)"),
                   ("calibrated", "Calibrated (α=0.6)")]

    for ax, (method, label) in zip(axes, method_list):
        values = [m["gini"] for m in per_user[method]]
        color = COLORS[method]

        ax.hist(values, bins=15, color=color, edgecolor="black",
                alpha=0.75, label=label)
        mean_g = np.mean(values)
        ax.axvline(mean_g, color="black", linestyle="--", linewidth=2,
                   label=f"Mean {mean_g:.3f}")
        ax.axvline(0.8, color="red", linestyle=":", linewidth=1.5,
                   label="Filter-bubble threshold (0.8)")
        ax.set_xlabel("Gini Coefficient", fontsize=11)
        ax.set_ylabel("Users", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Gini Coefficient Distribution — Baseline vs Re-Ranked\n"
        "(Lower Gini = Less Concentrated = Fewer Echo Chambers)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "gini_distribution_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – ILD Distribution Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_ild_distribution(per_user):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    method_list = [("baseline", "Baseline"), ("mmr", "MMR (λ=0.5)"),
                   ("calibrated", "Calibrated (α=0.6)")]

    for ax, (method, label) in zip(axes, method_list):
        values = [m["ild"] for m in per_user[method]]
        color = COLORS[method]

        ax.hist(values, bins=15, color=color, edgecolor="black",
                alpha=0.75, label=label)
        mean_i = np.mean(values)
        ax.axvline(mean_i, color="black", linestyle="--", linewidth=2,
                   label=f"Mean {mean_i:.3f}")
        ax.axvline(0.4, color="blue", linestyle=":", linewidth=1.5,
                   label="Target ILD (0.4)")
        ax.set_xlabel("ILD (Intra-List Diversity)", fontsize=11)
        ax.set_ylabel("Users", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Semantic Diversity Distribution — Baseline vs Re-Ranked\n"
        "(Higher ILD = More Diverse Recommendations)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "ild_distribution_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – Multi-Method Bar Comparison  (Phase 4 aggregate data)
# ─────────────────────────────────────────────────────────────────────────────

def plot_multimethod_bars(p4_results):
    metrics = [
        ("avg_gini",              "Avg Gini ↓",             True),
        ("avg_ild",               "Avg ILD ↑",              False),
        ("avg_coverage",          "Avg Coverage ↑",         False),
        ("avg_entropy",           "Avg Entropy ↑",          False),
        ("avg_calibration_error", "Avg Calibration Error ↓", True),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 6))

    methods  = [r["method"] for r in p4_results]
    labels   = [METHOD_LABELS.get(m, m) for m in methods]
    colors   = [COLORS.get(m, "#95a5a6") for m in methods]

    for ax, (metric_key, metric_label, lower_better) in zip(axes, metrics):
        values = [r["diversity"][metric_key] for r in p4_results]
        bars   = ax.bar(labels, values, color=colors, edgecolor="black",
                        linewidth=1.2)

        # Highlight best bar
        best_idx = int(np.argmin(values) if lower_better else np.argmax(values))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(values) * 1.22)

    fig.suptitle(
        "All Re-Ranking Methods — Diversity Metrics Comparison\n"
        "(Gold outline = best result per metric)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, "multimethod_bar_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – Lorenz Curve Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_lorenz_curves(per_user, p4_results):
    """
    Build Lorenz curves from per-user category counts.
    Also annotates the Gini value from Phase 4 aggregate results for each method.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    def lorenz(values):
        arr = np.sort(np.array(values, dtype=float))
        arr = arr / arr.sum()
        cumulative = np.cumsum(arr)
        return np.concatenate([[0], cumulative])

    method_list = [("baseline", "Baseline"), ("mmr", "MMR (λ=0.5)"),
                   ("calibrated", "Calibrated (α=0.6)")]

    # Build aggregated category counts across all users per method
    for method, label in method_list:
        cat_counts = Counter()
        for m in per_user[method]:
            # Use num_unique_cats as proxy (approximate Lorenz)
            for i in range(int(m["num_unique_cats"])):
                cat_counts[f"c{i}"] += 1
        if not cat_counts:
            continue
        y = lorenz(list(cat_counts.values()))
        x = np.linspace(0, 1, len(y))
        gini_val = np.mean([m["gini"] for m in per_user[method]])
        ax.plot(x, y, color=COLORS[method], linewidth=2.5,
                label=f"{label}  (Gini={gini_val:.3f})")
        ax.fill_between(x, y, x, alpha=0.08, color=COLORS[method])

    # Perfect equality
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
            label="Perfect Equality", alpha=0.6)

    ax.set_xlabel("Cumulative % of Categories", fontsize=12)
    ax.set_ylabel("Cumulative % of Recommendations", fontsize=12)
    ax.set_title(
        "Lorenz Curve — Category Distribution Inequality\n"
        "Baseline vs Re-Ranked  (larger gap = more echo chamber)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "lorenz_curve_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Coverage vs Gini Scatter  (Phase 4 aggregate, one point per method)
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage_gini_scatter(p4_results):
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in p4_results:
        m      = r["method"]
        gini   = r["diversity"]["avg_gini"]
        cov    = r["diversity"]["avg_coverage"]
        color  = COLORS.get(m, "#95a5a6")
        label  = METHOD_LABELS.get(m, m)
        size   = 280 if m == "baseline" else 180

        ax.scatter(gini, cov, color=color, s=size,
                   edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(label, (gini, cov),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=10, fontweight="bold", color=color)

    # Quadrant guides
    ax.axhline(0.3, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.4, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.72, 0.05, "Echo Chamber\n(high Gini, low coverage)",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.15))
    ax.text(0.12, 0.60, "Diverse\n(low Gini, high coverage)",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="green", alpha=0.15))

    ax.set_xlabel("Avg Gini Coefficient  (lower = more diverse)", fontsize=12)
    ax.set_ylabel("Avg Coverage  (fraction of categories shown)", fontsize=12)
    ax.set_title(
        "Coverage vs Concentration — All Algorithms\n"
        "(Each point = one re-ranking method)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "coverage_vs_gini_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 – Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(per_user, p4_results):
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])   # Gini hist: baseline
    ax2 = fig.add_subplot(gs[0, 1])   # Gini hist: MMR
    ax3 = fig.add_subplot(gs[0, 2])   # ILD hist overlay
    ax4 = fig.add_subplot(gs[1, :2])  # Multi-method Gini bar
    ax5 = fig.add_subplot(gs[1, 2])   # Coverage vs Gini scatter (compact)
    ax6 = fig.add_subplot(gs[2, :])   # Summary table

    # ── Row 0: Gini histograms ────────────────────────────────────────────────
    for ax, method, title in [(ax1, "baseline", "Baseline — Gini"),
                               (ax2, "mmr", "MMR — Gini")]:
        vals = [m["gini"] for m in per_user[method]]
        ax.hist(vals, bins=12, color=COLORS[method],
                edgecolor="black", alpha=0.75)
        ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean {np.mean(vals):.3f}")
        ax.set_xlabel("Gini", fontsize=10)
        ax.set_ylabel("Users", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── ILD overlay ──────────────────────────────────────────────────────────
    for method in ["baseline", "mmr", "calibrated"]:
        vals = [m["ild"] for m in per_user[method]]
        ax3.hist(vals, bins=12, color=COLORS[method], edgecolor="black",
                 alpha=0.5, label=f"{METHOD_LABELS[method]} ({np.mean(vals):.3f})")
    ax3.set_xlabel("ILD", fontsize=10)
    ax3.set_ylabel("Users", fontsize=10)
    ax3.set_title("ILD Overlay", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Row 1: Multi-method Gini bar ─────────────────────────────────────────
    methods_ordered = [r["method"] for r in p4_results]
    gini_vals = [r["diversity"]["avg_gini"] for r in p4_results]
    bars = ax4.bar(
        [METHOD_LABELS.get(m, m) for m in methods_ordered],
        gini_vals,
        color=[COLORS.get(m, "#95a5a6") for m in methods_ordered],
        edgecolor="black", linewidth=1.2,
    )
    best_idx = int(np.argmin(gini_vals))
    bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(3)
    for bar, val in zip(bars, gini_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax4.set_ylabel("Avg Gini ↓", fontsize=10)
    ax4.set_title("Avg Gini by Algorithm  (gold = best)", fontsize=11,
                  fontweight="bold")
    ax4.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_ordered],
                        rotation=25, ha="right", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ── Coverage vs Gini (compact) ────────────────────────────────────────────
    for r in p4_results:
        m = r["method"]
        ax5.scatter(r["diversity"]["avg_gini"], r["diversity"]["avg_coverage"],
                    color=COLORS.get(m, "#95a5a6"), s=120,
                    edgecolors="black", linewidth=1.2)
        ax5.annotate(METHOD_LABELS.get(m, m),
                     (r["diversity"]["avg_gini"], r["diversity"]["avg_coverage"]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax5.set_xlabel("Avg Gini", fontsize=10)
    ax5.set_ylabel("Avg Coverage", fontsize=10)
    ax5.set_title("Coverage vs Gini", fontsize=11, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # ── Summary table ─────────────────────────────────────────────────────────
    ax6.axis("off")
    col_labels = ["Method", "Avg Gini ↓", "Avg ILD ↑", "Coverage ↑",
                  "Entropy ↑", "Cal.Error ↓"]
    table_rows = []
    for r in p4_results:
        d = r["diversity"]
        table_rows.append([
            METHOD_LABELS.get(r["method"], r["method"]),
            f"{d['avg_gini']:.4f}",
            f"{d['avg_ild']:.4f}",
            f"{d['avg_coverage']:.4f}",
            f"{d['avg_entropy']:.4f}",
            f"{d['avg_calibration_error']:.4f}",
        ])

    tbl = ax6.table(cellText=table_rows, colLabels=col_labels,
                    cellLoc="center", loc="center",
                    colWidths=[0.18, 0.14, 0.14, 0.14, 0.14, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(weight="bold", color="white")
    # Highlight baseline row (row 1)
    for j in range(len(col_labels)):
        tbl[(1, j)].set_facecolor("#fadbd8")   # light red = baseline
    # Alternate other rows
    for i in range(2, len(table_rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                tbl[(i, j)].set_facecolor("#eaf4f4")

    fig.suptitle(
        "Phase 4 — Echo Chamber Reduction Summary Dashboard\n"
        "Baseline (red background) vs All Re-Ranking Algorithms",
        fontsize=15, fontweight="bold", y=1.01,
    )
    _save(fig, "summary_dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  PHASE 4 — ECHO CHAMBER COMPARISON VISUALIZATIONS")
    print("=" * 62)

    baseline, reranker, analyzer, news_df, users, p4_results = load_everything()

    # Compute per-user metrics for baseline / MMR / Calibrated
    logger.info("Computing per-user diversity metrics …")
    per_user = compute_per_user_metrics(
        users, baseline, reranker, analyzer,
        methods=["baseline", "mmr", "calibrated"],
    )
    for m, records in per_user.items():
        if records:
            gini_vals = [r["gini"] for r in records]
            ild_vals  = [r["ild"]  for r in records]
            logger.info(
                f"  {m:12s} | users={len(records):2d} | "
                f"Gini={np.mean(gini_vals):.3f} | ILD={np.mean(ild_vals):.3f}"
            )

    # ── Generate all 6 plots ─────────────────────────────────────────────────
    logger.info(f"\nSaving visualizations to {OUT_DIR}/")

    logger.info("  1/6  Gini Distribution Comparison …")
    plot_gini_distribution(per_user)

    logger.info("  2/6  ILD Distribution Comparison …")
    plot_ild_distribution(per_user)

    logger.info("  3/6  Multi-Method Bar Comparison …")
    plot_multimethod_bars(p4_results)

    logger.info("  4/6  Lorenz Curve Comparison …")
    plot_lorenz_curves(per_user, p4_results)

    logger.info("  5/6  Coverage vs Gini Scatter …")
    plot_coverage_gini_scatter(p4_results)

    logger.info("  6/6  Summary Dashboard …")
    plot_summary_dashboard(per_user, p4_results)

    print("\n" + "=" * 62)
    print("  ✔  All 6 visualizations saved")
    print(f"     {OUT_DIR}/")
    print("     gini_distribution_comparison.png / .pdf")
    print("     ild_distribution_comparison.png  / .pdf")
    print("     multimethod_bar_comparison.png   / .pdf")
    print("     lorenz_curve_comparison.png      / .pdf")
    print("     coverage_vs_gini_scatter.png     / .pdf")
    print("     summary_dashboard.png            / .pdf")
    print("=" * 62)


if __name__ == "__main__":
    main()
