"""
Phase 5: Interactive Web App — Streamlit
=========================================
Diversity-Aware News Recommender System — Capstone Project
"""

import ast
import hashlib
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Add project paths (script-relative, robust to cwd)
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

for sub in ["Phase2_baseline_rec", "Phase3_echo_chambers", "Phase4_reranker",
            "Phase1_NLP_encoding"]:
    p = base_dir.parent / sub
    if p.exists():
        sys.path.insert(0, str(p))

phase0_code_path = base_dir.parent / "Phase0_data_processing" / "data_processing"
if phase0_code_path.exists():
    sys.path.insert(0, str(phase0_code_path))

logger = logging.getLogger(__name__)

from baseline_recommender_phase2 import BaselineRecommender
from diversity_reranker import DiversityReranker
from personalized_tuner import PersonalizedWeightTuner, infer_time_of_day

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Diversity-Aware News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data / Model Loading (Cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="🔄 Loading recommender models… (first run ~30s)")
def load_system():
    embeddings_dir = str(base_dir.parent / "Phase1_NLP_encoding" / "embeddings")
    baseline_path  = str(base_dir.parent / "Phase2_baseline_rec" / "outputs"
                         / "baseline" / "baseline_recommender.pkl")
    news_path      = str(base_dir.parent / "Phase0_data_processing"
                         / "processed_data" / "news_features_train.csv")

    baseline = BaselineRecommender.load(baseline_path, embeddings_dir)
    news_df  = pd.read_csv(news_path)
    news_categories = dict(zip(news_df["news_id"], news_df["category"]))

    reranker = DiversityReranker(
        baseline_recommender=baseline,
        embeddings=baseline.final_embeddings,
        news_id_to_idx=baseline.news_id_to_idx,
        news_categories=news_categories,
        popularity_scores=baseline.popularity_scores,
    )

    tuner = PersonalizedWeightTuner()
    return baseline, reranker, news_df, tuner


@st.cache_data(show_spinner=False)
def load_user_histories() -> Dict[str, List[str]]:
    """Return {user_id: [news_id, ...]} from sample training interactions."""
    csv_path = (base_dir.parent / "Phase0_data_processing"
                / "processed_data" / "sample_train_interactions.csv")
    df = pd.read_csv(csv_path)

    user_histories: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        uid = str(row["user_id"])
        if uid in user_histories:
            continue
        raw = row["history"]
        try:
            hist = ast.literal_eval(raw) if isinstance(raw, str) else []
        except Exception:
            hist = []
        if hist:
            user_histories[uid] = hist

    return user_histories

# ---------------------------------------------------------------------------
# Cold-start helper for new profiles
# ---------------------------------------------------------------------------

def _build_cold_start_history(
    baseline: BaselineRecommender,
    selected_categories: List[str],
    articles_per_category: int = 4,
) -> List[str]:
    """
    Build a virtual reading history from the top-popular articles in the
    selected categories.  Used as the preference signal for cold-start users
    who have no real reading history yet.

    4 articles per category gives the recommender enough signal to build a
    meaningful profile without overwhelming any single category.
    """
    cat_map    = baseline.news_id_to_category   # news_id -> category
    pop_scores = baseline.popularity_scores      # news_id -> score

    history: List[str] = []
    for cat in selected_categories:
        cat_articles = [
            (nid, pop_scores.get(nid, 0.0))
            for nid, c in cat_map.items()
            if c == cat and nid in baseline.news_id_to_idx
        ]
        cat_articles.sort(key=lambda x: x[1], reverse=True)
        history.extend(nid for nid, _ in cat_articles[:articles_per_category])

    return history

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "logged_in":      False,
        "user_id":        None,
        "user_history":   [],
        "user_interests": [],   # selected interest categories (new profiles only)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sign-Up / Login Panel
# ---------------------------------------------------------------------------

def render_auth_panel(user_histories: Dict[str, List[str]], baseline: BaselineRecommender):
    """Landing page shown when no one is logged in."""
    st.markdown("## 👋 Welcome to the Diversity-Aware News Recommender")
    st.markdown(
        "This demo shows **real recommendations** from two models built during "
        "the capstone project:\n"
        "- 📊 **Baseline** — accuracy-optimised hybrid model\n"
        "- 🔭 **Re-Ranked** — composite diversity-aware re-ranking\n\n"
        "**Choose how you want to get started:**"
    )
    st.divider()

    tab_new, tab_existing = st.tabs(["🆕 Create New Profile", "🔑 Use Existing User ID"])

    # ── Tab 1: Fresh profile from interest categories ──────────────────────
    with tab_new:
        st.subheader("Select Your Topics of Interest")
        st.caption(
            "Pick the categories you care about. We'll build a fresh recommendation "
            "profile based on your preferences — no pre-read articles."
        )

        all_cats = sorted(set(baseline.news_id_to_category.values()))
        selected_cats = st.multiselect(
            "Topics of interest",
            options=all_cats,
            default=[],
            placeholder="Select one or more categories…",
            key="new_profile_cats",
        )

        if st.button(
            "🚀 Get My Recommendations",
            type="primary",
            disabled=(len(selected_cats) == 0),
            key="btn_new_profile",
        ):
            virtual_history = _build_cold_start_history(baseline, selected_cats)
            uid = "custom_" + hashlib.md5(
                (str(selected_cats) + str(time.time())).encode()
            ).hexdigest()[:6]
            st.session_state.logged_in      = True
            st.session_state.user_id        = uid
            st.session_state.user_history   = virtual_history
            st.session_state.user_interests = selected_cats
            st.rerun()

    # ── Tab 2: Existing test-set user ──────────────────────────────────────
    with tab_existing:
        st.subheader("Sign In with a Test-Set User ID")
        valid_ids = sorted(user_histories.keys())
        st.caption(
            f"Any of the **{len(valid_ids)} test-set user IDs** will work. "
            "These users have real reading histories from the MIND dataset."
        )

        chosen = st.selectbox("Select a User ID", options=[""] + valid_ids,
                              key="signup_select")
        typed  = st.text_input("… or type a User ID", placeholder="e.g. U87243",
                               key="signup_text")

        user_id = (typed.strip() if typed.strip() else chosen).strip()

        if st.button(
            "🚀 Get My Recommendations",
            type="primary",
            disabled=(user_id == ""),
            key="btn_existing",
        ):
            if user_id not in user_histories:
                st.error(
                    f"User ID **{user_id}** not found in the test set. "
                    f"Please choose one from the list above."
                )
            else:
                st.session_state.logged_in      = True
                st.session_state.user_id        = user_id
                st.session_state.user_history   = user_histories[user_id]
                st.session_state.user_interests = []
                st.rerun()

        with st.expander("📋 Available test-set User IDs"):
            st.write(", ".join(valid_ids))

# ---------------------------------------------------------------------------
# Sidebar (shown when logged in)
# ---------------------------------------------------------------------------

def render_sidebar(reranker, news_df, tuner) -> dict:
    st.sidebar.title("🎛️ Controls")

    st.sidebar.success(f"👤 Logged in as **{st.session_state.user_id}**")

    interests = st.session_state.user_interests
    if interests:
        st.sidebar.caption(f"Interests: {', '.join(interests)}")
    else:
        st.sidebar.caption(f"History: {len(st.session_state.user_history)} articles")

    if st.sidebar.button("🚪 Log Out"):
        for k in ["logged_in", "user_id", "user_history", "user_interests"]:
            st.session_state[k] = (False if k == "logged_in" else
                                   None  if k == "user_id"   else [])
        st.rerun()

    st.sidebar.divider()

    # ── Main slider: Accuracy ↔ Exploration ──────────────────────────────────
    st.sidebar.subheader("🔭 Accuracy vs Exploration")
    explore_level = st.sidebar.slider(
        "Accuracy  ←  →  Exploration",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        format="%.2f",
        help=(
            "0 = pure accuracy (baseline ranking, no re-ranking)  |  "
            "1 = full exploration (all four diversity pillars active)"
        ),
        key="explore_level",
    )

    params: dict = {"explore_level": explore_level}

    # ── Four pillar sub-sliders (only shown when exploration > 0) ────────────
    if explore_level > 0.0:
        st.sidebar.divider()
        st.sidebar.subheader("🎚️ Exploration Pillars")
        st.sidebar.caption(
            "Controls how the exploration budget is divided among the four "
            "diversity dimensions. All default to 50%."
        )

        w_div = st.sidebar.slider(
            "🌐 Diversity",
            0.0, 1.0, 0.5, 0.05,
            help="Penalises embedding-similar articles appearing together in the list.",
            key="w_div",
        )
        w_cal = st.sidebar.slider(
            "⚖️ Calibration",
            0.0, 1.0, 0.5, 0.05,
            help="Matches the category mix of recommendations to your interest profile.",
            key="w_cal",
        )
        w_ser = st.sidebar.slider(
            "✨ Serendipity",
            0.0, 1.0, 0.5, 0.05,
            help="Surfaces articles outside your usual reading patterns.",
            key="w_ser",
        )
        w_fair = st.sidebar.slider(
            "🤝 Fairness",
            0.0, 1.0, 0.5, 0.05,
            help=(
                "Prefers articles whose popularity level matches your "
                "reading habits (avoids forcing viral or ultra-niche content)."
            ),
            key="w_fair",
        )

        params.update({"w_div": w_div, "w_cal": w_cal, "w_ser": w_ser, "w_fair": w_fair})

    return params

# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _compute_composite_weights(params: dict) -> dict | None:
    """
    Translate the explore_level + pillar sliders into composite_rerank kwargs.

    explore_level = 0  →  None  (skip reranking entirely, use baseline top-10)
    explore_level = x  →  w_relevance = 1-x; remaining x split among pillars
                          proportional to their sub-slider values.

    Example (explore=0.6, all pillars at 0.5):
        total_sub = 2.0
        w_relevance   = 0.40
        w_diversity   = 0.60 × 0.5/2.0 = 0.15
        w_calibration = 0.15
        w_serendipity = 0.15
        w_fairness    = 0.15
    """
    explore_level = params["explore_level"]

    if explore_level == 0.0:
        return None  # signal: no reranking needed

    w_div  = params.get("w_div",  0.5)
    w_cal  = params.get("w_cal",  0.5)
    w_ser  = params.get("w_ser",  0.5)
    w_fair = params.get("w_fair", 0.5)

    total_sub = w_div + w_cal + w_ser + w_fair
    if total_sub == 0:
        total_sub = 1.0  # avoid division by zero; treat all pillars as equal

    return {
        "w_relevance":   1.0 - explore_level,
        "w_diversity":   explore_level * (w_div  / total_sub),
        "w_calibration": explore_level * (w_cal  / total_sub),
        "w_serendipity": explore_level * (w_ser  / total_sub),
        "w_fairness":    explore_level * (w_fair / total_sub),
        # explore_weight steers calibration target toward uniform distribution
        "explore_weight": explore_level,
    }

# ---------------------------------------------------------------------------
# Article Card
# ---------------------------------------------------------------------------

def render_article_card(row, rank: int, score: float):
    title       = row["title"]
    abstract    = row.get("abstract", "")
    category    = row["category"]
    subcategory = row.get("subcategory", "")

    with st.container():
        c1, c2 = st.columns([1, 15])
        with c1:
            st.markdown(f"### {rank}")
        with c2:
            st.markdown(f"**{title}**")
            st.caption(f"🏷️ {category} / {subcategory}  |  ⭐ {score:.4f}")
            if abstract:
                st.write(abstract[:200] + ("…" if len(abstract) > 200 else ""))
        st.divider()

# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

def render_metrics(recs_baseline, recs_reranked, news_df):
    def cat_list(recs):
        ids  = [nid for nid, _ in recs]
        cats = news_df.set_index("news_id")["category"].to_dict()
        return [cats[nid] for nid in ids if nid in cats]

    def gini(cats):
        from collections import Counter
        counts = list(Counter(cats).values())
        n = len(counts)
        if n == 0:
            return 0.0
        arr = np.sort(counts)
        idx = np.arange(1, n + 1)
        return float((2 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1) / n)

    base_cats   = cat_list(recs_baseline)
    rerank_cats = cat_list(recs_reranked)
    all_cats    = news_df["category"].unique()

    st.subheader("📈 Diversity Metrics Comparison")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Baseline Gini ↓", f"{gini(base_cats):.3f}",
                  help="Lower Gini = categories more evenly spread")
    with col2:
        st.metric("Re-Ranked Gini ↓", f"{gini(rerank_cats):.3f}")
    with col3:
        st.metric("Baseline Coverage",
                  f"{len(set(base_cats))}/{len(all_cats)} cats")
    with col4:
        st.metric("Re-Ranked Coverage",
                  f"{len(set(rerank_cats))}/{len(all_cats)} cats")

    from collections import Counter
    base_counts   = Counter(base_cats)
    rerank_counts = Counter(rerank_cats)
    all_cat_keys  = sorted(set(base_counts) | set(rerank_counts))

    chart_df = pd.DataFrame({
        "Category":  all_cat_keys,
        "Baseline":  [base_counts.get(c, 0)   for c in all_cat_keys],
        "Re-Ranked": [rerank_counts.get(c, 0) for c in all_cat_keys],
    })

    fig = px.bar(
        chart_df.melt(id_vars="Category", var_name="Model", value_name="Count"),
        x="Category", y="Count", color="Model", barmode="group",
        title="Category Distribution: Baseline vs Re-Ranked",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Main Recommendations View
# ---------------------------------------------------------------------------

def render_recommendations(baseline, reranker, news_df, params):
    history       = st.session_state.user_history
    explore_level = params["explore_level"]

    composite_kwargs = _compute_composite_weights(params)

    with st.spinner("Generating recommendations…"):
        baseline_candidates = baseline.recommend(
            user_history=history,
            k=100,
            exclude_history=True,
        )

        if not baseline_candidates:
            st.error("No candidates returned by the baseline model. "
                     "Try a different user or interest selection.")
            return

        recs_baseline = baseline_candidates[:10]

        if composite_kwargs is None:
            # explore_level == 0: pure accuracy, show baseline on both sides
            recs_reranked = recs_baseline
        else:
            recs_reranked = reranker.rerank(
                candidates=baseline_candidates,
                user_history=history,
                k=10,
                method="composite",
                **composite_kwargs,
            )

    # Mode label
    if explore_level == 0.0:
        mode_label = "Pure Accuracy (no re-ranking)"
    else:
        pct = f"{explore_level:.0%}"
        mode_label = (
            f"Exploration {pct}  |  "
            f"Relevance {1 - explore_level:.0%}  +  "
            f"Diversity / Calibration / Serendipity / Fairness  {pct}"
        )

    st.markdown(f"**History:** {len(history)} articles  |  **Mode:** {mode_label}")

    col_b, col_r = st.columns(2)
    news_idx = news_df.set_index("news_id")

    with col_b:
        st.subheader("📊 Baseline Recommendations")
        st.caption("Accuracy-optimised hybrid model")
        for rank, (nid, score) in enumerate(recs_baseline, 1):
            if nid in news_idx.index:
                render_article_card(news_idx.loc[nid], rank, score)

    with col_r:
        if explore_level == 0.0:
            st.subheader("📊 Re-Ranked (same as baseline)")
            st.caption("Move the slider right to enable exploration ↗")
        else:
            st.subheader("🔭 Re-Ranked (Composite Explorer)")
            st.caption(
                f"Relevance {1 - explore_level:.0%}  |  "
                f"Diversity + Calibration + Serendipity + Fairness  "
                f"{explore_level:.0%}"
            )
        for rank, (nid, score) in enumerate(recs_reranked, 1):
            if nid in news_idx.index:
                render_article_card(news_idx.loc[nid], rank, score)

    st.divider()
    render_metrics(recs_baseline, recs_reranked, news_df)

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    init_session()

    st.title("📰 Diversity-Aware News Recommender")
    st.markdown("*Breaking Echo Chambers with AI — Capstone Demo*")

    baseline, reranker, news_df, tuner = load_system()
    user_histories = load_user_histories()

    if not st.session_state.logged_in:
        render_auth_panel(user_histories, baseline)
    else:
        params = render_sidebar(reranker, news_df, tuner)
        st.header(f"🎯 Recommendations for {st.session_state.user_id}")
        render_recommendations(baseline, reranker, news_df, params)


if __name__ == "__main__":
    main()
