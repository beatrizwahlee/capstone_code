"""
Phase 5: Interactive Web App — Streamlit
=========================================
Diversity-Aware News Recommender System — Capstone Project

This Streamlit app provides an interactive interface for:
  1. Generating personalized news recommendations
  2. Visualizing diversity metrics in real-time
  3. Comparing baseline vs diversity-aware recommendations
  4. Tuning diversity preferences with sliders
  5. (Future) LLM-powered content reframing

Features:
---------
- **User Profile Builder**: Click on articles to build history
- **Algorithm Selector**: Choose baseline, MMR, xQuAD, Calibrated, or Serendipity
- **Diversity Sliders**: Tune λ, α, β parameters in real-time
- **Side-by-side Comparison**: See baseline vs diversity-aware recommendations
- **Metrics Dashboard**: Live Gini, ILD, Coverage visualization
- **Article Preview**: Click to see full article details

Usage:
    streamlit run app.py

    # Or with custom port
    streamlit run app.py --server.port 8501
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add project paths (script-relative, robust to cwd)
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

phase2_path = base_dir.parent / "Phase2_baseline_rec"
if phase2_path.exists():
    sys.path.insert(0, str(phase2_path))

phase3_path = base_dir.parent / "Phase3_echo_chambers"
if phase3_path.exists():
    sys.path.insert(0, str(phase3_path))

phase4_path = base_dir.parent / "Phase4_reranker"
if phase4_path.exists():
    sys.path.insert(0, str(phase4_path))

phase1_path = base_dir.parent / "Phase1_NLP_encoding"
if phase1_path.exists():
    sys.path.insert(0, str(phase1_path))

phase0_path = base_dir.parent / "Phase0_data_processing" / "processed_data"
phase0_code_path = base_dir.parent / "Phase0_data_processing" / "data_processing"
if phase0_code_path.exists():
    sys.path.insert(0, str(phase0_code_path))

logger = logging.getLogger(__name__)

# Import our modules
from baseline_recommender_phase2 import BaselineRecommender
from diversity_reranker import DiversityReranker
from echo_chamber_analyzer import EchoChamberAnalyzer
from personalized_tuner import PersonalizedWeightTuner, infer_time_of_day
from llm_reframer import LLMContentReframer


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Diversity-Aware News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load Models (Cached)
# ---------------------------------------------------------------------------

def load_system():
    if 'system_cache' in st.session_state and st.session_state['system_cache'] is not None:
        return st.session_state['system_cache']
    st.info("🔄 Loading recommender system... (this may take 30 seconds)")
    
    embeddings_dir = str(base_dir.parent / "Phase1_NLP_encoding" / "embeddings")
    baseline_path = str(base_dir.parent / "Phase2_baseline_rec" / "outputs" / "baseline" / "baseline_recommender.pkl")
    news_path = str(base_dir.parent / "Phase0_data_processing" / "processed_data" / "news_features_train.csv")
    
    baseline = BaselineRecommender.load(baseline_path, embeddings_dir)
    
    news_df = pd.read_csv(news_path)
    news_categories = dict(zip(news_df['news_id'], news_df['category']))
    
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
    
    tuner = PersonalizedWeightTuner()
    
    try:
        reframer = LLMContentReframer(provider='anthropic')
        has_llm = True
    except Exception as e:
        logger.warning(f"LLM reframer not available: {e}")
        reframer = None
        has_llm = False
    
    st.success("✅ System loaded successfully!")
    
    result = (baseline, reranker, analyzer, news_df, tuner, reframer, has_llm)
    st.session_state['system_cache'] = result
    return result


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize session state variables."""
    if 'user_history' not in st.session_state:
        st.session_state.user_history = []
    
    if 'selected_algorithm' not in st.session_state:
        st.session_state.selected_algorithm = 'baseline'
    
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar(reranker, news_df, tuner):
    """Render sidebar with controls."""
    st.sidebar.title("🎛️ Controls")
    
    # Personalization toggle
    st.sidebar.subheader("🎯 Personalization")
    use_personalized = st.sidebar.checkbox(
        "Use Personalized Weights",
        value=False,
        help="Learn optimal diversity settings from your interactions",
    )
    
    # Algorithm selection
    st.sidebar.subheader("1. Select Algorithm")
    algorithm = st.sidebar.selectbox(
        "Recommendation Method",
        ['baseline', 'mmr', 'xquad', 'calibrated', 'serendipity'],
        format_func=lambda x: {
            'baseline': '📊 Baseline (Pure Accuracy)',
            'mmr': '🎯 MMR (Balanced)',
            'xquad': '📚 xQuAD (Category Coverage)',
            'calibrated': '⚖️ Calibrated (Match History)',
            'serendipity': '✨ Serendipity (Unexpected)',
        }[x],
        key='algorithm_selector',
    )
    
    st.session_state.selected_algorithm = algorithm
    
    # Get user ID (or create one)
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{np.random.randint(10000, 99999)}"
    
    user_id = st.session_state.user_id
    
    # Algorithm-specific parameters
    params = {}
    
    if use_personalized:
        # Get personalized weights
        context = {
            'time_of_day': infer_time_of_day(),
            'device': 'desktop',  # Could detect from user agent
            'session_length': len(st.session_state.user_history),
        }
        personal_weights = tuner.get_user_weights(user_id, context)
        
        st.sidebar.info(f"**Personalized Settings**\n\n"
                       f"λ: {personal_weights['lambda']:.2f}\n\n"
                       f"α: {personal_weights['alpha']:.2f}\n\n"
                       f"β: {personal_weights['beta']:.2f}\n\n"
                       f"Confidence: {personal_weights['confidence']:.0%}")
        
        if algorithm == 'mmr':
            params['lambda_param'] = personal_weights['lambda']
        elif algorithm == 'xquad':
            params['lambda_param'] = personal_weights['lambda']
        elif algorithm == 'calibrated':
            params['alpha'] = personal_weights['alpha']
        elif algorithm == 'serendipity':
            params['beta'] = personal_weights['beta']
    
    else:
        # Manual parameter tuning
        if algorithm == 'mmr':
            st.sidebar.subheader("2. Tune Parameters")
            lambda_param = st.sidebar.slider(
                "λ (Diversity Weight)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="0.0 = Pure diversity, 1.0 = Pure relevance",
            )
            params['lambda_param'] = lambda_param
            
            st.sidebar.info(f"**Current:** {'High Diversity' if lambda_param < 0.4 else 'Balanced' if lambda_param < 0.7 else 'High Relevance'}")
        
        elif algorithm == 'xquad':
            st.sidebar.subheader("2. Tune Parameters")
            lambda_param = st.sidebar.slider(
                "λ (Coverage Weight)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher = More category coverage",
            )
            params['lambda_param'] = lambda_param
        
        elif algorithm == 'calibrated':
            st.sidebar.subheader("2. Tune Parameters")
            alpha = st.sidebar.slider(
                "α (Calibration Strength)",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Higher = Stricter match to history",
            )
            params['alpha'] = alpha
        
        elif algorithm == 'serendipity':
            st.sidebar.subheader("2. Tune Parameters")
            beta = st.sidebar.slider(
                "β (Serendipity Weight)",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.1,
                help="Higher = More unexpected items",
            )
            params['beta'] = beta
    
    # Comparison mode
    st.sidebar.subheader("3. View Options")
    show_comparison = st.sidebar.checkbox(
        "Show Baseline Comparison",
        value=False,
        help="Display baseline and diversity-aware recommendations side-by-side",
    )
    st.session_state.show_comparison = show_comparison
    
    # LLM reframing toggle
    use_llm_reframing = st.sidebar.checkbox(
        "🤖 Enable AI Content Reframing",
        value=False,
        help="Use LLM to show multiple perspectives",
    )
    st.session_state.use_llm_reframing = use_llm_reframing
    
    # User history
    st.sidebar.subheader("4. Your Reading History")
    st.sidebar.write(f"**{len(st.session_state.user_history)} articles** clicked")
    
    if st.session_state.user_history:
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.user_history = []
            st.experimental_rerun()
        
        # Show last 3 clicked articles
        st.sidebar.write("**Recent clicks:**")
        for nid in st.session_state.user_history[-3:]:
            rows = news_df[news_df['news_id'] == nid]
            if len(rows):
                title = rows.iloc[0]['title']
                st.sidebar.write(f"• {title[:40]}...")
    
    return params, use_personalized


def render_article_card(news_row, rank, score, clickable=True):
    """Render a single article card."""
    news_id = news_row['news_id']
    title = news_row['title']
    abstract = news_row.get('abstract', '')
    category = news_row['category']
    subcategory = news_row.get('subcategory', '')
    
    # Card container
    with st.container():
        col1, col2 = st.columns([1, 15])
        
        with col1:
            st.markdown(f"### {rank}")
        
        with col2:
            # Title
            st.markdown(f"**{title}**")
            
            # Metadata
            st.caption(f"🏷️ {category} / {subcategory}  |  ⭐ Score: {score:.4f}")
            
            # Abstract preview
            if abstract:
                st.write(abstract[:200] + "..." if len(abstract) > 200 else abstract)
            
            # Action buttons
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 8])
            
            if clickable:
                with col_btn1:
                    if st.button("👍 Click", key=f"click_{news_id}_{rank}"):
                        if news_id not in st.session_state.user_history:
                            st.session_state.user_history.append(news_id)
                            st.success(f"Added to history!")
                            st.experimental_rerun()
                
                with col_btn2:
                    if st.button("ℹ️ Details", key=f"details_{news_id}_{rank}"):
                        st.session_state[f'show_details_{news_id}'] = True
            
            # Expandable details
            if st.session_state.get(f'show_details_{news_id}', False):
                with st.expander("📄 Full Article Details", expanded=True):
                    st.write(f"**Category:** {category}")
                    st.write(f"**Subcategory:** {subcategory}")
                    st.write(f"**Abstract:** {abstract}")
                    st.write(f"**News ID:** {news_id}")
                    
                    if st.button("Close", key=f"close_{news_id}_{rank}"):
                        st.session_state[f'show_details_{news_id}'] = False
                        st.experimental_rerun()
        
        st.divider()


def render_metrics_dashboard(recommendations, analyzer):
    """Render real-time diversity metrics."""
    rec_ids = [nid for nid, _ in recommendations]
    
    # Calculate metrics
    rec_cats = [
        analyzer.news_to_category[nid]
        for nid in rec_ids
        if nid in analyzer.news_to_category
    ]
    
    if rec_cats:
        gini = analyzer.calculate_gini(rec_cats)
        ild = analyzer.calculate_ild(rec_ids)
        coverage = analyzer.calculate_coverage(rec_cats)
        entropy = analyzer.calculate_entropy(rec_cats)
    else:
        gini = ild = coverage = entropy = 0.0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Gini Coefficient",
            f"{gini:.3f}",
            delta=f"{'Lower is better' if gini < 0.5 else 'Echo chamber detected'}",
            delta_color="inverse" if gini > 0.5 else "normal",
        )
    
    with col2:
        st.metric(
            "ILD (Diversity)",
            f"{ild:.3f}",
            delta=f"{'Good diversity' if ild > 0.3 else 'Low diversity'}",
            delta_color="normal" if ild > 0.3 else "inverse",
        )
    
    with col3:
        st.metric(
            "Coverage",
            f"{len(set(rec_cats))}/{len(analyzer.all_categories)}",
            delta=f"{coverage*100:.1f}% of categories",
        )
    
    with col4:
        st.metric(
            "Entropy",
            f"{entropy:.3f}",
            delta=f"{'Balanced' if entropy > 1.5 else 'Concentrated'}",
        )
    
    # Category distribution chart
    st.subheader("📊 Category Distribution")
    
    if rec_cats:
        from collections import Counter
        cat_counts = Counter(rec_cats)
        
        fig = px.bar(
            x=list(cat_counts.keys()),
            y=list(cat_counts.values()),
            labels={'x': 'Category', 'y': 'Count'},
            title='Recommendations by Category',
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    
    # Header
    st.title("📰 Diversity-Aware News Recommender")
    st.markdown("**Breaking Echo Chambers with AI** — Your Capstone Project Demo")
    
    # Load system
    baseline, reranker, analyzer, news_df, tuner, reframer, has_llm = load_system()
    
    # Sidebar
    params, use_personalized = render_sidebar(reranker, news_df, tuner)
    
    # Main content
    st.header("🎯 Your Personalized News Feed")
    
    # Check if user has history
    if not st.session_state.user_history:
        st.warning("👋 **Welcome!** Start by clicking on some articles below to build your reading history. "
                   "Then the system will generate personalized recommendations.")
        
        # Show some popular articles to get started
        st.subheader("🔥 Popular Articles (Click to Build History)")
        
        # Get top popular articles
        popular_ids = sorted(
            baseline.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        for i, (news_id, pop_score) in enumerate(popular_ids[:10], 1):
            rows = news_df[news_df['news_id'] == news_id]
            if len(rows):
                render_article_card(rows.iloc[0], i, pop_score)
    
    else:
        # Generate recommendations
        st.info(f"📚 Generating recommendations based on your {len(st.session_state.user_history)} clicks...")
        
        # Get baseline candidates
        baseline_candidates = baseline.recommend(
            user_history=st.session_state.user_history,
            k=100,
            exclude_history=True,
        )
        
        if not baseline_candidates:
            st.error("No recommendations available. Please try adding more articles to your history.")
            return
        
        # Generate diversity-aware recommendations
        if st.session_state.selected_algorithm == 'baseline':
            final_recs = baseline_candidates[:10]
        else:
            final_recs = reranker.rerank(
                candidates=baseline_candidates,
                user_history=st.session_state.user_history,
                k=10,
                method=st.session_state.selected_algorithm,
                **params,
            )
        
        # Display mode
        if st.session_state.show_comparison:
            # Side-by-side comparison
            col_baseline, col_diverse = st.columns(2)
            
            with col_baseline:
                st.subheader("📊 Baseline Recommendations")
                for i, (news_id, score) in enumerate(baseline_candidates[:10], 1):
                    rows = news_df[news_df['news_id'] == news_id]
                    if len(rows):
                        render_article_card(rows.iloc[0], i, score, clickable=False)
            
            with col_diverse:
                st.subheader(f"🎯 {st.session_state.selected_algorithm.upper()} Recommendations")
                for i, (news_id, score) in enumerate(final_recs, 1):
                    rows = news_df[news_df['news_id'] == news_id]
                    if len(rows):
                        render_article_card(rows.iloc[0], i, score)
        
        else:
            # Single column
            for i, (news_id, score) in enumerate(final_recs, 1):
                rows = news_df[news_df['news_id'] == news_id]
                if len(rows):
                    render_article_card(rows.iloc[0], i, score)
        
        # Metrics dashboard
        st.header("📈 Diversity Metrics")
        render_metrics_dashboard(final_recs, analyzer)


if __name__ == "__main__":
    main()
