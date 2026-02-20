"""
main.py — FastAPI REST API for the NewsLens recommendation system.

Endpoints:
  GET  /api/health
  POST /api/session
  POST /api/quiz
  POST /api/click
  POST /api/rerank
  GET  /api/article/{id}
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from personalized_tuner import PersonalizedWeightTuner, infer_time_of_day
    _tuner = PersonalizedWeightTuner(learning_rate=0.1)
    _tuner_available = True
except Exception:
    _tuner_available = False

from recommender_service import RecommenderService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(title="NewsLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE_DIR = Path(__file__).resolve().parent
service = RecommenderService(_BASE_DIR)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

SESSION_TTL = timedelta(hours=24)

sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    sess = sessions.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    sess["last_accessed"] = datetime.utcnow()
    return sess


def _new_session(session_id: str | None = None) -> dict:
    sid = session_id or str(uuid.uuid4())[:8]
    sess = {
        "session_id": sid,
        "history": [],
        "seed_history": [],   # immutable copy of initial history — used for reset
        "last_recs": [],
        "quiz_prefs": None,
        "clicked_count": 0,   # user-initiated reads only (excludes cold-start seeds)
        "created_at": datetime.utcnow(),
        "last_accessed": datetime.utcnow(),
    }
    sessions[sid] = sess
    return sess


def _cleanup_old_sessions() -> None:
    cutoff = datetime.utcnow() - SESSION_TTL
    expired = [k for k, v in sessions.items() if v["last_accessed"] < cutoff]
    for k in expired:
        del sessions[k]


# ---------------------------------------------------------------------------
# Algorithm selection from slider values
# ---------------------------------------------------------------------------

def _sliders_to_method(sliders: dict) -> tuple[str, dict]:
    """
    Map slider values to a composite re-ranking method with proportional weights.

    All sub-sliders contribute simultaneously — not winner-take-all.
    main_diversity controls the overall relevance ↔ diversity trade-off.
    diversity / calibration / serendipity / fairness each share the diversity
    budget proportionally to their own values.

    Weights:
      w_relevance      = max(0.40,  1.0 − main_diversity × 0.60)
      diversity_budget = 1.0 − w_relevance
      w_diversity, w_calibration, w_serendipity, w_fairness share the full
        diversity_budget proportionally to their slider values.
    """
    main_div    = sliders.get("main_diversity", 0.5)
    diversity   = sliders.get("diversity",   0.5)
    calibration = sliders.get("calibration", 0.3)
    serendipity = sliders.get("serendipity", 0.2)
    fairness    = sliders.get("fairness",    0.2)

    if main_div < 0.05:
        return "baseline", {}

    # Relevance weight: 1.0 at main_div=0, 0.40 at main_div=1.0
    w_rel = max(0.40, 1.0 - main_div * 0.60)
    diversity_budget = 1.0 - w_rel

    # Split diversity_budget proportionally among all 4 sub-sliders
    total_sub = diversity + calibration + serendipity + fairness
    if total_sub > 0:
        w_div  = diversity_budget * (diversity   / total_sub)
        w_cal  = diversity_budget * (calibration / total_sub)
        w_ser  = diversity_budget * (serendipity / total_sub)
        w_fair = diversity_budget * (fairness    / total_sub)
    else:
        # All sub-sliders at 0 → equal split
        w_div = w_cal = w_ser = w_fair = diversity_budget / 4.0

    # Exploration prior blends uniform distribution into calibration target
    explore = round(min(0.5, main_div * 0.4), 4)

    return "composite", {
        "w_relevance":   round(w_rel, 4),
        "w_diversity":   round(w_div, 4),
        "w_calibration": round(w_cal, 4),
        "w_serendipity": round(w_ser, 4),
        "w_fairness":    round(w_fair, 4),
        "explore_weight": explore,
    }


def _style_to_sliders(style: str) -> dict:
    # accurate → baseline model (main_diversity < 0.1 threshold → "baseline")
    if style == "accurate":
        return {"main_diversity": 0.0, "diversity": 0.0, "calibration": 0.0, "serendipity": 0.0, "fairness": 0.0}
    if style == "explore":
        return {"main_diversity": 0.9, "diversity": 0.5, "calibration": 0.1, "serendipity": 0.8, "fairness": 0.2}
    # balanced (default)
    return {"main_diversity": 0.5, "diversity": 0.5, "calibration": 0.3, "serendipity": 0.2, "fairness": 0.2}


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _build_rec_response(recs: list[dict], history: list[str], sliders: dict) -> dict:
    rec_ids = [r["news_id"] for r in recs]
    metrics = service.compute_metrics(rec_ids, history)
    method, _ = _sliders_to_method(sliders)
    cat_dist = dict(Counter(r["category"] for r in recs))
    return {
        "recommendations": [
            {
                "news_id": r["news_id"],
                "title": r["title"],
                "category": r["category"],
                "subcategory": r.get("subcategory", ""),
                "abstract": r.get("abstract", ""),
                "score": round(float(r.get("score", 0.5)), 4),
            }
            for r in recs
        ],
        "metrics": metrics,
        "category_distribution": cat_dist,
        "active_method": method,
    }


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SessionRequest(BaseModel):
    session_id: str | None = None


class QuizPreferences(BaseModel):
    topics: list[str]
    style: str = "balanced"


class QuizRequest(BaseModel):
    session_id: str
    preferences: QuizPreferences


class ClickRequest(BaseModel):
    session_id: str
    news_id: str
    sliders: dict[str, float] | None = None


class RerankRequest(BaseModel):
    session_id: str
    sliders: dict[str, float]


class LoginRequest(BaseModel):
    user_id: str


class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    _cleanup_old_sessions()
    return {
        "status": "ok",
        "mock_mode": service.mock_mode,
        "active_sessions": len(sessions),
    }


@app.post("/api/session")
def create_or_restore_session(req: SessionRequest) -> dict:
    _cleanup_old_sessions()
    if req.session_id and req.session_id in sessions:
        sess = sessions[req.session_id]
        sess["last_accessed"] = datetime.utcnow()
        return {
            "session_id": sess["session_id"],
            "history_count": len(sess["history"]),
            "quiz_completed": sess["quiz_prefs"] is not None,
        }
    sess = _new_session(req.session_id)
    return {
        "session_id": sess["session_id"],
        "history_count": 0,
        "quiz_completed": False,
    }


@app.post("/api/quiz")
def submit_quiz(req: QuizRequest) -> dict:
    # Ensure session exists
    if req.session_id not in sessions:
        _new_session(req.session_id)
    sess = _get_session(req.session_id)

    sliders = _style_to_sliders(req.preferences.style)
    sess["quiz_prefs"] = {
        "topics": req.preferences.topics,
        "style": req.preferences.style,
        "sliders": sliders,
    }

    # Cold-start: seed history with top-5 popular articles from each selected topic.
    # Aliases handle topics not in the real MIND dataset (e.g. science → news).
    seed_history: list[str] = []
    for topic in req.preferences.topics:
        popular = service.get_popular_by_category(topic, k=5)
        seed_history.extend(a["news_id"] for a in popular)
    sess["history"] = seed_history
    sess["seed_history"] = list(seed_history)   # save for reset
    sess["clicked_count"] = 0                   # user hasn't read anything yet

    # Cold-start initial recommendations: heavy calibration, no serendipity.
    # This ensures the user's selected topics dominate the first feed rather
    # than being crowded out by the serendipity bonus for unseen categories.
    cold_start_params = {
        "w_relevance":   0.40,
        "w_diversity":   0.15,
        "w_calibration": 0.30,
        "w_serendipity": 0.00,
        "w_fairness":    0.15,
        "explore_weight": 0.0,
    }
    recs = service.rerank(history=seed_history, k=10, method="composite", **cold_start_params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    return {
        **_build_rec_response(recs, seed_history, sliders),
        "history_count": 0,   # seed articles are internal; user hasn't read anything
    }


@app.post("/api/click")
def record_click(req: ClickRequest) -> dict:
    sess = _get_session(req.session_id)
    if req.news_id not in sess["history"]:
        sess["history"].append(req.news_id)

    # Feed implicit click signal into the personalized tuner
    diversity_preference: float | None = None
    if _tuner_available:
        art = service.get_article(req.news_id)
        _tuner.update_from_interactions(req.session_id, [{
            "action": "click",
            "news_id": req.news_id,
            "category": art["category"] if art else "unknown",
        }])
        patterns = _tuner.interaction_patterns[req.session_id]
        if patterns["total_interactions"] >= 5:
            diversity_preference = round(patterns["diversity_preference_score"], 3)

    sliders = req.sliders or (
        sess["quiz_prefs"]["sliders"] if sess["quiz_prefs"] else _style_to_sliders("balanced")
    )
    method, params = _sliders_to_method(sliders)
    recs = service.rerank(history=sess["history"], k=10, method=method, **params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    sess["clicked_count"] = sess.get("clicked_count", 0) + 1
    response = {
        **_build_rec_response(recs, sess["history"], sliders),
        "history_count": sess["clicked_count"],
    }
    if diversity_preference is not None:
        response["diversity_preference"] = diversity_preference
    return response


@app.post("/api/rerank")
def rerank(req: RerankRequest) -> dict:
    sess = _get_session(req.session_id)
    method, params = _sliders_to_method(req.sliders)
    recs = service.rerank(history=sess["history"], k=10, method=method, **params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    # Update stored slider values
    if sess["quiz_prefs"]:
        sess["quiz_prefs"]["sliders"] = req.sliders

    return _build_rec_response(recs, sess["history"], req.sliders)


@app.get("/api/users")
def list_users() -> dict:
    """Return demo user profiles available for login."""
    return {"users": service.get_available_users()}


@app.post("/api/login")
def login_with_profile(req: LoginRequest) -> dict:
    """Create a session pre-loaded with an existing user's history.
    If the user ID is unknown, create a fresh session so the frontend
    can redirect them to the quiz instead of returning an error."""
    _cleanup_old_sessions()
    user_info = service.get_user_info(req.user_id)
    if user_info is None:
        sess = _new_session()
        return {
            "session_id": sess["session_id"],
            "is_new_user": True,
            "display_name": req.user_id,
        }

    sess = _new_session()  # fresh session id
    history = user_info.get("history", [])
    sess["history"] = list(history)
    sess["seed_history"] = list(history)   # save for reset
    sess["clicked_count"] = len(history)   # existing users already read their history
    sess["quiz_prefs"] = {
        "topics": user_info.get("top_categories", []),
        "style": "balanced",
        "sliders": _style_to_sliders("balanced"),
        "user_id": req.user_id,
        "display_name": user_info.get("display_name", req.user_id),
    }

    sliders = _style_to_sliders("balanced")
    method, params = _sliders_to_method(sliders)
    recs = service.rerank(history=history, k=10, method=method, **params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    return {
        **_build_rec_response(recs, history, sliders),
        "session_id": sess["session_id"],
        "history_count": len(history),
        "display_name": user_info.get("display_name", req.user_id),
    }


@app.post("/api/reset")
def reset_session(req: ResetRequest) -> dict:
    """Restore session history to the initial seed and return fresh recommendations."""
    sess = _get_session(req.session_id)
    sess["history"] = list(sess.get("seed_history", []))
    sess["clicked_count"] = 0

    sliders = (
        sess["quiz_prefs"]["sliders"] if sess.get("quiz_prefs")
        else _style_to_sliders("balanced")
    )
    method, params = _sliders_to_method(sliders)
    recs = service.rerank(history=sess["history"], k=10, method=method, **params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    return {
        **_build_rec_response(recs, sess["history"], sliders),
        "history_count": 0,
    }


@app.get("/api/profile/{session_id}")
def get_session_profile(session_id: str) -> dict:
    """
    Return the personalized diversity profile inferred from this session's clicks.
    Requires at least 5 interactions before the score becomes meaningful.
    """
    sess = _get_session(session_id)
    if not _tuner_available:
        return {"available": False}
    patterns = _tuner.interaction_patterns[session_id]
    n = patterns["total_interactions"]
    score = patterns["diversity_preference_score"] if n >= 5 else None
    weights = _tuner.get_user_weights(
        session_id,
        context={"time_of_day": infer_time_of_day(), "session_length": n},
    ) if n >= 5 else None
    return {
        "available": True,
        "total_interactions": n,
        "diversity_preference_score": round(score, 3) if score is not None else None,
        "inferred_weights": {k: round(v, 3) for k, v in weights.items()} if weights else None,
        "label": (
            "Explorer" if (score or 0) > 0.65
            else "Specialist" if (score or 1) < 0.35
            else "Balanced reader"
        ) if score is not None else None,
    }


@app.get("/api/compare/{session_id}")
def compare_feeds(session_id: str) -> dict:
    """
    Return baseline vs diversity recommendations side-by-side.
    Read-only — does NOT mutate session slider state.
    """
    sess = _get_session(session_id)
    history = sess.get("history", [])

    baseline_sliders  = {"main_diversity": 0.0, "diversity": 0.0, "calibration": 0.0, "serendipity": 0.0, "fairness": 0.0}
    diversity_sliders = {"main_diversity": 0.5, "diversity": 0.5, "calibration": 0.3, "serendipity": 0.2, "fairness": 0.2}

    baseline_recs = service.rerank(history=history, k=10, method="baseline")
    div_method, div_params = _sliders_to_method(diversity_sliders)
    diversity_recs = service.rerank(history=history, k=10, method=div_method, **div_params)

    def _fmt(recs: list[dict], sliders: dict) -> dict:
        ids = [r["news_id"] for r in recs]
        return {
            "recommendations": [
                {
                    "news_id": r["news_id"],
                    "title": r["title"],
                    "category": r["category"],
                    "subcategory": r.get("subcategory", ""),
                    "abstract": r.get("abstract", ""),
                    "score": round(float(r.get("score", 0.5)), 4),
                }
                for r in recs
            ],
            "metrics": service.compute_metrics(ids, history),
            "category_distribution": dict(Counter(r["category"] for r in recs)),
            "active_method": _sliders_to_method(sliders)[0],
        }

    return {
        "baseline": _fmt(baseline_recs, baseline_sliders),
        "diversity": _fmt(diversity_recs, diversity_sliders),
    }


@app.get("/api/history/{session_id}")
def get_history(session_id: str) -> dict:
    """Return article objects for the session's reading history (most recent first)."""
    sess = _get_session(session_id)
    history = sess.get("history", [])
    articles = []
    for news_id in reversed(history[-30:]):   # up to 30 most recent
        art = service.get_article(news_id)
        if art:
            articles.append({
                "news_id": news_id,
                "title": art.get("title", news_id),
                "category": art.get("category", ""),
                "subcategory": art.get("subcategory", ""),
            })
    return {"articles": articles, "total": len(history)}


@app.get("/api/article/{news_id}")
def get_article(news_id: str) -> dict:
    art = service.get_article(news_id)
    if art is None:
        raise HTTPException(status_code=404, detail=f"Article {news_id} not found")
    return art
