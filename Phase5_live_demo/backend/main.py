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
        "last_recs": [],
        "quiz_prefs": None,
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
    main_div = sliders.get("main_diversity", 0.5)
    calibration = sliders.get("calibration", 0.3)
    serendipity = sliders.get("serendipity", 0.2)
    fairness = sliders.get("fairness", 0.2)

    if main_div < 0.1:
        return "baseline", {}

    max_sub = max(calibration, serendipity, fairness)
    if serendipity >= max_sub and serendipity > 0:
        return "serendipity", {"beta": float(serendipity)}
    if calibration >= max_sub and calibration > 0:
        return "calibrated", {"alpha": float(calibration)}
    if fairness >= max_sub and fairness > 0:
        return "xquad", {"lambda_param": float(fairness)}
    return "mmr", {"lambda_param": float(1 - main_div)}


def _style_to_sliders(style: str) -> dict:
    if style == "accurate":
        return {"main_diversity": 0.1, "calibration": 0.5, "serendipity": 0.1, "fairness": 0.1}
    if style == "explore":
        return {"main_diversity": 0.8, "calibration": 0.1, "serendipity": 0.6, "fairness": 0.2}
    # balanced (default)
    return {"main_diversity": 0.5, "calibration": 0.3, "serendipity": 0.2, "fairness": 0.2}


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

    # Cold-start: seed history with top-2 popular articles from each selected topic
    seed_history: list[str] = []
    for topic in req.preferences.topics:
        popular = service.get_popular_by_category(topic, k=2)
        seed_history.extend(a["news_id"] for a in popular)
    sess["history"] = seed_history

    # Generate initial recommendations
    method, params = _sliders_to_method(sliders)
    recs = service.rerank(history=seed_history, k=10, method=method, **params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    return _build_rec_response(recs, seed_history, sliders)


@app.post("/api/click")
def record_click(req: ClickRequest) -> dict:
    sess = _get_session(req.session_id)
    if req.news_id not in sess["history"]:
        sess["history"].append(req.news_id)

    sliders = req.sliders or (
        sess["quiz_prefs"]["sliders"] if sess["quiz_prefs"] else _style_to_sliders("balanced")
    )
    method, params = _sliders_to_method(sliders)
    recs = service.rerank(history=sess["history"], k=10, method=method, **params)
    sess["last_recs"] = [r["news_id"] for r in recs]

    return {
        **_build_rec_response(recs, sess["history"], sliders),
        "history_count": len(sess["history"]),
    }


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


@app.get("/api/article/{news_id}")
def get_article(news_id: str) -> dict:
    art = service.get_article(news_id)
    if art is None:
        raise HTTPException(status_code=404, detail=f"Article {news_id} not found")
    return art
