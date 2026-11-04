"""
FraudShield - FastAPI application
==================================

This module provides the web API for FraudShield. It loads the R2 predictor
and exposes a /api/predict endpoint for real-time URL risk analysis.

Run:
    uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Ensure project root is in sys.path if running as "python backend/app.py"
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.predictor import predict_url
from backend.inspector import inspect_website
from backend.risk_engine import decide

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraudshield")

# ---------------------------------------------------------------------------
# Interactive API documentation (Swagger UI, ReDoc, OpenAPI schema)
#
# These development conveniences are enabled by default but disabled in
# production so the public API schema is not exposed on the deployed site.
#
# Environment variables (set on Render via render.yaml):
#   FRAUDSHIELD_ENV             "production" / "prod" -> docs OFF
#                               anything else         -> docs ON (default: development)
#   FRAUDSHIELD_ENABLE_API_DOCS "1" / "true" / "yes"  -> force docs ON (overrides above)
# ---------------------------------------------------------------------------
_raw_env = os.environ.get("FRAUDSHIELD_ENV", "development").strip().lower()
_is_production = _raw_env in ("production", "prod")
_docs_forced = os.environ.get("FRAUDSHIELD_ENABLE_API_DOCS", "").strip().lower() in ("1", "true", "yes")
_enable_docs = _docs_forced or not _is_production

app = FastAPI(
    title="FraudShield API",
    description="URL Risk Prediction API using Machine Learning (R2 Model)",
    version="1.0.0",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
)

# Enable CORS so the Vite frontend (port 5173) can talk to FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class URLRequest(BaseModel):
    """A URL to scan for risk."""
    url: str = Field(
        ...,
        json_schema_extra={"example": "https://secure-login.example/verify"},
    )

    @field_validator("url")
    @classmethod
    def validate_url_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("URL cannot be empty.")
        return cleaned

class PredictionResponse(BaseModel):
    """The risk assessment for a URL.

    Verdict fields (``prediction``, ``risk_level``, ``risk_score``, ``reasons``)
    describe the FINAL policy-adjusted verdict. The untouched R2 model output
    is preserved in ``model_score`` and ``raw_*`` fields so diagnostics and
    logs always see the original probability and prediction.
    """

    url: str
    valid: bool
    prediction: int | None
    prediction_label: str | None
    risk_level: str
    risk_score: float | None
    raw_prediction: int | None
    raw_prediction_label: str | None
    raw_risk_level: str | None
    raw_risk_score: float | None
    model_score: float | None
    reasons: list[str]
    features: dict[str, int]
    policy: dict[str, Any] | None = None
    model_info: dict[str, Any] = Field(..., alias="model")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Verify that the API and model are ready."""
    return {
        "status": "ok",
        "model": "R2 (Experiment C Winner)",
        "engine": "layered website scanner (R2 + live inspection + policy)",
    }


class ScanResponse(BaseModel):
    """Full layered scan: verdict, evidence, inspection and raw model output."""

    url: str
    verdict: str
    availability: str
    evidence: list[str]
    findings: list[dict[str, Any]]
    inspection: dict[str, Any]
    security: dict[str, Any]
    url_model: dict[str, Any]
    signals: dict[str, Any]
    threat_intelligence: dict[str, Any]

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: URLRequest):
    """
    Analyze a URL and return a risk assessment.

    This endpoint takes a URL string, extracts lexical features, and runs
    it through the FraudShield R2 machine-learning model.
    """
    logger.info(f"Scanning URL: {request.url}")

    try:
        result = predict_url(request.url)
        if not result.get("valid"):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": result.get("error", "invalid_url"),
                    "message": result.get("message", "Invalid URL."),
                },
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for URL '{request.url}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during URL analysis.")


@app.post("/api/scan")
async def scan(request: URLRequest):
    """
    Full layered website scan for one URL.

    1. Runs the R2 URL-string model (raw score preserved, never decisive).
    2. Performs the safe live website inspection (DNS + SSRF validation,
       TLS certificate check, redirect re-validation, read-only HTML
       analysis).
    3. Combines both plus policy notes into an evidence-based verdict:
       SAFE / SUSPICIOUS / MALICIOUS / INSUFFICIENT EVIDENCE,
       plus an independent availability classification (REACHABLE /
       PAGE_NOT_FOUND / TIMEOUT / CONNECTION_FAILED / DNS_FAILED /
       TITLE_UNAVAILABLE / UNKNOWN).
    """
    logger.info(f"Scanning URL with live inspection: {request.url}")

    try:
        url_model_result = predict_url(request.url)
        if not url_model_result.get("valid"):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": url_model_result.get("error", "invalid_url"),
                    "message": url_model_result.get("message", "Invalid URL."),
                },
            )

        inspection = inspect_website(request.url)
        assessment = decide(url_model_result, inspection)
        logger.info(
            f"Inspection verdict for '{request.url}': {assessment['verdict']} "
            f"(R2 raw prediction {url_model_result.get('raw_prediction')}, "
            f"score {url_model_result.get('model_score')})"
        )

        url_only = dict(url_model_result)
        url_only.pop("features", None)  # keep the response focused

        security = {
            "tls": inspection.get("tls"),
            "dns": inspection.get("dns"),
            "redirects": inspection.get("hops"),
            "threat_intelligence": assessment.get("threat_intelligence"),
        }

        return {
            "url": request.url,
            "verdict": assessment["verdict"],
            "availability": assessment["availability"],
            "evidence": assessment["evidence"],
            "findings": assessment["findings"],
            "inspection": inspection,
            "security": security,
            "url_model": url_only,
            "signals": assessment["signals"],
            "threat_intelligence": assessment["threat_intelligence"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Scan error for URL '{request.url}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during website scan.")

# ---------------------------------------------------------------------------
# Static frontend (served after API routes)
# ---------------------------------------------------------------------------
# Mounting the frontend last lets the app serve the polished UI at "/" with a
# single command: python -m uvicorn backend.app:app --port 8000
# The Vite dev server (port 5173) also continues to work via CORS.
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
