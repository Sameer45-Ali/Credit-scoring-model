"""
Production FastAPI Application for Credit Risk Decisioning.
Exposes real-time scoring, batch evaluation, SHAP explainability, and health monitoring.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

# Ensure src modules are resolvable regardless of execution directory
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from engine import CreditDecisionEngine
from schemas import (
    ApplicantProfile,
    ScoringResponse,
    BatchApplicantsRequest,
    BatchScoringResponse,
    ModelInfoResponse
)

# Global engine instance and dashboard path
engine: CreditDecisionEngine = None
DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading Credit Decision Engine and ML Artifacts...")
    try:
        engine = CreditDecisionEngine()
        print("Model bundle and TreeSHAP explainer successfully loaded into memory.")
    except Exception as e:
        print(f"ERROR loading model bundle: {e}")
        raise e
    yield
    print("Shutting down Credit Decision Engine...")


app = FastAPI(
    title="FinTech Credit Risk & Decisioning API",
    description=(
        "Production AI microservice for credit underwriting, FICO score mapping, "
        "and automated adverse action generation with SHAP explainability."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def root():
    """Serves the interactive web UI dashboard."""
    if DASHBOARD_PATH.exists():
        return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(
        content="<h2>Credit Scoring Microservice Online</h2><p>Visit <a href='/docs'>/docs</a></p>",
        status_code=200
    )


@app.get("/health", tags=["Monitoring"])
async def health_check():
    if engine is None or engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credit Decision Engine is not initialized."
        )
    return {
        "status": "healthy",
        "engine_ready": True,
        "features_loaded": len(engine.feature_columns)
    }


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Model Governance"])
async def get_model_info():
    """Returns governance metadata, metrics, and algorithms."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded.")
    
    return ModelInfoResponse(
        model_name="Consumer Credit Default Risk Model",
        champion_algorithm="LightGBM Gradient Boosted Decision Trees",
        baseline_algorithm="Regularized Logistic Regression (Balanced)",
        metrics=engine.metrics,
        feature_count=len(engine.feature_columns),
        status="ACTIVE_PRODUCTION"
    )


@app.post("/api/v1/score", response_model=ScoringResponse, tags=["Inference"])
async def score_applicant(applicant: ApplicantProfile):
    """
    Evaluates a loan/credit applicant in real time.
    Returns:
    - FICO-equivalent credit score (300 - 850)
    - Probability of Default (PD)
    - Underwriting Decision (Approved, Review, Rejected)
    - Top adverse action reasons via TreeSHAP
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    
    try:
        applicant_dict = applicant.model_dump()
        result = engine.score_applicant(applicant_dict)
        return ScoringResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/api/v1/batch-score", response_model=BatchScoringResponse, tags=["Inference"])
async def batch_score(batch: BatchApplicantsRequest):
    """Processes multiple applicant profiles in a single request."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    
    results = []
    for applicant in batch.applicants:
        applicant_dict = applicant.model_dump()
        score_res = engine.score_applicant(applicant_dict)
        results.append(ScoringResponse(**score_res))
        
    return BatchScoringResponse(
        total_processed=len(results),
        results=results
    )
