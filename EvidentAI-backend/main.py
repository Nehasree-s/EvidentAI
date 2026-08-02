from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import time

from pipeline import run_factshield_pipeline


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FactShield API Gateway",
    description="Enterprise Autonomous RAG & Hallucination Verification Gateway",
    version="1.0.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

# "*" is convenient during deployment/testing because your
# Netlify URL may change while setting up the frontend.
#
# Once the final Netlify domain is known, you can replace "*"
# with that specific URL.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC REQUEST / RESPONSE MODELS
# ============================================================

class VerificationRequest(BaseModel):
    source_document: str = Field(
        ...,
        description=(
            "Authoritative reference source document "
            "or knowledge base context."
        ),
        json_schema_extra={
            "example": (
                "Diabetes mellitus is a metabolic disease "
                "characterized by high blood sugar levels."
            )
        },
    )

    llm_output: str = Field(
        ...,
        description="The LLM-generated text or response to be fact-checked.",
        json_schema_extra={
            "example": (
                "Aspirin completely cures Type 1 diabetes "
                "within two weeks."
            )
        },
    )


class ClaimResult(BaseModel):
    claim: str
    verdict: str
    confidence: float
    evidence: str
    all_probabilities: Dict[str, float]


class VerificationResponse(BaseModel):
    trust_score: float
    total_claims: int
    verdict_counts: Dict[str, int]
    latency_seconds: float
    results: List[ClaimResult]


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["System"])
def root():
    """
    Root endpoint used to verify that the API deployment
    is running successfully.
    """

    return {
        "status": "online",
        "service": "FactShield API Gateway",
        "version": "1.0.0",
        "message": "FactShield backend is running successfully.",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", tags=["System"])
def health_check():
    """
    Health-check endpoint for deployment monitoring.
    """

    return {
        "status": "online",
        "service": "FactShield Gateway",
    }


# ============================================================
# VERIFICATION ENDPOINT
# ============================================================

@app.post(
    "/verify",
    response_model=VerificationResponse,
    tags=["Verification"],
)
def verify_claims(payload: VerificationRequest):
    """
    Verify factual claims contained in an LLM-generated response
    against the supplied authoritative source document.

    Pipeline:
        Source Document
            ↓
        Claim Extraction
            ↓
        FAISS Retrieval
            ↓
        DeBERTa-v3 NLI Verification
            ↓
        Trust Score + Claim Verdicts
    """

    # --------------------------------------------------------
    # Validate source document
    # --------------------------------------------------------

    source_document = payload.source_document.strip()

    if not source_document:
        raise HTTPException(
            status_code=400,
            detail="Source document cannot be empty.",
        )

    # --------------------------------------------------------
    # Validate LLM output
    # --------------------------------------------------------

    llm_output = payload.llm_output.strip()

    if not llm_output:
        raise HTTPException(
            status_code=400,
            detail="LLM output text cannot be empty.",
        )

    # --------------------------------------------------------
    # Start latency measurement
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:

        # Run FactShield pipeline
        pipeline_output = run_factshield_pipeline(
            source_document=source_document,
            llm_output=llm_output,
        )

        # Calculate request latency
        latency = round(
            time.perf_counter() - start_time,
            3,
        )

        pipeline_output["latency_seconds"] = latency

        return pipeline_output

    except HTTPException:
        raise

    except Exception as exc:

        # Avoid exposing unnecessary internal implementation
        # details directly to the client.
        print(f"FactShield pipeline error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while running the verification pipeline.",
        ) from exc