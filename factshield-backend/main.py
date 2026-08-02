from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import our orchestrator pipeline built on Day 4
from pipeline import run_factshield_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="FactShield API Gateway",
    description="Enterprise Autonomous RAG & Hallucination Verification Gateway",
    version="1.0.0"
)

# Enable CORS for frontend connectivity (React / Vite)
# NOTE: allow_origins=["*"] combined with allow_credentials=True is invalid per
# the CORS spec — browsers will reject it. Since this API doesn't use cookies/auth
# headers that need credentialed requests, we drop allow_credentials. If you later
# need credentialed requests, replace "*" with an explicit list of origins instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend during local development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC SCHEMAS ---

class VerificationRequest(BaseModel):
    source_document: str = Field(
        ..., 
        description="Authoritative reference source document or knowledge base context.",
        json_schema_extra={"example": "Diabetes mellitus is a metabolic disease characterized by high blood sugar levels."}
    )
    llm_output: str = Field(
        ..., 
        description="The LLM-generated text or response to be fact-checked.",
        json_schema_extra={"example": "Aspirin completely cures Type 1 diabetes within two weeks."}
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

# --- API ENDPOINTS ---

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "online", "service": "FactShield Gateway"}

@app.post("/verify", response_model=VerificationResponse, tags=["Verification"])
def verify_claims(payload: VerificationRequest):
    """
    Main verification endpoint: Accepts source document and generated response,
    runs FAISS retrieval + spaCy claim parsing + DeBERTa-v3 NLI, and returns
    trust metrics and claim-level verdicts.
    """
    if not payload.source_document.strip():
        raise HTTPException(status_code=400, detail="Source document cannot be empty.")
    if not payload.llm_output.strip():
        raise HTTPException(status_code=400, detail="LLM output text cannot be empty.")

    start_time = time.time()
    
    try:
        pipeline_output = run_factshield_pipeline(
            source_document=payload.source_document,
            llm_output=payload.llm_output
        )
        
        latency = round(time.time() - start_time, 3)
        pipeline_output["latency_seconds"] = latency
        
        return pipeline_output
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

# Test server launch via script execution
if __name__ == "__main__":
    import uvicorn
    print("Starting FactShield FastAPI Server on http://127.0.0.1:8000...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)