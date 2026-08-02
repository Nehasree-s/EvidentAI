from parser import extract_claims
from retriever import VectorRetriever
from verifier import verify_claim
from typing import Dict, Any


VALID_VERDICTS = {
    "entailment",
    "neutral",
    "contradiction",
}


def run_factshield_pipeline(
    source_document: str,
    llm_output: str,
) -> Dict[str, Any]:
    """
    Run the complete FactShield verification pipeline.

    Pipeline:
        1. Extract claims from LLM output
        2. Split source document into chunks
        3. Build vector index
        4. Retrieve relevant evidence
        5. Verify claims using NLI
        6. Calculate trust score
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    source_document = (source_document or "").strip()
    llm_output = (llm_output or "").strip()

    if not source_document:
        raise ValueError("Source document cannot be empty.")

    if not llm_output:
        raise ValueError("LLM output cannot be empty.")


    # ========================================================
    # 1. EXTRACT CLAIMS
    # ========================================================

    claims = extract_claims(llm_output)

    if not claims:
        return {
            "trust_score": 0.0,
            "total_claims": 0,
            "verdict_counts": {
                "entailment": 0,
                "neutral": 0,
                "contradiction": 0,
            },
            "results": [],
        }


    # ========================================================
    # 2. PREPARE SOURCE DOCUMENT
    # ========================================================

    source_chunks = extract_claims(source_document)

    # Fallback for documents that could not be split.
    if not source_chunks:
        source_chunks = [source_document]


    # ========================================================
    # 3. INITIALIZE VECTOR RETRIEVER
    # ========================================================

    retriever = VectorRetriever()

    retriever.index_document(source_chunks)


    # ========================================================
    # 4. VERIFY CLAIMS
    # ========================================================

    verification_results = []

    verdict_counts = {
        "entailment": 0,
        "neutral": 0,
        "contradiction": 0,
    }

    for claim in claims:

        # ----------------------------------------------------
        # Retrieve evidence
        # ----------------------------------------------------

        retrieved_context = retriever.retrieve_context(
            claim,
            top_k=1,
        )

        if retrieved_context:
            evidence = retrieved_context[0]
        else:
            # Safe fallback
            evidence = source_document


        # ----------------------------------------------------
        # Run NLI verification
        # ----------------------------------------------------

        nli_response = verify_claim(
            premise=evidence,
            hypothesis=claim,
        )


        # ----------------------------------------------------
        # Validate verifier response
        # ----------------------------------------------------

        if not isinstance(nli_response, dict):
            raise RuntimeError(
                "Verifier returned an invalid response."
            )

        verdict = str(
            nli_response.get("verdict", "neutral")
        ).lower()

        probabilities = nli_response.get(
            "probabilities",
            {},
        )

        # Protect the pipeline from unexpected model labels.
        if verdict not in VALID_VERDICTS:
            verdict = "neutral"


        # ----------------------------------------------------
        # Normalize probability values
        # ----------------------------------------------------

        normalized_probabilities = {
            "entailment": float(
                probabilities.get("entailment", 0.0)
            ),
            "neutral": float(
                probabilities.get("neutral", 0.0)
            ),
            "contradiction": float(
                probabilities.get("contradiction", 0.0)
            ),
        }


        # ----------------------------------------------------
        # Update statistics
        # ----------------------------------------------------

        verdict_counts[verdict] += 1

        confidence = normalized_probabilities.get(
            verdict,
            0.0,
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        verification_results.append(
            {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "evidence": evidence,
                "all_probabilities": normalized_probabilities,
            }
        )


    # ========================================================
    # 5. CALCULATE TRUST SCORE
    # ========================================================

    total_claims = len(claims)

    weighted_sum = (
        verdict_counts["entailment"] * 1.0
        + verdict_counts["neutral"] * 0.5
        + verdict_counts["contradiction"] * 0.0
    )

    trust_score = round(
        (weighted_sum / total_claims) * 100,
        2,
    )


    # ========================================================
    # 6. RETURN FINAL RESPONSE
    # ========================================================

    return {
        "trust_score": trust_score,
        "total_claims": total_claims,
        "verdict_counts": verdict_counts,
        "results": verification_results,
    }