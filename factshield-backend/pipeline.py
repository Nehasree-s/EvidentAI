from parser import extract_claims
from retriever import VectorRetriever
from verifier import verify_claim

def run_factshield_pipeline(source_document: str, llm_output: str) -> dict:
    """
    Orchestrates the entire FactShield verification flow:
    1. Parse LLM output into claims (spaCy)
    2. Index source document into FAISS vector store
    3. Retrieve relevant context for each claim
    4. Run DeBERTa-v3 NLI verification
    5. Calculate Explainable Trust Score
    """
    # 1. Parse LLM response into claims
    claims = extract_claims(llm_output)
    if not claims:
        return {
            "trust_score": 0.0,
            "total_claims": 0,
            "verdict_counts": {"entailment": 0, "neutral": 0, "contradiction": 0},
            "results": []
        }

    # 2. Chunk source document and populate FAISS
    # We split source document into sentences to serve as context chunks
    source_chunks = extract_claims(source_document)
    
    # Fallback if source document is a single block
    if not source_chunks and source_document.strip():
        source_chunks = [source_document.strip()]

    retriever = VectorRetriever()
    retriever.index_document(source_chunks)

    # 3. Process each claim
    verification_results = []
    verdict_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}

    for claim in claims:
        # Retrieve top-1 relevant chunk from FAISS
        retrieved_context = retriever.retrieve_context(claim, top_k=1)
        evidence = retrieved_context[0] if retrieved_context else source_document

        # Run NLI check
        nli_response = verify_claim(premise=evidence, hypothesis=claim)
        verdict = nli_response["verdict"]
        
        # Track counts
        verdict_counts[verdict] += 1

        verification_results.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": nli_response["probabilities"][verdict],
            "evidence": evidence,
            "all_probabilities": nli_response["probabilities"]
        })

    # 4. Calculate Explainable Trust Score
    total_claims = len(claims)
    weighted_sum = (
        (1.0 * verdict_counts["entailment"]) + 
        (0.5 * verdict_counts["neutral"]) + 
        (0.0 * verdict_counts["contradiction"])
    )
    trust_score = round((weighted_sum / total_claims) * 100, 2)

    return {
        "trust_score": trust_score,
        "total_claims": total_claims,
        "verdict_counts": verdict_counts,
        "results": verification_results
    }

# Test execution block
if __name__ == "__main__":
    sample_source = (
        "Diabetes mellitus is a metabolic disease characterized by high blood sugar levels over a prolonged period. "
        "Symptoms often include frequent urination, increased thirst, and increased appetite. "
        "Type 1 diabetes must be managed with insulin injections. "
        "Healthy diet, physical exercise, and maintaining a normal body weight can prevent or delay Type 2 diabetes."
    )

    sample_llm_response = (
        "Diabetes is a chronic metabolic disease with high blood glucose. "
        "Aspirin can cure Type 1 diabetes within two weeks without insulin. "
        "Regular exercise and a healthy diet help prevent Type 2 diabetes."
    )

    print("\n--- RUNNING FACTSHIELD PIPELINE ---")
    output = run_factshield_pipeline(source_document=sample_source, llm_output=sample_llm_response)
    
    print(f"\n📊 TRUST SCORE: {output['trust_score']}%")
    print(f"Total Claims Analyzed: {output['total_claims']}")
    print(f"Summary Counts: {output['verdict_counts']}\n")
    
    print("--- INDIVIDUAL CLAIM RESULTS ---")
    for idx, item in enumerate(output["results"], 1):
        symbol = "🟢" if item["verdict"] == "entailment" else ("🔴" if item["verdict"] == "contradiction" else "🟡")
        print(f"\n{symbol} Claim {idx}: {item['claim']}")
        print(f"   Verdict: {item['verdict'].upper()} (Confidence: {item['confidence']*100:.1f}%)")
        print(f"   Retrieved Evidence: \"{item['evidence']}\"")