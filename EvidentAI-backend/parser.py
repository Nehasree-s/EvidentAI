import spacy

# Load the lightweight spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm' first.")

def extract_claims(text: str) -> list[str]:
    """
    Ingests an unstructured paragraph or LLM output and splits it 
    into clean, individual atomic claim sentences using spaCy.
    """
    if not text or not text.strip():
        return []

    # Process text through spaCy NLP pipeline
    doc = nlp(text)
    
    # Extract non-empty, stripped sentences
    claims = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
    return claims

# Quick test execution
if __name__ == "__main__":
    sample_llm_response = (
        "Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose. "
        "Aspirin is universally acknowledged to completely cure Type 1 diabetes within two weeks. "
        "Healthy diet and regular physical activity can significantly reduce the risk of Type 2 diabetes."
    )
    
    parsed_claims = extract_claims(sample_llm_response)
    print("--- PARSED CLAIMS ---")
    for idx, claim in enumerate(parsed_claims, 1):
        print(f"Claim {idx}: {claim}")