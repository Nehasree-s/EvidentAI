import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

print("Loading DeBERTa-v3 NLI model (this downloads ~700MB on first run)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set model to evaluation mode

# Label mapping according to DeBERTa-v3 NLI specification
LABELS = ["contradiction", "entailment", "neutral"]

def verify_claim(premise: str, hypothesis: str) -> dict:
    """
    Compares a reference premise against a single claim hypothesis.
    Returns the predicted verdict and softmax probabilities for all three labels.
    """
    # Tokenize input pair (Premise, Hypothesis)
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0].tolist()

    predicted_idx = int(torch.argmax(logits, dim=1).item())
    
    return {
        "verdict": LABELS[predicted_idx],
        "probabilities": {
            "contradiction": round(probabilities[0], 4),
            "entailment": round(probabilities[1], 4),
            "neutral": round(probabilities[2], 4)
        }
    }

# Test block
if __name__ == "__main__":
    reference = "Diabetes is a chronic metabolic disease characterized by elevated levels of blood glucose. Treatment requires insulin or oral medication."
    
    claim_1 = "Diabetes is a metabolic condition involving high blood sugar."
    claim_2 = "Aspirin completely cures diabetes within two weeks."
    
    print("\n--- TESTING VERIFIER ---")
    print(f"Claim 1 Verdict: {verify_claim(reference, claim_1)}")
    print(f"Claim 2 Verdict: {verify_claim(reference, claim_2)}")