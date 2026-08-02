import spacy
from typing import List


# ============================================================
# SPACY INITIALIZATION
# ============================================================

def create_nlp():
    """
    Create a lightweight spaCy pipeline for sentence segmentation.

    FactShield currently uses spaCy only to split LLM-generated
    text into individual claims/sentences. Therefore, loading the
    full en_core_web_sm model is unnecessary.

    Using a blank English pipeline with the sentencizer:
    - avoids downloading en_core_web_sm
    - reduces deployment size
    - reduces cold-start time
    - works better in serverless environments such as Vercel
    """

    nlp = spacy.blank("en")

    # Add sentence boundary detection
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    return nlp


# Initialize once when the serverless instance starts.
nlp = create_nlp()


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(text: str) -> List[str]:
    """
    Split unstructured LLM-generated text into individual
    claim sentences.

    Args:
        text:
            LLM-generated response or paragraph.

    Returns:
        List of extracted claim strings.
    """

    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    try:

        # Process text through spaCy
        doc = nlp(text)

        claims = []

        for sentence in doc.sents:

            claim = sentence.text.strip()

            # Ignore extremely short fragments
            if len(claim) > 3:
                claims.append(claim)

        return claims

    except Exception as exc:

        print(f"Claim extraction error: {exc}")

        # Returning an empty list allows the pipeline to
        # handle the failure instead of crashing the API.
        return []