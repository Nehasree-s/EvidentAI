import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from typing import Optional, Tuple


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

LABELS = [
    "contradiction",
    "entailment",
    "neutral",
]


# ============================================================
# LAZY MODEL STORAGE
# ============================================================

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None


# ============================================================
# MODEL LOADER
# ============================================================

def get_model() -> Tuple[
    AutoTokenizer,
    AutoModelForSequenceClassification,
]:
    """
    Lazily load the DeBERTa-v3 NLI model.

    The model is loaded only when verification is first requested,
    rather than when FastAPI imports this module.
    """

    global _tokenizer
    global _model

    if _tokenizer is None or _model is None:

        print(f"Loading NLI model: {MODEL_NAME}")

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        # Force CPU execution for server environments.
        _model.to("cpu")

        # Disable training behavior.
        _model.eval()

        print("NLI model loaded successfully.")

    return _tokenizer, _model


# ============================================================
# CLAIM VERIFICATION
# ============================================================

def verify_claim(
    premise: str,
    hypothesis: str,
) -> dict:
    """
    Compare retrieved evidence (premise) against an extracted
    claim (hypothesis) using Natural Language Inference.

    Returns:
        {
            "verdict": "entailment" | "neutral" | "contradiction",

            "probabilities": {
                "contradiction": float,
                "entailment": float,
                "neutral": float
            }
        }
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    premise = (premise or "").strip()
    hypothesis = (hypothesis or "").strip()

    if not premise:
        raise ValueError(
            "Premise cannot be empty."
        )

    if not hypothesis:
        raise ValueError(
            "Hypothesis cannot be empty."
        )


    # --------------------------------------------------------
    # Lazy-load model
    # --------------------------------------------------------

    tokenizer, model = get_model()


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )


    # Explicitly keep tensors on CPU.
    inputs = {
        key: value.to("cpu")
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # NLI inference
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities_tensor = torch.softmax(
            logits,
            dim=-1,
        )[0]


    # --------------------------------------------------------
    # Determine predicted label
    # --------------------------------------------------------

    predicted_idx = int(
        torch.argmax(
            probabilities_tensor
        ).item()
    )

    if predicted_idx >= len(LABELS):
        raise RuntimeError(
            f"Unexpected model label index: {predicted_idx}"
        )

    verdict = LABELS[predicted_idx]


    # --------------------------------------------------------
    # Convert probabilities
    # --------------------------------------------------------

    probabilities = (
        probabilities_tensor
        .detach()
        .cpu()
        .tolist()
    )


    result = {
        "verdict": verdict,

        "probabilities": {
            "contradiction": round(
                float(probabilities[0]),
                4,
            ),

            "entailment": round(
                float(probabilities[1]),
                4,
            ),

            "neutral": round(
                float(probabilities[2]),
                4,
            ),
        },
    }


    return result