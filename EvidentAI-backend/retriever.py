import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional


# ============================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """
    Lazily load the SentenceTransformer model.

    Lazy loading prevents the embedding model from being loaded
    unnecessarily when the module is imported and is generally
    friendlier to serverless environments.
    """

    global _embedder

    if _embedder is None:
        print(f"Loading embedding model: {MODEL_NAME}")

        _embedder = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )

        print("Embedding model loaded successfully.")

    return _embedder


# ============================================================
# VECTOR RETRIEVER
# ============================================================

class VectorRetriever:
    """
    In-memory FAISS retriever used by the FactShield RAG pipeline.

    Document chunks are converted into normalized 384-dimensional
    embeddings using all-MiniLM-L6-v2.

    FAISS IndexFlatIP is then used for similarity search.

    Because vectors are normalized, inner-product similarity
    corresponds to cosine similarity.
    """

    def __init__(self):

        self.dimension = EMBEDDING_DIMENSION

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.chunks: List[str] = []


    # ========================================================
    # DOCUMENT INDEXING
    # ========================================================

    def index_document(
        self,
        text_chunks: List[str],
    ) -> None:
        """
        Encode document chunks and populate the FAISS index.
        """

        # Clear previous vectors.
        self.index.reset()
        self.chunks = []

        if not text_chunks:
            return


        # ----------------------------------------------------
        # Clean input
        # ----------------------------------------------------

        cleaned_chunks = [
            chunk.strip()
            for chunk in text_chunks
            if isinstance(chunk, str)
            and chunk.strip()
        ]

        if not cleaned_chunks:
            return

        self.chunks = cleaned_chunks


        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        embedder = get_embedder()


        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embeddings = embedder.encode(
            cleaned_chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


        # ----------------------------------------------------
        # Ensure FAISS-compatible format
        # ----------------------------------------------------

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )


        if embeddings.ndim != 2:
            raise RuntimeError(
                "Embedding model returned an invalid shape."
            )

        if embeddings.shape[1] != self.dimension:
            raise RuntimeError(
                f"Expected embedding dimension "
                f"{self.dimension}, but received "
                f"{embeddings.shape[1]}."
            )


        # ----------------------------------------------------
        # Add embeddings to FAISS
        # ----------------------------------------------------

        self.index.add(embeddings)


    # ========================================================
    # CONTEXT RETRIEVAL
    # ========================================================

    def retrieve_context(
        self,
        query_claim: str,
        top_k: int = 1,
    ) -> List[str]:
        """
        Retrieve the most semantically relevant source chunks
        for a claim.
        """

        if not query_claim:
            return []

        query_claim = query_claim.strip()

        if not query_claim:
            return []


        # ----------------------------------------------------
        # Ensure index contains vectors
        # ----------------------------------------------------

        if (
            not self.chunks
            or self.index.ntotal == 0
        ):
            return []


        # Prevent requesting more results than available.
        top_k = max(
            1,
            min(
                top_k,
                len(self.chunks),
            ),
        )


        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        embedder = get_embedder()


        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------

        query_vector = embedder.encode(
            [query_claim],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # FAISS similarity search
        # ----------------------------------------------------

        _, indices = self.index.search(
            query_vector,
            top_k,
        )


        # ----------------------------------------------------
        # Convert FAISS indices back into text
        # ----------------------------------------------------

        retrieved_chunks: List[str] = []

        for idx in indices[0]:

            # FAISS can return -1 when no valid match exists.
            if (
                idx >= 0
                and idx < len(self.chunks)
            ):
                retrieved_chunks.append(
                    self.chunks[idx]
                )


        return retrieved_chunks