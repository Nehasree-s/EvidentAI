import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load lightweight 80MB embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class VectorRetriever:
    """
    In-memory FAISS Vector Database for Retrieval-Augmented Generation (RAG).
    Converts document chunks into normalized 384-d vectors for Cosine Similarity search.
    """
    def __init__(self):
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = Cosine similarity when normalized
        self.chunks = []

    def index_document(self, text_chunks: list[str]):
        """
        Encodes document text chunks into vector embeddings and adds them to FAISS.
        """
        self.index.reset()  # Clear existing index
        self.chunks = text_chunks

        if not text_chunks:
            # Nothing to index; leave the FAISS index empty rather than
            # calling embedder.encode([]) which errors on some backends.
            return

        embeddings = embedder.encode(text_chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve_context(self, query_claim: str, top_k: int = 1) -> list[str]:
        """
        Given a single claim, retrieves the top_k most semantically relevant context chunks.
        """
        if not self.chunks or self.index.ntotal == 0:
            return []

        query_vector = embedder.encode([query_claim], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(np.array(query_vector).astype("float32"), top_k)

        retrieved_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
                
        return retrieved_chunks

# Test block
if __name__ == "__main__":
    sample_knowledge_base = [
        "Diabetes mellitus refers to a group of diseases that affect how your body uses blood sugar.",
        "Aspirin is a nonsteroidal anti-inflammatory drug used to reduce pain, fever, or inflammation.",
        "Dietary changes and physical activity help manage glucose levels in diabetic patients."
    ]

    retriever = VectorRetriever()
    retriever.index_document(sample_knowledge_base)

    query = "Does physical exercise help control blood sugar?"
    matches = retriever.retrieve_context(query, top_k=1)

    print("\n--- TESTING RETRIEVER ---")
    print(f"Query Claim: '{query}'")
    if matches:
        print(f"Top Retrieved Context: '{matches[0]}'")
    else:
        print("No matches retrieved.")