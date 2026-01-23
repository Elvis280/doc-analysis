from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class LocalEmbedding(Embeddings):
    def __init__(self):
        print("Loading Nomic Embed Text Model...")
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True
        )
        print("Model loaded.")

    def embed_query(self, text: str) -> List[float]:
        # Nomic specific prefix for queries
        return self.model.encode(f"search_query: {text}").tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Nomic specific prefix for documents
        return [self.model.encode(f"search_document: {t}").tolist() for t in texts]

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Keep callable for backward compatibility if needed
        return self.embed_documents(input)

# Global singleton? prefer creating instances where needed or a singleton here.
# For now, we can just export the class.
