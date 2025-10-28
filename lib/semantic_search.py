from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
        )

    def generate_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        if len(text) == 0:
            raise ValueError("text is empty")

        if any(ch.isspace() for ch in text):
            raise ValueError("text contains whitespaces")

        return self.model.encode(
            [text],
        )[0]


def embed_txt(text: str) -> None:
    semantic_search = SemanticSearch()

    embedding: Optional[np.ndarray] = None
    try:
        embedding = semantic_search.generate_embedding(
            text,
        )
    except ValueError as e:
        print("Error:", e)
        return

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def validate_model() -> None:
    MODEL = SemanticSearch().model

    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MODEL.max_seq_length}")
