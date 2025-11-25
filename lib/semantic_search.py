import re
from typing import Dict, List, Optional, Tuple, Any
from numpy._typing import ArrayLike
from sentence_transformers import SentenceTransformer
import numpy as np

from config.data import MOVIE_EMBDEDDINGS_PATH
from lib.data_loaders import load_movie_data
from typedicts.movies import Movie


class SemanticSearch:
    def __init__(self, model_name: Optional[str] = None) -> None:
        # model that is used for embedding
        self.model = SentenceTransformer(
            model_name or "all-MiniLM-L6-v2",
        )

        # n- dimensional array of embeddings
        self.embeddings: np.ndarray = np.empty((0,))
        # list of documents
        self.documents: List[Movie] = []
        # map for doucment_id -> Movie
        self.doc_map: Dict[int, Movie] = {}

    # this one is actual search function now
    def search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        if not self.embeddings.size > 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        smilarity_res: List[Tuple[float, Movie]] = []
        for embedding, document in zip(self.embeddings, self.documents):
            similiarity_score = cosine_similarity(query_embedding, embedding)
            smilarity_res.append(
                (
                    similiarity_score,
                    document,
                )
            )

        sorted_res = sorted(smilarity_res, key=lambda mem: mem[0], reverse=True)[:limit]

        return [
            {
                "id": doc["id"],
                "score": round(score, 4),
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in sorted_res
        ]

    # this function does not check if
    # the cache dir for MOVIE_EMBDEDDINGS_PATH exists
    # because that's the starting point of the application
    # if cache is missing files the main function will exit
    # therefore the flow won't reach the function in the fist place
    def load_or_create_embeddings(
        self,
        documents: List[Movie],
    ) -> np.ndarray:
        if len(self.embeddings) == len(documents):
            return self.embeddings

        self.documents = documents

        for document in documents:
            self.doc_map[int(document["id"])] = document

        self.embeddings = np.load(MOVIE_EMBDEDDINGS_PATH)
        return self.embeddings

    def build_embeddings(
        self,
        documents: List[Movie],
    ) -> np.ndarray:
        self.documents = documents

        content_list = []
        for document in documents:
            self.doc_map[int(document["id"])] = document
            content_list.append(
                f"{document['title']}:{document['description']}",
            )

        self.embeddings = self.model.encode(content_list, show_progress_bar=True)
        return self.embeddings

    def save_embeddings(self) -> None:
        np.save(
            MOVIE_EMBDEDDINGS_PATH,
            self.embeddings,
        )

    def generate_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        text = text.strip()

        if not text:
            print("Warning: Empty text provided for embedding.")
            return np.array([])

        return self.model.encode(
            [text],
        )[0]


def semantic_chunk(
    text: str,
    max_chunk_size: int = 4,
    overlap: int = 0,
) -> List[str]:
    # split the sentences from the text and
    # remove the empty spaces
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    if not sentences:
        return []

    step = max_chunk_size - overlap

    return [
        " ".join(
            sentences[i : i + max_chunk_size],
        )
        for i in range(0, len(sentences), step)
    ]


# naive chunking with overlaping
# to maintain the context
def chunk(
    text: str,
    chunk_size: int = 200,
    overlap_size: int = 50,
) -> List[str]:
    words = text.split()
    if not words:
        return []

    if overlap_size >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap_size

    return [
        " ".join(words[i : i + chunk_size])
        for i in range(
            0,
            len(words),
            step,
        )
        if words[i : i + chunk_size]
    ]


def semantic_search(query: str, limit: int) -> List[Dict[str, Any]]:
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(
        load_movie_data(),
    )

    search_results = []
    try:
        search_results = semantic_search.search(
            query,
            limit,
        )
    except ValueError as e:
        print("Error:", e)

    print("search_results", search_results)
    return search_results


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()

    embedding: Optional[np.ndarray] = None
    embedding = semantic_search.generate_embedding(
        query,
    )

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike) -> float:
    dot_prod = np.dot(vec1, vec2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_prod / (norm1 * norm2)


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()

    documents = load_movie_data()
    embeddings = semantic_search.load_or_create_embeddings(
        documents,
    )

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_txt(text: str) -> None:
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(
        text,
    )

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def validate_model() -> None:
    MODEL = SemanticSearch().model

    print(f"Model loaded: {MODEL}")
    print(f"Max sequence length: {MODEL.max_seq_length}")
