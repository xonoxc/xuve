from typing import List, Tuple

from lib.chunked_semantic_search import ChunkedSemanticSearch
from typedicts.movies import Movie

from .keyword_search import InvertedIndex


class HybridSearch:
    def __init__(self, documents):
        self.documents: List[Movie] = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_chunk_embeddings()

        self.inverted_idx = InvertedIndex()

    def bm25_search(
        self,
        query: str,
        limit: int,
    ) -> List[Tuple[int, str, float]]:
        self.inverted_idx.load()
        return self.inverted_idx.bm25_search(
            query,
            limit,
        )

    def weighted_search(
        self,
        query: str,
        alpha,
        limit: int = 5,
    ) -> None:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(
        self,
        query: str,
        k: int = 5,
        limit: int = 10,
    ):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


# this function normalizes the smeantic and bm25 scores
def normalize_scores(scores: List[float]) -> List[float]:
    max_score, min_score = (
        max(scores),
        min(
            scores,
        ),
    )
    diff_range = max_score - min_score

    return [(s - min_score) / diff_range for s in scores]
