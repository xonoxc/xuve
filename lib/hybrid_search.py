from typing import Dict, List, Optional, Tuple

from config.data import ALPHA
from lib.chunked_semantic_search import ChunkedSemanticSearch
from typedicts.movies import Movie
from typedicts.search_res import HybridScores, WeightedSearchResult


from lib.data_loaders import load_movie_data
from .keyword_search import InvertedIndex


# i know this is unrelated to the code but
# once a wise man said "The fastest way to loop in python is to not loop in python"
# -- Me , 2025
class HybridSearch:
    def __init__(self):
        self.documents: List[Movie] = load_movie_data()
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

    # search based on weighted average of bm25 and semantic search scores
    def weighted_search(
        self,
        query: str,
        alpha,
        limit: int = 5,
    ) -> List[WeightedSearchResult]:
        keyword_search_results = self.bm25_search(
            query,
            500 * limit,
        )

        self.semantic_search.load_or_create_embeddings(self.documents)
        semantic_search_results = self.semantic_search.search(
            query,
            500 * limit,
        )

        nm_kw_score = normalize_scores([s[-1] for s in keyword_search_results])
        nm_semantic_score = normalize_scores(
            [e["score"] for e in semantic_search_results]
        )

        scores_map_dict: Dict[int, HybridScores] = dict()

        for (doc_id, _, _), nscore in zip(keyword_search_results, nm_kw_score):
            scores_map_dict[doc_id] = HybridScores(
                movie=self.documents[doc_id],
                kw_score=nscore,
            )

        for res, score in zip(semantic_search_results, nm_semantic_score):
            doc_id = res["id"]
            if doc_id in scores_map_dict:
                scores_map_dict[doc_id].sem_score = score
            else:
                scores_map_dict[doc_id] = HybridScores(
                    movie=self.documents[doc_id],
                    sem_score=score,
                )

        return sorted(
            (
                WeightedSearchResult(
                    id=doc_id,
                    movie=hs.movie,
                    hybrid_score=hybrid_score(
                        bm25_score=hs.kw_score,
                        semantic_score=hs.sem_score,
                        alpha=alpha,
                    ),
                    keyword_score=hs.kw_score,
                    semantic_score=hs.sem_score,
                )
                for doc_id, hs in scores_map_dict.items()
            ),
            key=lambda w: w.hybrid_score,
            reverse=True,
        )[:limit]

    # placeholder for RRF hybrid search
    def rrf_search(
        self,
        query: str,
        k: int = 5,
        limit: int = 10,
    ):
        raise NotImplementedError(
            "RRF hybrid search is not implemented yet.",
        )


# this function normalizes the smeantic and bm25 scores
def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []

    max_score, min_score = (
        max(scores),
        min(
            scores,
        ),
    )
    diff_range = max_score - min_score

    if diff_range == 0:
        return [0.0 for _ in scores]

    return [(s - min_score) / diff_range for s in scores]


# function to calculate hybrid search score
def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: Optional[float] = None,
) -> float:
    if alpha is None:
        alpha = ALPHA

    return alpha * bm25_score + (1 - alpha) * semantic_score


def exec_weighted_search(
    query: str,
    alpha,
    limit: int = 5,
) -> List[WeightedSearchResult]:
    # load movie data
    hybrid_search_instance = HybridSearch()

    return hybrid_search_instance.weighted_search(
        query,
        alpha,
        limit,
    )
