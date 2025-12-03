from typing import Dict, List, Optional, Tuple

from config.data import ALPHA
from lib.ai.action import enhance_query, generate_resp
from lib.ai.prompt_builders import (
    build_batch_doc_rating_prompt,
    build_individual_doc_rating_prompt,
)
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.enums.enahnce_methods import EnhanceMethod
from lib.enums.rerank_methods import RerankMethod
from typedicts.movies import Movie
from typedicts.rrf_search import DocumentRanks
from typedicts.search_res import (
    HybridScores,
    RRFSearchResult,
    WeightedSearchResult,
)


from lib.data_loaders import load_movie_data
from utils.parse_id import parse_id_list
from utils.safe_float import convert_to_float
from .keyword_search import InvertedIndex


# i know this is unrelated to the code but
# once a wise man said "The fastest way to loop in python is to not loop in python"
# -- Me , 2025
class HybridSearch:
    def __init__(self):
        movie_data = load_movie_data()

        self.documents_list: List[Movie] = movie_data
        self.semantic_search = ChunkedSemanticSearch()

        self.documents: Dict[int, Movie] = {m["id"]: m for m in movie_data}

        self.semantic_search.load_or_create_embeddings(
            movie_data,
        )

        self.inverted_idx = InvertedIndex()
        self.inverted_idx.load()

    def bm25_search(
        self,
        query: str,
        limit: int,
    ) -> List[Tuple[int, str, float]]:
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

        semantic_search_results = self.semantic_search.search(
            query,
            500 * limit,
        )

        nm_kw_score = normalize_scores([s[-1] for s in keyword_search_results])
        nm_semantic_score = normalize_scores([e.score for e in semantic_search_results])

        scores_map_dict: Dict[int, HybridScores] = dict()

        for (doc_id, _, _), nscore in zip(keyword_search_results, nm_kw_score):
            scores_map_dict[doc_id] = HybridScores(
                movie=self.documents[doc_id],
                kw_score=nscore,
            )

        for res, score in zip(semantic_search_results, nm_semantic_score):
            doc_id = res.id
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
    ) -> List[RRFSearchResult]:
        kw_search_res = self.bm25_search(
            query,
            500 * limit,
        )

        sem_search_res = self.semantic_search.search(
            query,
            500 * limit,
        )

        score_map: Dict[int, DocumentRanks] = dict()

        for rank, (doc_id, _, _) in enumerate(kw_search_res, start=1):
            if doc_id not in score_map:
                score_map[doc_id] = DocumentRanks(
                    document=self.documents[doc_id],
                    semantic_rank=0,
                    keyword_rank=rank,
                )

            else:
                score_map[doc_id].keyword_rank = rank

        for rank, res in enumerate(sem_search_res, start=1):
            doc_id = res.id
            if doc_id not in score_map:
                score_map[doc_id] = DocumentRanks(
                    document=self.documents[doc_id],
                    keyword_rank=0,
                    semantic_rank=rank,
                )
            else:
                score_map[doc_id].semantic_rank = rank

        sorted_fused_scores = self.calc_fused_ranks(
            score_map,
            k,
        )

        return [
            RRFSearchResult(
                id=doc_id,
                movie=self.documents[doc_id],
                rrf_score=fused,
                keyword_rank=score_map[doc_id].keyword_rank,
                semantic_rank=score_map[doc_id].semantic_rank,
            )
            for doc_id, fused in sorted_fused_scores[:limit]
        ]

    # helper function to claculate fused ranks
    # and returned the reverse sorted list
    def calc_fused_ranks(
        self, score_mapping: Dict[int, DocumentRanks], k: int
    ) -> List[Tuple[int, float]]:
        fused_scores = []
        for doc_id, ranks in score_mapping.items():
            fused = 0.0

            if ranks.keyword_rank > 0:
                fused += rrf_score(ranks.keyword_rank, k)

            if ranks.semantic_rank > 0:
                fused += rrf_score(ranks.semantic_rank, k)

            fused_scores.append((doc_id, fused))

        return sorted(fused_scores, key=lambda x: x[1], reverse=True)


# function to calcualte rrf score
def rrf_score(rank: float, k=60) -> float:
    return 1 / (k + rank)


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


def exec_rrf_search(
    query: str,
    limit: int = 5,
    k: int = 60,
    enahnce_method: EnhanceMethod | None = None,
    rerank_method: RerankMethod | None = None,
) -> List[RRFSearchResult]:
    hybrid_search_instance = HybridSearch()

    enahanced_query = enhance_query(
        query,
        method=enahnce_method,
    )

    search_limit = 5 * limit if rerank_method else limit

    search_res = hybrid_search_instance.rrf_search(
        enahanced_query,
        k,
        limit=search_limit,
    )

    if rerank_method is None:
        return search_res

    return rerank_results(query, results=search_res, method=rerank_method)[:limit]


def rerank_results(
    query: str,
    results: List[RRFSearchResult],
    method: RerankMethod,
) -> List[RRFSearchResult]:
    stripped_query = query.strip()

    result_map = {doc.id: doc for doc in results}

    reranked_items = []

    match RerankMethod(method):
        case RerankMethod.INDIVIDUAL:
            for doc in results:
                resp = generate_resp(
                    build_individual_doc_rating_prompt(query=stripped_query, doc=doc),
                    delay=4,
                )

                reranked_items.append(
                    (
                        doc.id,
                        convert_to_float(
                            value=resp.text.strip() if (resp and resp.text) else "",
                            fallback=0.0,
                        ),
                    )
                )

            reranked_items.sort(key=lambda x: x[0], reverse=True)

        case RerankMethod.BATCH:
            resp = generate_resp(
                build_batch_doc_rating_prompt(query=stripped_query, docs=results),
                delay=4,
            )

            print("list returned by llm", resp.text)
            id_list = parse_id_list(
                resp.text.strip() if (resp and resp.text) else "",
            )

            reranked_items = [(doc_id, None) for doc_id in id_list]

    return [result_map.get(doc_id) for doc_id, _ in reranked_items]


def exec_weighted_search(
    query: str,
    alpha,
    limit: int = 5,
) -> List[WeightedSearchResult]:
    hybrid_search_instance = HybridSearch()

    return hybrid_search_instance.weighted_search(
        query,
        alpha,
        limit,
    )
