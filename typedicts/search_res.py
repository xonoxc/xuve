from dataclasses import dataclass
from typedicts.movies import Movie


@dataclass
class HybridScores:
    movie: Movie
    kw_score: float = 0.0
    sem_score: float = 0.0


@dataclass
class BaseSearchResult:
    id: int
    movie: Movie


@dataclass
class WeightedSearchResult(BaseSearchResult):
    hybrid_score: float
    semantic_score: float
    keyword_score: float


@dataclass
class RRFSearchResult(BaseSearchResult):
    rrf_score: float
    semantic_rank: float
    keyword_rank: float


@dataclass
class SemanticSearchRes:
    id: int
    score: float
    title: str
    description: str


@dataclass
class ChunkMetadata:
    chunk_idx: int
    movie_idx: int
    total_chunks: int


@dataclass
class SemanticChunkSearchRes:
    id: int
    title: str
    document: str
    score: float
    metadata: ChunkMetadata


@dataclass
class RerankedRRFSearchResult:
    doc: Movie
    reranked_score: float
