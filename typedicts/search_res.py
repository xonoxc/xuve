from dataclasses import dataclass
from typedicts.movies import Movie


@dataclass
class HybridScores:
    movie: Movie
    kw_score: float = 0.0
    sem_score: float = 0.0


@dataclass
class WeightedSearchResult:
    id: int
    movie: Movie
    hybrid_score: float
    semantic_score: float
    keyword_score: float
