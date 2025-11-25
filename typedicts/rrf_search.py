from dataclasses import dataclass

from typedicts.movies import Movie


@dataclass()
class DocumentRanks:
    document: Movie
    semantic_rank: float
    keyword_rank: float
