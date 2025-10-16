from typing import TypedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Movie(TypedDict):
    id: int
    title: str
    description: str
