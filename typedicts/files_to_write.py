from typing import TypedDict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheFilesToWrite(TypedDict):
    path: str
    data: Any
