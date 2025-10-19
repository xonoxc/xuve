from enum import Enum, auto


# represents status of the cache
class CacheStatus(Enum):
    NOT_BUILT = auto()
    CORRUPT = auto()
    BUILT = auto()
