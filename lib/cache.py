import os

from config.data import (
    CACHE_DIR_PATH,
    EXPECTED_CACHE_DIR_FILES,
)
from lib.enums.cache_status import CacheStatus


class Cache:
    def __init__(self) -> None:
        self.cache_status: CacheStatus = self.check_cache_integrity()

    #  method to check if the cache is broken
    def check_cache_integrity(
        self,
    ) -> CacheStatus:
        print("Checking cache integrity........................")
        cache_dir_present = os.path.isdir(CACHE_DIR_PATH)
        if not cache_dir_present:
            return CacheStatus.NOT_BUILT

        missing_files = [f for f in EXPECTED_CACHE_DIR_FILES if not os.path.exists(f)]
        if missing_files:
            return CacheStatus.CORRUPT

        return CacheStatus.BUILT

    def is_broken(self) -> bool:
        return (
            self.cache_status == CacheStatus.CORRUPT
            or self.cache_status == CacheStatus.NOT_BUILT
        )
