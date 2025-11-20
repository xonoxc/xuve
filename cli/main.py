from cli.parser import setup_parsers
from lib.actions import act
from lib.cache import Cache
from lib.enums.cache_status import CacheStatus


def main() -> None:
    CACHE = Cache()
    if CACHE.cache_status != CacheStatus.BUILT:
        match CACHE.cache_status:
            case CacheStatus.NOT_BUILT:
                print(
                    "Cache not built. Please build the index first using 'python -m cli.build'."
                )
            case CacheStatus.CORRUPT:
                print("Cache corrupted. Please rebuild it before continuing.")
        return

    parser = setup_parsers()
    args = parser.parse_args()

    act(args, parser)


if __name__ == "__main__":
    main()
