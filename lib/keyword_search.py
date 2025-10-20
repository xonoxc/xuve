from itertools import islice
from typing import List

from config.data import DEFAULT_SEARCH_LIMIT
from lib.indexes.inverted_index import CacheStatus, InvertedIndex
from typedicts.movies import Movie

from .tokenize import tokenize


CURRENT_INVERTED_INDEX = InvertedIndex()


def search(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> List[Movie]:
    global index

    populate_index()

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    result_list: List[Movie] = []

    for token in tokenized_query:
        matching_docs = CURRENT_INVERTED_INDEX.get_documents(token)
        result_list.extend(matching_docs)

    result_list = list(
        {movie["id"]: movie for movie in result_list}.values(),
    )

    return list(
        islice(result_list, limit),
    )


def calc_bm25_idf(term: str) -> float:
    populate_index()
    try:
        return CURRENT_INVERTED_INDEX.get_bm25_idf(
            term,
        )
    except ValueError as e:
        print(f"Error calculating BM25 IDF for term '{term}': {str(e)}")
        return 0.0


def term_freq(doc_id: int, term: str) -> int:
    populate_index()
    try:
        return CURRENT_INVERTED_INDEX.get_token_frequencies(
            doc_id,
            text=term,
        )
    except ValueError:
        return 0


def tf_idf(doc_id: int, term: str) -> float:
    return term_freq(doc_id, term) * inverse_document_freq(
        term,
    )


def inverse_document_freq(term: str) -> float:
    populate_index()

    term_doc_count = len(
        CURRENT_INVERTED_INDEX.get_doc_ids(term),
    )
    total_doc_count = len(CURRENT_INVERTED_INDEX.docmap)

    import math

    # Formual for IDF with smoothing that is adding 1 to numerator and denominator
    # to prevent division by zero and log(0)
    return math.log(
        (total_doc_count + 1) / (term_doc_count + 1),
    )


# populate index func
# only runs if index not loaded
# if not build , use :- python -m cli.build  to build and cache the index
def populate_index() -> None:
    if CURRENT_INVERTED_INDEX.is_loaded:
        print("already loaded returning early...")
        return

    match CURRENT_INVERTED_INDEX.cache_status:
        case CacheStatus.NOT_BUILT:
            print(
                "Index not built. Please build the index first using build command.",
            )
            process_dot_exit()

        case CacheStatus.CORRUPT:
            print(
                "Cache Corrupted! Please rebuild the cache to proceed.",
            )
            process_dot_exit()

        case CacheStatus.BUILT:
            print("Loading index from cache...")
            CURRENT_INVERTED_INDEX.load()
            print("Index loaded from cache.")


def process_dot_exit():
    import sys

    sys.exit(1)
