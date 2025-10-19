import os
from itertools import islice
from typing import List

from config.data import DEFAULT_SEARCH_LIMIT
from lib.indexes.inverted_index import InvertedIndex
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


# populate index func
# only runs if index not loaded
# if not build , use :- python -m cli.build  to build and cache the index
def populate_index() -> None:
    if CURRENT_INVERTED_INDEX.is_loaded:
        return

    if not CURRENT_INVERTED_INDEX.is_built:
        print("Index not built. Please build the index first. using build command.")
        os._exit(1)

    print("Loading index from cache...")
    CURRENT_INVERTED_INDEX.load()
    print("Index loaded from cache.")
