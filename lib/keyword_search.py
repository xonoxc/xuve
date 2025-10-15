import json
import string
from typing import List
from config.data import DATA_PATH, DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH
from decors.handle_file_errors import handle_file_errors
from itertools import islice

from typedicts.movies import Movie


@handle_file_errors
def load_data() -> List[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


@handle_file_errors
def load_stop_words() -> List[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
    return stop_words


def search(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> List[Movie]:
    movies = load_data()
    query_tokens = set(tokenize(query))  # set for faster q in t checks

    # checks if any of the query tokens
    def matches(movie: Movie) -> bool:
        title_tokens = tokenize(movie["title"])
        return any(q in t for q in query_tokens for t in title_tokens)

    # creating a slice from the generator
    # with the given limit
    return list(
        islice(
            (movie for movie in movies if matches(movie)),
            limit,
        )
    )


# loading the stopd words in memory for once
# yeah this could have been an array for set gives O(1) time complextiy for checks
STOPWORDS = set(load_stop_words())


# tokenizes the passed string
def tokenize(arg_str: str) -> list[str]:
    cleaned = get_cleaned_string(
        arg_str,
    )
    return [token for token in cleaned.split() if token and token not in STOPWORDS]


# this function removes all
# the puncutaiton marks from query string
# and makes then lower case for comparison

# caching the translation table so that it does not get created
# everytime [IMPORTANT]: specically because this function is
# used in loops
PUNCTUATION_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)


def get_cleaned_string(raw_string: str) -> str:
    cleaned = raw_string.translate(
        PUNCTUATION_TRANSLATION_TABLE,
    ).lower()

    # whitespace normalization
    return " ".join(cleaned.split())
