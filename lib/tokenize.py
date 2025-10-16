from .data_loaders import load_stop_words
from .clean_str import get_cleaned_string

from .stem_words import stem_words

# loading the stopd words in memory for once
# yeah this could have been an array for set gives O(1) time complextiy for checks
STOPWORDS = set(load_stop_words())


# tokenizes the passed string
def tokenize(arg_str: str) -> list[str]:
    cleaned = get_cleaned_string(
        arg_str,
    )
    return stem_words(
        [token for token in cleaned.split() if token and token not in STOPWORDS],
    )
