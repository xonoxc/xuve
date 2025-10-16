import string
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
