import os


# search limit for movie titles
DEFAULT_SEARCH_LIMIT = 5


# Define the project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(__file__),
)

# Path where we have stopwords stored
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

# Define the path to the data file
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


# index cache path
CACHE_DIR_PATH = os.path.join(PROJECT_ROOT, "cache")

# index for the files to be written in the path for cache

# file containing indexes
INDEX_CACHE_PATH = os.path.join(CACHE_DIR_PATH, "index.json")

# file containing term frequeincies for ranking
TERM_FREQUENCIES_PATH = os.path.join(
    CACHE_DIR_PATH,
    "term_frequencies.json",
)


# file containing document length for each doc for ranking
DOC_LENGTH_PATH = os.path.join(
    CACHE_DIR_PATH,
    "doc_length.json",
)


AVG_DOC_LENGTH = os.path.join(CACHE_DIR_PATH, "avg_doc_length.json")


MOVIE_EMBDEDDINGS_PATH = os.path.join(
    CACHE_DIR_PATH,
    "movie_embeddings.npy",
)


CHUNK_EMBDEDDINGS_PATH = os.path.join(
    CACHE_DIR_PATH,
    "chunk_embeddings.npy",
)

CHUNK_METADATA_PATH = os.path.join(
    CACHE_DIR_PATH,
    "chunk_embeddings.json",
)


# EXPECTED CACHE FILES (if these files are not the cache directory the cache is considered corrupt)
EXPECTED_CACHE_DIR_FILES = [
    INDEX_CACHE_PATH,
    TERM_FREQUENCIES_PATH,
    DOC_LENGTH_PATH,
    AVG_DOC_LENGTH,
    MOVIE_EMBDEDDINGS_PATH,
    CHUNK_EMBDEDDINGS_PATH,
    CHUNK_METADATA_PATH,
]


# k1 value for BM25 algorithm
# this controls term frequency saturation
# i.e  how quickly the term frequency contribution to the score
BM25_K1 = 1.5


# this one is the factor for document length normalization
BM25_B = 0.75


# ranking score precision
SCORE_PRECISION = 4


# alpha constant value that controls how much keyword
# for example => alpha = 0.7 means 70% keyword score weightage and 30% semantic
# alpha = 0.5 means 50/50 weightage
# alpha = 0.3 means 30% keyword score weight and 70% sematic
ALPHA = 0.5
