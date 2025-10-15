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
