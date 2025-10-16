import json
from config.data import DATA_PATH, STOPWORDS_PATH
from decors.handle_file_errors import handle_file_errors

from typing import List
from typedicts.movies import Movie


@handle_file_errors(custom_handlers=None)
def load_movie_data() -> List[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


@handle_file_errors(custom_handlers=None)
def load_stop_words() -> List[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stop_words = f.read().splitlines()
    return stop_words
