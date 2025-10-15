import json
from config.data import DATA_PATH, DEFAULT_SEARCH_LIMIT
from decors.handle_file_errors import handle_file_errors


@handle_file_errors
def load_data() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def search(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[dict]:
    movies = load_data()
    results = []

    for movie in movies:
        if query.lower() in movie["title"].lower():
            results.append(movie)
            if len(results) >= limit:
                break

    return results
