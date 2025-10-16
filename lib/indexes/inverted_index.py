import os
from typing import Dict, List, Set

from config.data import INDEX_CACHE_PATH
from decors.handle_file_errors import handle_file_errors, raise_error
from lib.tokenize import tokenize
from typedicts.movies import Movie

from concurrent.futures import ThreadPoolExecutor


class InvertedIndex:
    index: Dict[str, Set[int]] = {}
    docmap: Dict[int, Movie] = {}
    is_loaded: bool = False
    is_built: bool = False

    def __init__(self) -> None:
        # if cache file exists, mark as built
        self.is_built = os.path.isfile(
            INDEX_CACHE_PATH,
        )

    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize(text)

        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(
        self,
        term: str,
        limit: int = 5,
    ) -> Set[Movie]:
        doc_ids = self.index.get(
            term,
            set(),
        )
        return {self.docmap[doc_id] for doc_id in list(doc_ids)[:limit]}

    def build(self, movies: List[Movie]) -> None:
        print("Building index.....")

        for movie in movies:
            movie_id = int(movie["id"])  # ensure consistent type
            self.docmap[movie_id] = movie
            self.__add_document(movie_id, f"{movie['title']} {movie['description']}")

        print("Index built.....")

    @handle_file_errors(custom_handlers=None)
    def save(self, path: str = INDEX_CACHE_PATH) -> None:
        import json

        print("Saving index to path...")

        print("creating cache directory...")
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )
        print("cache directory created!!")

        print("writing index to path...")

        data_to_save = {
            "index": {token: list(ids) for token, ids in self.index.items()},
            "docmap": self.docmap,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data_to_save,
                f,
                ensure_ascii=False,
                indent=2,
            )

        self.is_built = True
        print("Index saved to path...")

    @handle_file_errors({FileNotFoundError: raise_error})
    def load(self) -> None:
        import json

        with open(INDEX_CACHE_PATH, "r") as f:
            data = json.load(f)

        self.index = {token: set(ids) for token, ids in data["index"].items()}
        self.docmap = data["docmap"]

        self.is_loaded = True

    print("Index loaded successfully!")
