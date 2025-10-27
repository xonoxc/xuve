import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Counter, Dict, List, Optional, Set, Tuple

from nltk import defaultdict

from config.data import (
    AVG_DOC_LENGTH,
    BM25_B,
    BM25_K1,
    CACHE_DIR_PATH,
    DOC_LENGTH_PATH,
    EXPECTED_CACHE_DIR_FILES,
    INDEX_CACHE_PATH,
    TERM_FREQUENCIES_PATH,
)
from decors.handle_file_errors import handle_file_errors, raise_error
from lib.enums.cache_status import CacheStatus
from lib.tokenize import tokenize
from typedicts.files_to_write import CacheFilesToWrite
from typedicts.movies import Movie


class InvertedIndex:
    def __init__(self) -> None:
        # index
        self.index: Dict[str, Set[int]] = {}
        # document map
        self.docmap: Dict[int, Movie] = {}
        # term frequencies
        self.term_frequencies: Dict[int, Counter] = defaultdict(Counter)

        # is cache loaded status
        self.is_loaded: bool = False
        # cache status
        self.cache_status: CacheStatus = CacheStatus.NOT_BUILT

        # check cache integrity
        self.check_cache_integrity()

        # document lengths mapped to document ids
        self.doc_lengths: Dict[int, int] = defaultdict(int)

        self.avg_doc_length: float = 0.0

    def bm25_search(self, query: str, limit: int) -> List[Tuple[int, str, float]]:
        tokenized_query = tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        for doc_id in self.docmap.keys():
            current_score = 0.0
            for token in tokenized_query:
                current_score += self.bm25(doc_id, token)

            scores[doc_id] = current_score

        # returning ranked docs with limit
        ranked_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            (doc_id, self.docmap[doc_id]["title"], score)
            for doc_id, score in ranked_res
        ]

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(
            term,
        )

    # method to get the BM25 IDF for a term
    #  handles more cases then normal IDF
    # and this one is  (recommended)
    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = tokenize(term)
        if len(tokenized_term) != 1:
            raise ValueError(
                "term must be a single token",
            )

        df = len(self.get_doc_ids(tokenized_term[0]))
        total_docs = len(self.docmap)

        import math

        # bm25 idf formula
        return math.log(
            (total_docs - df + 0.5) / (df + 0.5) + 1,
        )

    # method to get bm25tf which handles term saturation
    # i.e. prevents high occurence terms from dominating the
    # search
    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        raw_term_freq = self.get_token_frequencies(
            doc_id,
            term,
        )

        doc_len = self.doc_lengths.get(doc_id)
        if doc_len is None:
            raise KeyError(f"doc with the id : {doc_id} does not exist")

        doc_length_normalization = 1 - b + b * (doc_len / self.avg_doc_length)

        return (raw_term_freq * (k1 + 1)) / (
            raw_term_freq + k1 * doc_length_normalization
        )

    #  method to check if the cache is broken
    def check_cache_integrity(self) -> None:
        cache_dir_present = os.path.isdir(CACHE_DIR_PATH)
        if not cache_dir_present:
            return

        missing_files = [f for f in EXPECTED_CACHE_DIR_FILES if not os.path.exists(f)]
        if missing_files:
            self.cache_status = CacheStatus.CORRUPT
            return

        self.cache_status = CacheStatus.BUILT

    # method to add documents to the structures we are playing with
    def __add_document(self, doc_id: int, text: str):
        tokenized_text = tokenize(text)

        self.doc_lengths[doc_id] = len(tokenized_text)

        for token in set(tokenized_text):
            self.index.setdefault(
                token,
                set(),
            )
            self.index[token].add(doc_id)

        self.term_frequencies.setdefault(doc_id, Counter())
        self.term_frequencies[doc_id].update(
            tokenized_text,
        )

    def get_token_frequencies(self, doc_id: int, text: str) -> int:
        tokens = tokenize(text)
        if len(tokens) != 1:
            raise ValueError(
                "term must be a single token",
            )
        return self.term_frequencies.get(
            doc_id,
            Counter(),
        ).get(tokens[0], 0)

    def get_doc_ids(self, term: str) -> Set[int]:
        return self.index.get(
            term,
            set(),
        )

    def get_documents(
        self,
        term: str,
        limit: Optional[int] = 5,
    ) -> List[Movie]:
        doc_ids = self.get_doc_ids(term)
        docs = [self.docmap[doc_id] for doc_id in list(doc_ids)[:limit]]

        if limit is None:
            return docs

        return docs[:limit]

    # Builder
    def build(self, movies: List[Movie]) -> None:
        print("Building index.....")

        for movie in movies:
            movie_id = int(movie["id"])
            self.docmap[movie_id] = movie
            self.__add_document(
                movie_id, f"{movie.get('title')} {movie.get('description')}"
            )

        print("Index built.....")

    def calc_avg_doclen(self) -> float:
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_avg_doc_len(self) -> float:
        if not self.avg_doc_length:
            return 0.0

        return self.calc_avg_doclen()

    # Save method
    @handle_file_errors(custom_handlers=None)
    def save(self, cache_path: str = CACHE_DIR_PATH) -> None:
        import json

        print("Saving index to path...")

        os.makedirs(cache_path, exist_ok=True)

        files_to_be_written: List[CacheFilesToWrite] = [
            {
                "path": INDEX_CACHE_PATH,
                "data": {
                    "index": {token: list(ids) for token, ids in self.index.items()},
                    "docmap": self.docmap,
                },
            },
            {"path": TERM_FREQUENCIES_PATH, "data": self.term_frequencies},
            {"path": DOC_LENGTH_PATH, "data": self.doc_lengths},
            {
                "path": AVG_DOC_LENGTH,
                "data": {"value": self.calc_avg_doclen()},
            },
        ]

        for file in files_to_be_written:
            with open(file.get("path"), "w", encoding="utf-8") as f:
                json.dump(
                    file.get("data"),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        self.cache_status = CacheStatus.BUILT
        print("Index saved to path...")

    @staticmethod
    def load_file(file_path: str) -> Tuple[str, Any]:
        with open(file_path, "r") as f:
            data = json.load(f)

        return file_path, data

    @handle_file_errors({FileNotFoundError: raise_error})
    def load(self) -> None:
        """Load all cache files concurrently."""
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.load_file, f) for f in EXPECTED_CACHE_DIR_FILES
            ]

            for future in as_completed(futures):
                file_path, data = future.result()

                if os.path.samefile(
                    file_path,
                    INDEX_CACHE_PATH,
                ):
                    self.index = {
                        token: set(ids) for token, ids in data["index"].items()
                    }
                    self.docmap = {int(k): v for k, v in data["docmap"].items()}
                elif os.path.samefile(
                    file_path,
                    TERM_FREQUENCIES_PATH,
                ):
                    self.term_frequencies = {
                        int(k): Counter(v) for k, v in data.items()
                    }
                elif os.path.samefile(
                    file_path,
                    DOC_LENGTH_PATH,
                ):
                    self.doc_lengths = {int(k): int(v) for k, v in data.items()}
                elif os.path.samefile(
                    file_path,
                    AVG_DOC_LENGTH,
                ):
                    self.avg_doc_length = data["value"]

        self.is_loaded = True
        print("Index loaded successfully!")
