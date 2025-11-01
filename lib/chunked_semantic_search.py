import os
import json
import numpy as np
from config.data import CHUNK_EMBDEDDINGS_PATH, CHUNK_METADATA_PATH, SCORE_PRECISION
from decors.handle_json_load_errors import handle_json_errors
from lib.data_loaders import load_movie_data
from lib.semantic_search import SemanticSearch, semantic_chunk, cosine_similarity
from typedicts.movies import Movie

from typing import List, Dict, Optional, Any


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        # initialize the parent class
        super().__init__(model_name)

        # chunked embeddings and metadata
        self.chunked_embeddings: np.ndarray = np.empty((0,))
        # chunked metadata
        self.chunked_metadata: Optional[List[Dict[str, Any]]] = None
        # total number of chunks
        self.total_chunks: int = 0

        self.is_loaded = False

    def search_chunks(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.generate_embedding(
            query,
        )

        chunk_scores = []
        for _, (embedding, meta_data) in enumerate(
            zip(self.chunked_embeddings, self.chunked_metadata or []),
        ):
            cosine_sim = cosine_similarity(embedding, query_embedding)

            movie_idx = meta_data["movie_idx"]
            chunk_idx = meta_data["chunk_idx"]

            movie = self.documents[movie_idx]
            title = movie.get("title", "Unknown")
            document_text = movie.get("description", "")

            chunk_scores.append(
                {
                    "id": movie.get("id", movie_idx),
                    "title": title,
                    "document": document_text[:100],
                    "score": round(float(cosine_sim), SCORE_PRECISION),
                    "metadata": {
                        "chunk_idx": chunk_idx,
                        "movie_idx": movie_idx,
                        "total_chunks": meta_data.get("total_chunks", 0),
                    },
                }
            )

        chunk_scores.sort(key=lambda x: x["score"], reverse=True)

        return chunk_scores[:limit]

    # checcks if chunked embedding cache files exist
    def check_embedding_cache_exists(self) -> bool:
        return os.path.exists(CHUNK_EMBDEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        )

    # method to build chunked embeddings
    def build_chunk_embeddings(
        self,
        documents: List[Movie],
    ) -> None:
        print("building chunked embeddings ......")
        self.documents = documents

        all_chunks: List[str] = []
        meta_data: List[Dict] = []

        for doc_idx, document in enumerate(documents):
            description = document.get("description", "").strip()
            if not description:
                continue

            # split into 4 sentence chunks with 1 sentence overlap
            semantic_chunks = semantic_chunk(
                text=document.get("description"), max_chunk_size=4, overlap=1
            )

            for chunk_idx, chunk in enumerate(semantic_chunks):
                print(
                    f"\r⏳ Processing chunk {chunk_idx + 1}/{len(semantic_chunks)}...",
                    end="",
                )
                all_chunks.append(chunk)
                meta_data.append(
                    {
                        "movie_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(semantic_chunks),
                    }
                )

            self.doc_map[int(document["id"])] = document

        print("encoding embeddings...")
        embeddings = self.model.encode(all_chunks)

        self.chunked_embeddings = embeddings
        self.chunked_metadata = meta_data

        print("writing files to cache....")
        np.save(CHUNK_EMBDEDDINGS_PATH, embeddings)

        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": meta_data, "total_chunks": len(all_chunks)}, f, indent=2
            )

    # utitlity method to load json with error handling
    @handle_json_errors
    def load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    # method to load chunked embeddings from cache
    def load_chunk_embeddings(self) -> np.ndarray:
        documents = load_movie_data()

        self.documents = documents

        for _, document in enumerate(documents):
            self.doc_map[int(document["id"])] = document

        # we are not checking if these files exist
        # cause if they don't exist, the program won't reach here
        # you'll get a cache error
        embddings = np.load(CHUNK_EMBDEDDINGS_PATH)
        metadata_file_json = self.load_json(CHUNK_METADATA_PATH)

        self.chunked_embeddings = embddings
        self.chunked_metadata = metadata_file_json.get("chunks", [])
        self.total_chunks = metadata_file_json.get("total_chunks", 0)

        return embddings


CHUNKED_SEMEANTIC_SEARCH = ChunkedSemanticSearch()


def populate_embeddings() -> np.ndarray:
    if not CHUNKED_SEMEANTIC_SEARCH.check_embedding_cache_exists():
        CHUNKED_SEMEANTIC_SEARCH.build_chunk_embeddings(
            load_movie_data(),
        )

    return CHUNKED_SEMEANTIC_SEARCH.load_chunk_embeddings()


# actual searching function
def chunked_semantic_search(
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    populate_embeddings()
    return CHUNKED_SEMEANTIC_SEARCH.search_chunks(
        query,
        limit,
    )


def embed_chunks() -> None:
    embeddings = populate_embeddings()
    print(
        f"Generated {len(embeddings)} chunked embeddings",
    )
