import os
import json
import numpy as np
from config.data import CHUNK_EMBDEDDINGS_PATH, CHUNK_METADATA_PATH
from decors.handle_json_load_errors import handle_json_errors
from lib.data_loaders import load_movie_data
from lib.semantic_search import SemanticSearch, semantic_chunk
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
        self.chunked_embeddings: Optional[np.ndarray] = None
        # chunked metadata
        self.chunked_metadata: Optional[List[Dict[str, Any]]] = None
        # total number of chunks
        self.total_chunks: int = 0

        self.is_loaded = False

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


def embed_chunks() -> None:
    chunked_semeantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semeantic_search.load_chunk_embeddings()

    print(f"Generated {len(embeddings)} chunked embeddings")
