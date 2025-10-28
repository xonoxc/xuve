from lib.data_loaders import load_movie_data
from lib.indexes.inverted_index import InvertedIndex
from lib.semantic_search import SemanticSearch


def build_cache() -> None:
    movie_data = load_movie_data()
    # build inverted index cache
    CURRENT_INVERTED_INDEX = InvertedIndex()
    CURRENT_INVERTED_INDEX.build(
        movie_data,
    )
    CURRENT_INVERTED_INDEX.save()

    # build semantic cache
    SEMANTICSEARCH = SemanticSearch()
    SEMANTICSEARCH.build_embeddings(
        movie_data,
    )
    SEMANTICSEARCH.save_embeddings()


if __name__ == "__main__":
    build_cache()
