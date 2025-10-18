from lib.data_loaders import load_movie_data
from lib.indexes.inverted_index import InvertedIndex


if __name__ == "__main__":
    CURRENT_INVERTED_INDEX = InvertedIndex()
    CURRENT_INVERTED_INDEX.build(
        load_movie_data(),
    )
    CURRENT_INVERTED_INDEX.save()
