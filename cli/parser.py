import argparse


# Function to set up the argument parser
def setup_parser() -> argparse.ArgumentParser:
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Keyword Search CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create the parser for the "search" command
    search_parsers = subparsers.add_parser(
        "search",
        help="Search Movies using BM25",
    )
    search_parsers.add_argument("query", type=str, help="Search Query")

    # Create the parser for the "tf" command
    tf_parser = subparsers.add_parser(
        "tf",
        help="search term frequency in a document",
    )
    tf_parser.add_argument("doc_id", type=str, help="Document ID to search in")
    tf_parser.add_argument("term", type=str, help="Term to search frequency for")

    # Create the parser for the "tfidf" command
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="calcualte the TF-IDF value for the term",
    )
    tfidf_parser.add_argument(
        "doc_id", type=str, help="Document ID to use for calcualation of TF-IDF"
    )
    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate TF-IDF value for",
    )

    # Create the parser for the "idf" command
    idf_parser = subparsers.add_parser(
        "idf",
        help="calcualte the IDF(Inverse Document Frequency) value for the term",
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate IDF(Inverse Document Frequency) value for",
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="calcualte the BM2_IDF value for the term",
    )
    bm25_idf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate BM2_IDF value for",
    )
    return parser
