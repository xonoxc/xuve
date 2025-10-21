import argparse
from typing import List, Tuple


# function responsible for creating parses
# with the given arguments
# on subparsers
def create_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
    arguments: List[Tuple[str, type, str]],
) -> None:
    parser = subparsers.add_parser(
        name,
        help=help_text,
    )
    for arg_name, arg_type, arg_help in arguments:
        parser.add_argument(
            arg_name,
            type=arg_type,
            help=arg_help,
        )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    create_parser(
        subparsers,
        "search",
        "Search Movies using BM25",
        [("query", str, "Search Query")],
    )

    create_parser(
        subparsers,
        "tf",
        "Search term frequency in a document",
        [
            ("doc_id", str, "Document ID to search in"),
            ("term", str, "Term to search frequency for"),
        ],
    )

    create_parser(
        subparsers,
        "tfidf",
        "Calculate the TF-IDF value for the term",
        [
            ("doc_id", str, "Document ID to use for calculation of TF-IDF"),
            ("term", str, "Term to calculate TF-IDF value for"),
        ],
    )

    create_parser(
        subparsers,
        "idf",
        "Calculate the IDF (Inverse Document Frequency) value for the term",
        [("term", str, "Term to calculate IDF")],
    )

    create_parser(
        subparsers,
        "bm25idf",
        "Calculate the BM25_IDF value for the term",
        [("term", str, "Term to calculate BM25_IDF")],
    )

    create_parser(
        subparsers,
        "bm25tf",
        "Calculate the BM25_TF value for the term",
        [
            ("doc_id", str, "Document ID to use for calculation of BM25_TF"),
            ("term", str, "Term to calculate BM25_TF"),
        ],
    )

    return parser
