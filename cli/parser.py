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

    tf_parser = subparsers.add_parser(
        "tf",
        help="search term frequency in a document",
    )
    tf_parser.add_argument("doc_id", type=str, help="Document ID to search in")
    tf_parser.add_argument("term", type=str, help="Term to search frequency for")

    idf_parser = subparsers.add_parser(
        "idf",
        help="calcualte the IDF(Inverse Document Frequency) value for the term",
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Term to calculate IDF(Inverse Document Frequency) value for",
    )

    return parser
