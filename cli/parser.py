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

    return parser
