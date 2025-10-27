import argparse
from typing import List
from config.data import BM25_B, BM25_K1
from typedicts.command_args import CLIarg


# function responsible for creating parsers
# with the given arguements
# handles optional args as well
def create_parser(
    subparsers: argparse._SubParsersAction,
    name: str,
    help_text: str,
    arguments: List[CLIarg],
):
    parser = subparsers.add_parser(name, help=help_text)

    for arg in arguments:
        if not arg.is_optional:
            parser.add_argument(arg.name, type=arg.type, help=arg.help)
        else:
            parser.add_argument(
                arg.name,
                type=arg.type,
                help=arg.help,
                default=arg.default,
            )


def setup_keyword_search_parser(master_subparser: argparse._SubParsersAction) -> None:
    keyword_parser = master_subparser.add_parser(
        "keyword", help="Keyword-based search commands"
    )
    subparsers = keyword_parser.add_subparsers(dest="command", required=True)

    # Search command
    create_parser(
        subparsers,
        "search",
        "Search Movies using BM25",
        [
            CLIarg(
                name="query",
                type=str,
                help="Search Query",
                is_optional=False,
                default=None,
            ),
            # Example optional flag
            CLIarg(
                name="--limit",
                type=int,
                help="Max results to return",
                is_optional=True,
                default=10,
            ),
        ],
    )

    # Term frequency command
    create_parser(
        subparsers,
        "tf",
        "Search term frequency in a document",
        [
            CLIarg(
                name="doc_id",
                type=str,
                help="Document ID to search in",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="term",
                type=str,
                help="Term to search frequency for",
                is_optional=False,
                default=None,
            ),
        ],
    )

    # TF-IDF command
    create_parser(
        subparsers,
        "tfidf",
        "Calculate the TF-IDF value for the term",
        [
            CLIarg(
                name="doc_id",
                type=str,
                help="Document ID for TF-IDF calculation",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="term",
                type=str,
                help="Term to calculate TF-IDF value for",
                is_optional=False,
                default=None,
            ),
        ],
    )

    # IDF command
    create_parser(
        subparsers,
        "idf",
        "Calculate the IDF (Inverse Document Frequency) value for the term",
        [
            CLIarg(
                name="term",
                type=str,
                help="Term to calculate IDF",
                is_optional=False,
                default=None,
            ),
        ],
    )

    # BM25 IDF command
    create_parser(
        subparsers,
        "bm25idf",
        "Calculate the BM25_IDF value for the term",
        [
            CLIarg(
                name="term",
                type=str,
                help="Term to calculate BM25_IDF",
                is_optional=False,
                default=None,
            ),
        ],
    )

    # BM25 TF command
    create_parser(
        subparsers,
        "bm25tf",
        "Calculate the BM25_TF value for the term",
        [
            CLIarg(
                name="doc_id",
                type=str,
                help="Document ID for BM25_TF calculation",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="term",
                type=str,
                help="Term to calculate BM25_TF",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--k1",
                type=float,
                help="K1 constant for term frequency control",
                is_optional=True,
                default=BM25_K1,
            ),
            CLIarg(
                name="--b",
                type=float,
                help="Tunable BM25 b parameter",
                is_optional=True,
                default=BM25_B,
            ),
        ],
    )

    create_parser(
        subparsers,
        "bm25search",
        "Search movies using full BM25 scoring",
        [
            CLIarg(
                name="query",
                type=str,
                help="Search query",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--limit",
                type=int,
                help="Limit of the number of search results",
                is_optional=True,
                default=5,
            ),
        ],
    )


def setup_semanitc_parser(master_subparser: argparse._SubParsersAction) -> None:
    keyword_parser = master_subparser.add_parser(
        "semantic", help="Keyword-based search commands"
    )
    subparsers = keyword_parser.add_subparsers(dest="command", required=True)

    create_parser(subparsers, "verify", "Verify model", [])


def setup_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-mode search CLI 🌸")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    setup_semanitc_parser(subparsers)
    setup_keyword_search_parser(subparsers)

    return parser
