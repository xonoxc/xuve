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
    choices: List[str] | None = None,
):
    parser = subparsers.add_parser(name, help=help_text)

    for arg in arguments:
        if not arg.is_optional:
            parser.add_argument(
                arg.name,
                type=arg.type,
                help=arg.help,
                choices=choices,
                nargs=arg.nargs,
            )
        else:
            parser.add_argument(
                arg.name,
                type=arg.type,
                help=arg.help,
                default=arg.default,
                choices=choices,
                nargs=arg.nargs,
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


def setup_semanitc_parser(
    master_subparser: argparse._SubParsersAction,
) -> None:
    keyword_parser = master_subparser.add_parser(
        "semantic", help="Keyword-based search commands"
    )
    subparsers = keyword_parser.add_subparsers(dest="command", required=True)

    create_parser(subparsers, "verify", "Verify model", [])
    create_parser(
        subparsers,
        "embed_text",
        "Create embedding from the given text",
        [
            CLIarg(
                name="text",
                type=str,
                help="argument text",
                is_optional=False,
                default=None,
            ),
        ],
    )
    create_parser(
        subparsers,
        "embedquery",
        "Create embedding from the given text",
        [
            CLIarg(
                name="embed_query_text",
                type=str,
                help="Query text",
                is_optional=False,
                default=None,
            ),
        ],
    )
    create_parser(subparsers, "verify_embeddings", "Verify embeddings", [])
    create_parser(
        subparsers,
        "semantic_search",
        "Search Semantically for specific movies",
        [
            CLIarg(
                name="query",
                type=str,
                help="Search Query",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--limit",
                type=int,
                help="Max results to return",
                is_optional=True,
                default=10,
            ),
        ],
    )

    create_parser(
        subparsers,
        "semantic_chunk",
        "Semantically chunk the given text",
        [
            CLIarg(
                name="text",
                type=str,
                help="Text to chunk",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--max_chunk_size",
                type=int,
                help="Max chunk size",
                is_optional=True,
                default=4,
            ),
            CLIarg(
                name="--overlap",
                type=int,
                help="Max results to return",
                is_optional=True,
                default=0,
            ),
        ],
    )

    create_parser(
        subparsers, "embed_chunks", "Build Chunk Embeddings the given text", []
    )

    # TF-IDF command
    create_parser(
        subparsers,
        "search_chunked",
        "Search using chunked semantic search",
        [
            CLIarg(
                name="query",
                type=str,
                help="query text to search",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--limit",
                type=str,
                help="the nummber of search resuts to return",
                is_optional=True,
                default=10,
            ),
        ],
    )

    create_parser(
        subparsers,
        "chunk",
        "Chunk the given text",
        [
            CLIarg(
                name="text",
                type=str,
                help="Text to chunk",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--chunksize",
                type=int,
                help="Max chunk size",
                is_optional=True,
                default=200,
            ),
            CLIarg(
                name="--overlap",
                type=int,
                help="Max results to return",
                is_optional=True,
                default=50,
            ),
        ],
    )


def setup_hybrid_search_parser(
    master_subparser: argparse._SubParsersAction,
) -> None:
    hybrid_search_parser = master_subparser.add_parser(
        "hybrid",
        help="use hybrid search mode",
    )

    subparsers = hybrid_search_parser.add_subparsers(dest="command", required=True)

    create_parser(
        subparsers,
        "normalize",
        "normalize bm25 and consine scores",
        [
            CLIarg(
                name="scores",
                type=float,
                help="Search scores to be normalized",
                is_optional=False,
                default=None,
                nargs="+",
            ),
        ],
    )

    create_parser(
        subparsers,
        "rrf-search",
        "rrf based hybrid search",
        [
            CLIarg(
                name="query",
                type=str,
                help="Search Query",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--k",
                type=int,
                help="rrf-k value constant value",
                is_optional=True,
                default=0.5,
            ),
            CLIarg(
                name="--limit",
                type=int,
                help="Limit of the number of search results",
                is_optional=True,
                default=5,
            ),
            CLIarg(
                name="--enhance",
                type=str,
                choices=["spell", "rewrite", "expand"],
                is_optional=False,
                help="Query enhancement method",
                default=None,
            ),
            CLIarg(
                name="--rerank-method",
                type=str,
                choices=["spell", "rewrite", "expand"],
                is_optional=False,
                help="Query enhancement method",
                default=None,
            ),
        ],
    )

    create_parser(
        subparsers,
        "weighted-search",
        "hybrid search based on normalized bm25 and consine scores",
        [
            CLIarg(
                name="query",
                type=str,
                help="Search Query",
                is_optional=False,
                default=None,
            ),
            CLIarg(
                name="--alpha",
                type=int,
                help="Alpha constant value",
                is_optional=True,
                default=0.5,
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


# just add functions here and they'll be
# setup automatically
PARSER_FUNCS = [
    setup_semanitc_parser,
    setup_keyword_search_parser,
    setup_hybrid_search_parser,
]


# master parser setup function
# this initializes all subparsers
# that are :-
# 1. semantic search parser
# 2. keyword keyword search parser
# 3. hybrid search parser
def setup_parsers() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-mode search CLI",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    for parser_setup_func in PARSER_FUNCS:
        parser_setup_func(
            subparsers,
        )

    return parser
