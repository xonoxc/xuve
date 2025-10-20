from argparse import Namespace, ArgumentParser
from lib.keyword_search import search, term_freq, inverse_dcoument_freq


def act(args: Namespace, parser: ArgumentParser) -> None:
    match args.command:
        case "search":
            print("Searching For:", args.query, "....")
            result = search(args.query)

            if len(result) == 0:
                print("No results found.")
                return

            for i, res in enumerate(result, start=1):
                print(f"{i}. {res['title']}")

        case "tf":
            print("Finding term frequency for:", args.term, "....")
            freq = term_freq(
                int(args.doc_id),
                args.term,
            )

            print(
                f"Term Frequency of '{args.term}' in Document ID {args.doc_id}: {freq}"
            )

        case "idf":
            print(f"Calcualting the idf value for term {args.term}....")

            idf = inverse_dcoument_freq(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()
