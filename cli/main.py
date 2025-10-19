from cli.parser import setup_parser
from lib.keyword_search import search, term_freq


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
