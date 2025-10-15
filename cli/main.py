from cli.parser import setup_parser
from lib.keyword_search import search


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching For:", args.query, "....")
            result = search(args.query)

            for i, res in enumerate(result, start=1):
                print(f"{i}. {res['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
