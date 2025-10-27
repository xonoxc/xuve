from cli.parser import setup_parsers
from lib.actions import act


def main() -> None:
    parser = setup_parsers()
    args = parser.parse_args()

    act(args, parser)


if __name__ == "__main__":
    main()
