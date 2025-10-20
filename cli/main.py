from cli.parser import setup_parser
from lib.actions import act


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    act(args, parser)


if __name__ == "__main__":
    main()
