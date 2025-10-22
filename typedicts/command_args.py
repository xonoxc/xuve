from typing import NamedTuple, Any


class CLIarg(NamedTuple):
    name: str
    type: type
    help: str
    is_optional: bool
    default: Any
