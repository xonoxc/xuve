from typing import NamedTuple, Any, Optional


class CLIarg(NamedTuple):
    name: str
    type: type
    help: str
    is_optional: bool
    default: Any
    nargs: Optional[str] = None
