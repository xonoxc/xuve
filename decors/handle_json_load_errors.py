from json import JSONDecodeError
from functools import wraps
from typing import Callable, TypeVar, Any, cast


F = TypeVar("F", bound=Callable[..., Any])


def handle_json_errors(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return fn(*args, **kwargs)
        except JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    return cast(F, wrapper)
