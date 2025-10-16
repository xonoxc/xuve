from typing import Callable, Any, Type
import json


def handle_file_errors(
    custom_handlers: dict[Type[Exception], Callable[[Exception], Any]] | None = None,
):
    custom_handlers = custom_handlers or {}

    def decorator(func: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # check if there's a custom handler
                for exc_type, handler in custom_handlers.items():
                    if isinstance(e, exc_type):
                        return handler(e)

                # default behavior
                match e:
                    case FileNotFoundError():
                        print("Error: file not found.")
                    case PermissionError():
                        print("Error: Permission denied when reading movies.json.")
                    case json.JSONDecodeError():
                        print(f"Error decoding JSON: {e}")
                    case OSError():
                        print(f"OS error: {e}")
                    case Exception():
                        print(f"Unexpected error: {e}")
                return None

        return wrapper

    return decorator


def raise_error(e: Exception):
    raise e
