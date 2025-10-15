import json
from typing import Any, Callable


def handle_file_errors(func: Callable[..., Any]):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            match e:
                case FileNotFoundError():
                    print("Error: data.json file not found.")
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
