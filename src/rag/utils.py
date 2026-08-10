from typing import Callable
import sys
import functools
import json
import pydantic


def catch_cli_errors(func: Callable) -> Callable:
    """Catches CLI runtime errors and logs formatted messages to stderr.

    Args:
        func (Callable): Target function to wrap with exception handling.

    Returns:
        Callable: The wrapped function with centralized error catching.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except OSError as e:
            path_str = f" '{e.filename}'" if e.filename else ""
            print(f"Error: {e.strerror}{path_str}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(
                f"Error: Malformed JSON at line {e.lineno}, column {e.colno}:",
                f" {e.msg}.\nPlease verify file syntax.", file=sys.stderr
            )
            sys.exit(1)
        except pydantic.ValidationError as e:
            print(
                "Error: Invalid data structure in input file.", file=sys.stderr
            )
            for error in e.errors():
                field = " -> ".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                print(f"  - Field '{field}': {msg}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occurred ({type(e).__name__}): {str(e)}",
                file=sys.stderr
            )
    return wrapper
