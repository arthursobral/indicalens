"""Optional Langfuse tracing. With LANGFUSE_PUBLIC_KEY/SECRET_KEY unset, `observe`
is a no-op decorator and nothing leaves the machine, so tests/CI need no account."""

import os

from src import config  # noqa: F401  (loads .env before we read the keys)

ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, observe
else:
    def observe(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


def record_generation(**fields) -> None:
    if ENABLED:
        get_client().update_current_generation(**fields)


def flush() -> None:
    if ENABLED:
        get_client().flush()
