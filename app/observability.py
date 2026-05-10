from __future__ import annotations

try:
    from langsmith import traceable
except Exception:

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator
