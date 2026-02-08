from __future__ import annotations

import asyncio
import functools
import time
from collections import deque


def rate_limited(category: str, max_requests: int, per_seconds: int):
    """Rate-limit a tool handler using a monotonic sliding window."""

    def decorator(func):
        def _apply_rate_limit(args, kwargs):
            ctx = kwargs.get("ctx")
            if ctx is None:
                if not args:
                    raise KeyError(
                        "rate_limited wrapper requires ctx as a keyword argument or first positional argument"
                    )
                ctx = args[0]

            rate_limit = ctx.request_context.lifespan_context.rate_limit
            bucket = rate_limit.get(category)
            if not isinstance(bucket, deque):
                bucket = deque(bucket or [])
                rate_limit[category] = bucket

            now = time.monotonic()
            while bucket and (now - bucket[0]) > per_seconds:
                bucket.popleft()

            if len(bucket) >= max_requests:
                return (
                    False,
                    f"Tool limit exceeded: {max_requests} requests per {per_seconds} s. Try again later.",
                )

            bucket.append(now)
            return True, None

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                allowed, msg = _apply_rate_limit(args, kwargs)
                if not allowed:
                    return msg
                return func(*args, **kwargs)

        doc = wrapper.__doc__ or ""
        wrapper.__doc__ = f"Limit: {max_requests}req/{per_seconds}s. {doc}"
        return wrapper

    return decorator
