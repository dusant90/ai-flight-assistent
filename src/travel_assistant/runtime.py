# Small runtime helpers shared by entry points (CLIs, later the API).

import asyncio
import sys


def configure_event_loop() -> None:
    # On Windows, Python's default asyncio loop is the ProactorEventLoop, which
    # psycopg's async mode cannot use. Switch to the SelectorEventLoop so the
    # Postgres checkpointer works. No-op on other platforms. Call this once,
    # before asyncio.run(), in every entry point that touches async Postgres.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
