"""Tracing setup (LangSmith).

A "trace" is a recorded tree of everything that happened during one run: each
LLM call, each tool call, their inputs, outputs, timing and token counts.
LangSmith (smith.langchain.com) shows these as an interactive timeline, which
is how you debug and understand an agent's behaviour.

LangChain enables tracing automatically when certain environment variables are
present. Our settings live in a typed Settings object, so this module's job is
simply to copy the relevant ones into os.environ, once, at startup.
"""

import os

from travel_assistant.config import Settings, get_settings


def setup_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith tracing if configured.

    Reads settings and, when tracing is turned on, sets the environment
    variables LangChain looks for. Safe to call multiple times.

    Returns:
        True if tracing was enabled, False otherwise.
    """
    settings = settings or get_settings()

    if not settings.langsmith_tracing:
        return False

    if not settings.langsmith_api_key:
        # Turned on but no key: warn rather than crash, so the app still runs.
        print("[tracing] LANGSMITH_TRACING is true but LANGSMITH_API_KEY is empty; skipping.")
        return False

    # These are the exact variable names LangChain/LangSmith look for.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    return True
