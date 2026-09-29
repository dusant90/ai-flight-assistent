"""AI travel assistant: flight + hotel booking agents built with LangGraph."""

# Turn on LangSmith tracing as soon as the package is imported, so EVERY entry
# point (CLI, tests, later the API) records traces without having to remember to
# call setup_tracing() itself. Guarded so a tracing problem can never stop the
# app from importing.
try:
    from travel_assistant.tracing import setup_tracing

    setup_tracing()
except Exception as _exc:  # pragma: no cover - tracing is best-effort
    print(f"[travel_assistant] tracing setup skipped: {_exc}")
