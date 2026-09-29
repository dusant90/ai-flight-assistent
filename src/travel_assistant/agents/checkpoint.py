# Postgres checkpointer.
#
# A checkpointer saves the graph's state after every step, keyed by a thread_id.
# This gives us two things:
#   1. Memory: a conversation survives a restart (resume the same thread_id).
#   2. Pause/resume: interrupt() can halt the graph mid-run and later continue
#      exactly where it stopped - which is what makes human approval possible.
#
# We store checkpoints in the SAME Postgres as the RAG vectors and (later) the
# bookings. LangGraph creates its own tables automatically via saver.setup().

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from travel_assistant.config import Settings, get_settings


def _conn_string(settings: Settings) -> str:
    # The checkpointer talks to Postgres via psycopg directly, which wants a
    # plain libpq URL. Our DATABASE_URL uses the SQLAlchemy dialect form
    # ("postgresql+psycopg://"), so strip the "+psycopg" part.
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://")


@asynccontextmanager
async def open_checkpointer(settings: Settings | None = None):
    # Async context manager that yields a ready-to-use checkpointer.
    # setup() creates the checkpoint tables on first use (safe to call again).
    settings = settings or get_settings()
    async with AsyncPostgresSaver.from_conn_string(_conn_string(settings)) as saver:
        await saver.setup()
        yield saver
