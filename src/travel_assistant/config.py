# """Application configuration.

# Everything the app needs to know from the outside world (which LLM to use,
# database URL, API keys) is defined here in ONE typed place. We use
# `pydantic-settings`, which:

#   1. reads values from the `.env` file and real environment variables,
#   2. checks their types (e.g. a port must be an int),
#   3. gives clear errors early if something required is missing.

# Anywhere else in the code we just call `get_settings()` instead of reading
# `os.environ` directly. That keeps secrets and defaults in one spot.
# """

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration values, loaded from environment / .env."""

    # Tells pydantic-settings where to read from and how to behave.
    model_config = SettingsConfigDict(
        env_file=".env",           # read this file if present
        env_file_encoding="utf-8",
        extra="ignore",            # ignore env vars we don't declare here
        case_sensitive=False,      # LLM_PROVIDER and llm_provider both work
    )

    # --- LLM provider ---
    # Literal[...] means only these exact strings are allowed; anything else
    # raises a validation error at startup instead of failing deep in the code.
    llm_provider: Literal["ollama", "openai"] = "ollama"

    ollama_model: str = "qwen3"
    ollama_base_url: str = "http://localhost:11434"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Database ---
    database_url: str = "postgresql+psycopg://travel:travel@localhost:5433/travel"

    # --- RAG chunking ---
    # How to split documents into chunks before embedding:
    #   "auto"     -> use the best strategy per document (header for structured
    #                 markdown, semantic for prose). This is the sensible default.
    #   "header"   -> force structural splitting on markdown headings.
    #   "semantic" -> force meaning-based splitting on every document.
    # Set CHUNK_STRATEGY in .env to force one and re-ingest to compare results.
    chunk_strategy: Literal["auto", "header", "semantic"] = "auto"

    # --- Tracing ---
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "travel-assistant"
    # US default: https://api.smith.langchain.com
    # EU region:  https://eu.api.smith.langchain.com
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- Duffel (flights, used from phase 3) ---
    duffel_api_key: str = ""


# lru_cache makes this a singleton: the .env file is read once, then the same
# Settings object is returned on every later call (fast, and consistent).
@lru_cache
def get_settings() -> Settings:
    """Return the app settings (loaded once, then cached)."""
    return Settings()
