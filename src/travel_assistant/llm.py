# """LLM factory.

# The rest of the app should NOT care whether we're talking to a local Ollama
# model or the OpenAI API. It just calls `get_llm()` and gets back a chat model
# that follows LangChain's standard interface (`.invoke()`, `.bind_tools()`,
# streaming, etc.).

# Switching providers is then a one-line change in `.env`:
#     LLM_PROVIDER=ollama   ->   LLM_PROVIDER=openai

# This is the classic "factory" pattern: one function decides which concrete
# object to build, based on configuration, and returns a common type.
# """

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from travel_assistant.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model based on the configured provider.

    Args:
        settings: optional Settings; defaults to the cached app settings.
                  (Being able to pass settings in makes testing easier.)

    Returns:
        A LangChain chat model (Ollama or OpenAI).
    """
    settings = settings or get_settings()

    if settings.llm_provider == "ollama":
        # Imported inside the branch so the app doesn't require the OpenAI
        # package to be configured just to run locally, and vice versa.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,  # low = more deterministic, better for tool use
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Add your key to .env."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

    # Should be unreachable because Settings validates the Literal, but we fail
    # loudly just in case.
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    # Returns the embedding model: it turns a piece of text into a list of
    # numbers (a "vector") that captures its meaning. Two texts with similar
    # meaning get vectors that are close together, which is how RAG finds the
    # policy chunk most relevant to a question.
    #
    # We always use a LOCAL Ollama embedding model (nomic-embed-text): it is
    # free, fast enough on CPU for a handful of documents, and keeps embeddings
    # consistent regardless of which chat provider is selected. Mixing embedding
    # models is dangerous: vectors created by model A are not comparable to
    # vectors created by model B, so the whole store must use ONE model.
    settings = settings or get_settings()

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=settings.ollama_base_url,
    )
