# Ingestion: turn source documents into searchable vectors.
#
# Pipeline:  read file  ->  split into chunks  ->  embed + store in pgvector
#
# Run it with:  uv run python -m travel_assistant.rag.ingest
# It is safe to run repeatedly: it clears the collection first, so you never
# get duplicate chunks.
#
# Two splitting strategies are supported (see config.chunk_strategy):
#   - "header":   split structured markdown on its headings. Fast, predictable,
#                 keeps section metadata. Best for well-structured docs.
#   - "semantic": embed each sentence and cut where the MEANING shifts. Works on
#                 messy prose with no headings, at the cost of extra embedding
#                 calls. Best for scraped pages, PDFs, transcripts.

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from travel_assistant.config import Settings, get_settings
from travel_assistant.llm import get_embeddings
from travel_assistant.rag.store import get_vector_store

# Repo root: this file is .../src/travel_assistant/rag/ingest.py, so parents[3]
# climbs rag -> travel_assistant -> src -> repo root.
DATA_DIR = Path(__file__).parents[3] / "data"

# Each source document and the strategy that fits it best (used when
# CHUNK_STRATEGY=auto). The policy has clean headings; the tips are prose.
SOURCES: list[tuple[Path, str]] = [
    (DATA_DIR / "policy" / "acme_travel_policy.md", "header"),
    (DATA_DIR / "knowledge" / "travel_tips.md", "semantic"),
]

# Heading levels for the "header" strategy. The splitter attaches the heading
# text as metadata, so each chunk remembers which section it came from.
HEADERS_TO_SPLIT_ON = [
    ("#", "doc_title"),
    ("##", "section"),
    ("###", "subsection"),
]


def _split_header(text: str) -> list[Document]:
    # Structural split: one chunk per markdown section.
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep the heading text inside the chunk
    )
    return splitter.split_text(text)


def _split_semantic(text: str, settings: Settings) -> list[Document]:
    # Meaning-based split. SemanticChunker embeds sentences and starts a new
    # chunk wherever the embedding distance between consecutive sentences
    # exceeds a threshold (here: the 90th percentile of all the gaps it sees).
    # It uses the SAME embedding model as the store, which is what makes the
    # distances meaningful.
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(
        embeddings=get_embeddings(settings),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    return chunker.create_documents([text])


def load_and_split(
    path: Path, strategy: str, settings: Settings
) -> list[Document]:
    # Read one document and split it with the chosen strategy.
    # config.chunk_strategy can force a strategy; "auto" uses the per-source one.
    text = path.read_text(encoding="utf-8")

    chosen = settings.chunk_strategy if settings.chunk_strategy != "auto" else strategy

    if chosen == "semantic":
        chunks = _split_semantic(text, settings)
    else:
        chunks = _split_header(text)

    # Tag every chunk with its source file and the strategy used, so we can see
    # in the DB which document (and which splitter) each chunk came from.
    for chunk in chunks:
        chunk.metadata["source"] = path.name
        chunk.metadata["strategy"] = chosen
    return chunks


def ingest(settings: Settings | None = None) -> dict[str, int]:
    # Full ingestion of all SOURCES into a single pgvector collection.
    settings = settings or get_settings()
    store = get_vector_store(settings)

    # Idempotent: wipe previous vectors, then add the current ones.
    store.delete_collection()
    store.create_collection()

    counts: dict[str, int] = {}
    for path, default_strategy in SOURCES:
        chunks = load_and_split(path, default_strategy, settings)
        store.add_documents(chunks)
        counts[path.name] = len(chunks)
    return counts


if __name__ == "__main__":
    result = ingest()
    total = sum(result.values())
    for name, n in result.items():
        print(f"  {name}: {n} chunks")
    print(f"Ingested {total} chunks total into pgvector collection 'acme_travel_policy'.")
