# Connection to the pgvector store.
#
# Both the ingestion script (which WRITES chunks) and the retriever (which
# READS them) need a handle to the same vector store. We define that in one
# place so they can't drift apart (same collection name, same embeddings).

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from travel_assistant.config import Settings, get_settings
from travel_assistant.llm import get_embeddings

# A "collection" is like a named table inside pgvector. Keeping the name in one
# constant means ingest and retrieve always point at the same set of vectors.
COLLECTION_NAME = "acme_travel_policy"


def get_vector_store(settings: Settings | None = None) -> PGVector:
    # Build (or connect to) the pgvector store for our policy collection.
    #
    # PGVector automatically creates the needed tables on first use. It stores,
    # for each chunk: the original text, its embedding vector, and any metadata
    # (like which heading the chunk came from).
    settings = settings or get_settings()

    return PGVector(
        embeddings=get_embeddings(settings),
        collection_name=COLLECTION_NAME,
        connection=settings.database_url,
        use_jsonb=True,  # store chunk metadata as JSONB (queryable, efficient)
    )


def load_all_chunks(settings: Settings | None = None) -> list[Document]:
    # Read every chunk (text + metadata) for our collection straight from the
    # DB. The BM25 keyword retriever needs the whole corpus in memory to compute
    # term statistics, and reading it from the DB keeps ONE source of truth:
    # BM25 and the vector search rank the exact same chunks.
    #
    # We deliberately select only `document` and `cmetadata` (not the `embedding`
    # vector column), so plain SQL is enough and we avoid the vector type.
    settings = settings or get_settings()

    engine = create_engine(settings.database_url)
    sql = text(
        """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :name
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"name": COLLECTION_NAME}).fetchall()
    engine.dispose()

    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in rows
    ]
