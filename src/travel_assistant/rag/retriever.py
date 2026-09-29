# Hybrid retriever: semantic (vector) + keyword (BM25), fused with Reciprocal
# Rank Fusion (RRF).
#
# Why hybrid:
#   - Vector search matches MEANING. Great for paraphrases ("lie-flat seat" ->
#     business class), weak on exact tokens ("USD 600", "First Class").
#   - BM25 matches EXACT TERMS. Great for codes/numbers/rare words, blind to
#     synonyms.
#   Running both and fusing their rankings is robust to either kind of query.
#
# RRF combines ranked lists without needing the raw scores to be comparable:
# a document's fused score is the sum, over each list it appears in, of
#     weight / (k_rrf + rank)
# where rank is its 1-based position in that list. Documents near the top of
# either list bubble up; documents near the top of BOTH win.

from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langsmith import traceable

from travel_assistant.config import Settings, get_settings
from travel_assistant.rag.store import get_vector_store, load_all_chunks

# Constant in the RRF formula. 60 is the value from the original RRF paper; it
# softens the impact of the exact rank so lists blend smoothly.
RRF_K = 60


def get_vector_retriever(k: int = 5, settings: Settings | None = None):
    # Semantic retriever: embeds the query and finds the k nearest chunks in
    # pgvector by vector distance.
    settings = settings or get_settings()
    store = get_vector_store(settings)
    return store.as_retriever(search_kwargs={"k": k})


def get_bm25_retriever(k: int = 5, settings: Settings | None = None) -> BM25Retriever:
    # Keyword retriever: builds Okapi BM25 over the whole corpus (read from the
    # DB) and returns the k best chunks by term overlap with the query.
    settings = settings or get_settings()
    retriever = BM25Retriever.from_documents(load_all_chunks(settings))
    retriever.k = k
    return retriever


@traceable(name="reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    top_n: int,
) -> list[Document]:
    # Fuse several ranked lists of Documents into one, using weighted RRF.
    # Documents are identified by their text content (chunks are unique).
    scores: dict[str, float] = defaultdict(float)
    by_key: dict[str, Document] = {}

    for docs, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            scores[key] += weight / (RRF_K + rank)
            by_key[key] = doc  # remember one Document object per unique chunk

    # Highest fused score first.
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [by_key[key] for key in ordered[:top_n]]


@traceable(run_type="retriever", name="hybrid_search")
def hybrid_search(
    query: str,
    k: int = 5,
    top_n: int = 4,
    weights: tuple[float, float] = (0.5, 0.5),
    settings: Settings | None = None,
) -> list[Document]:
    # Run both retrievers and fuse their results.
    #   weights = (bm25_weight, vector_weight)
    settings = settings or get_settings()

    bm25_docs = get_bm25_retriever(k, settings).invoke(query)
    vector_docs = get_vector_retriever(k, settings).invoke(query)

    return reciprocal_rank_fusion(
        ranked_lists=[bm25_docs, vector_docs],
        weights=[weights[0], weights[1]],
        top_n=top_n,
    )
