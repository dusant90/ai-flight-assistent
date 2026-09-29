# Generation: the "G" in RAG.
#
# We take the chunks the hybrid retriever found, put them into the prompt as
# CONTEXT, and ask the LLM to answer using ONLY that context. This is what
# "grounding" means: the model must base its answer on retrieved facts, not on
# whatever it happens to remember. It keeps answers accurate and lets us cite
# exactly which policy section a rule came from.

import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from travel_assistant.config import Settings, get_settings
from travel_assistant.llm import get_llm
from travel_assistant.rag.retriever import hybrid_search

# The instruction that enforces grounding. The rules matter: without them a
# small model will happily invent plausible-sounding policy.
SYSTEM_PROMPT = """You are the Acme Corp Travel Assistant. Answer the user's \
question using ONLY the CONTEXT below.

Rules:
- If the answer is not in the context, say "That isn't covered in the travel \
policy or travel tips." Do not guess.
- Be concise and specific. Quote the exact numbers or rules that apply.
- Cite the section you used in parentheses, e.g. (Section 2.1 Cabin Class).

CONTEXT:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_context(docs: list[Document]) -> str:
    # Turn retrieved chunks into a labelled block the model can cite from.
    blocks = []
    for doc in docs:
        # Prefer the most specific label available in the chunk's metadata.
        label = (
            doc.metadata.get("subsection")
            or doc.metadata.get("section")
            or doc.metadata.get("source", "unknown")
        )
        blocks.append(f"[{label}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def _strip_think(text: str) -> str:
    # qwen3 is a reasoning model that may wrap internal reasoning in
    # <think>...</think>. Strip it so the user sees only the final answer.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@traceable(run_type="chain", name="rag_answer")
def answer_question(query: str, settings: Settings | None = None) -> dict:
    # Full RAG: retrieve -> ground -> generate. Returns the answer plus the
    # sources used, so the caller can show where the answer came from.
    settings = settings or get_settings()

    docs = hybrid_search(query, k=5, top_n=4, weights=(0.5, 0.5), settings=settings)
    context = _format_context(docs)

    chain = PROMPT | get_llm(settings)
    response = chain.invoke({"context": context, "question": query})

    sources = sorted(
        {
            doc.metadata.get("section")
            or doc.metadata.get("source", "unknown")
            for doc in docs
        }
    )
    return {"answer": _strip_think(response.content), "sources": sources}


def _cli() -> None:
    # A tiny interactive loop so you can ask the policy questions from the
    # terminal. Type 'quit' to exit.
    print("Acme Travel Assistant (RAG). Ask a policy question. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"quit", "exit", "q", ""}:
            break
        result = answer_question(query)
        print(f"\nassistant> {result['answer']}")
        print(f"[sources: {', '.join(result['sources'])}]\n")


if __name__ == "__main__":
    _cli()
