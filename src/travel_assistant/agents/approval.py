# Human-approval wrapper for sensitive tools.
#
# We take a write tool (book_flight / cancel_flight) and wrap it so that, when
# the agent calls it, the graph PAUSES via interrupt() and waits for a human
# decision. Only on explicit approval does the real tool run.
#
# Why this is safe: interrupt() is a code-level halt inside the graph, not a
# prompt instruction. The LLM cannot "decide" to skip it - execution literally
# stops until the app resumes the graph with the human's answer. This is the
# deterministic gate that sits on top of the (bypassable) prompt-level hints.
#
# Note on interrupt() semantics: when the graph resumes, the wrapped function
# re-runs FROM THE TOP, and interrupt() returns the resume value instead of
# pausing again. So we must not perform side effects before interrupt() - we
# only build a summary (safe), then interrupt, then act.

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt


def _wrap(underlying: BaseTool) -> BaseTool:
    async def _run(**kwargs) -> str:
        # Pause and ask the human. The dict we pass is what the UI/CLI shows;
        # the graph stays paused (persisted in the checkpointer) until resumed.
        decision = interrupt(
            {
                "type": "approval_required",
                "tool": underlying.name,
                "args": kwargs,
            }
        )

        # The app resumes with Command(resume=...). Accept either {"approved":
        # bool} or a plain truthy/falsey value.
        approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)

        if not approved:
            return f"The user did NOT approve {underlying.name}. No action was taken."

        # Approved: run the real tool exactly once.
        return await underlying.ainvoke(kwargs)

    # Reuse the underlying tool's name, description and args schema so the LLM
    # sees an identical tool - the approval step is invisible to the model.
    return StructuredTool.from_function(
        coroutine=_run,
        name=underlying.name,
        description=underlying.description,
        args_schema=underlying.args_schema,
    )


def wrap_with_approval(tools: list[BaseTool]) -> list[BaseTool]:
    # Wrap every given tool with the approval gate.
    return [_wrap(t) for t in tools]
