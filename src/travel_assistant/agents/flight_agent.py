# The flight agent.
#
# This is a ReAct agent: a loop where the LLM can either answer, or decide to
# call a tool, see the result, and think again - repeating until it has an
# answer. LangGraph's prebuilt create_react_agent builds that loop for us as a
# small graph (agent node <-> tools node).
#
# For now the agent gets only READ-ONLY tools (search / look up). Booking is
# withheld until the human-approval gate is in place (step 3.5).
#
# Chat with it:  uv run python -m travel_assistant.agents.flight_agent

import asyncio
from datetime import date

from langgraph.prebuilt import create_react_agent

from travel_assistant.agents.approval import wrap_with_approval
from travel_assistant.config import Settings, get_settings
from travel_assistant.llm import get_llm
from travel_assistant.mcp_client import (
    WRITE_FLIGHT_TOOLS,
    get_flight_tools,
)
from travel_assistant.runtime import configure_event_loop

# The system prompt is the agent's job description. Being explicit keeps a small
# local model on track: what it does, what data it needs, and its limits.
SYSTEM_PROMPT = """You are a flight search assistant for Acme Corp travellers.

Today's date is {today}. Interpret relative dates ("next Friday") accordingly.

How to work:
- Flights are searched by airport IATA code. NEVER guess or invent IATA codes.
  For every city or airport name the user gives, first call find_airport to look
  up the correct code, then use that code. (Guessing leads to booking the wrong
  city - e.g. Belgrade is BEG, not BEO.)
- When find_airport returns a city plus several airports, use the city/main
  airport code and mention which airport you chose.
- If a required detail is missing (origin, destination, or date), ask a short
  clarifying question instead of guessing.
- Assume 1 adult in economy unless the user says otherwise. Do NOT ask about
  passenger count or cabin class - just search with those defaults.
- Use the search_flights tool to find options. Present the best few clearly:
  price, airline, times, and number of stops.
- You can look up an offer (get_offer) or an existing order (get_order).
{booking_clause}"""

# Two variants of the last line, depending on whether booking is enabled.
NO_BOOKING_CLAUSE = (
    "- You CANNOT book or cancel. If asked to book, explain that booking is not "
    "enabled in this mode."
)
BOOKING_CLAUSE = (
    "- To book (book_flight) you need a specific offer_id and full passenger "
    "details: given_name, family_name, born_on (YYYY-MM-DD), title (mr/ms/mrs/"
    "miss/dr), gender (m/f), email, phone_number (E.164, e.g. +442080160508). "
    "Collect anything missing before calling book_flight. Every booking or "
    "cancellation is paused for human approval, so call the tool only once you "
    "have all details."
)


async def build_flight_agent(
    settings: Settings | None = None,
    checkpointer=None,
    include_booking: bool = False,
):
    # Assemble the agent: model + flight tools + system prompt.
    # Async because loading MCP tools launches the server subprocess.
    #
    #   include_booking=False -> read-only tools only (safe, no checkpointer needed).
    #   include_booking=True  -> also book/cancel, each wrapped in the approval
    #                            gate. Requires a checkpointer so interrupt() can
    #                            pause and resume.
    settings = settings or get_settings()

    if include_booking:
        all_tools = await get_flight_tools(include_write=True, settings=settings)
        read_tools = [t for t in all_tools if t.name not in WRITE_FLIGHT_TOOLS]
        write_tools = [t for t in all_tools if t.name in WRITE_FLIGHT_TOOLS]
        # Wrap the write tools so they pause for approval before executing.
        tools = read_tools + wrap_with_approval(write_tools)
        booking_clause = BOOKING_CLAUSE
    else:
        tools = await get_flight_tools(include_write=False, settings=settings)
        booking_clause = NO_BOOKING_CLAUSE

    llm = get_llm(settings)
    prompt = SYSTEM_PROMPT.format(today=date.today().isoformat(), booking_clause=booking_clause)

    # create_react_agent returns a compiled LangGraph graph. The checkpointer
    # lets it pause at interrupts and resume later on the same thread_id.
    return create_react_agent(llm, tools, prompt=prompt, checkpointer=checkpointer)


def _print_approval_request(payload: dict) -> None:
    # Show the human what they are being asked to approve.
    print("\n" + "!" * 60)
    print(f"APPROVAL REQUIRED: {payload['tool']}")
    for key, value in payload["args"].items():
        print(f"  {key}: {value}")
    print("!" * 60)


async def _cli() -> None:
    # Interactive chat loop WITH booking + human approval. Uses a Postgres
    # checkpointer and one thread_id so the whole conversation (and any paused
    # approval) persists. Type 'quit' to exit.
    from langgraph.types import Command

    from travel_assistant.agents.checkpoint import open_checkpointer

    config = {"configurable": {"thread_id": "flight-cli-session"}}

    async with open_checkpointer() as saver:
        agent = await build_flight_agent(checkpointer=saver, include_booking=True)
        print("Flight agent (search + booking). Try 'flights from Berlin to Rome")
        print("on 2026-11-20', then ask to book one. Type 'quit' to exit.\n")

        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in {"quit", "exit", "q", ""}:
                break

            # Send the user's turn; then keep resuming while the graph pauses
            # for approvals (there can be more than one in a turn).
            result = await agent.ainvoke({"messages": [("user", user)]}, config)
            while "__interrupt__" in result:
                _print_approval_request(result["__interrupt__"][0].value)
                answer = input("approve this action? [y/N] ").strip().lower()
                approved = answer in {"y", "yes"}
                result = await agent.ainvoke(Command(resume={"approved": approved}), config)

            print(f"\nassistant> {result['messages'][-1].content}\n")


if __name__ == "__main__":
    configure_event_loop()
    asyncio.run(_cli())
