# MCP client: launches our MCP server(s) and exposes their tools as LangChain
# tools the agent can use.
#
# langchain-mcp-adapters' MultiServerMCPClient starts each configured server as
# a subprocess (stdio transport) and converts every MCP tool into a LangChain
# StructuredTool. The agent then binds these like any other tool.
#
# We also split the flight tools into read-only vs write. Read-only tools are
# safe for the agent to call freely; write tools (book/cancel) must go through
# the human-approval gate (added in step 3.5), so the agent should not get them
# directly.

import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from travel_assistant.config import Settings, get_settings

# Repo root: this file is src/travel_assistant/mcp_client.py -> parents[2].
REPO_ROOT = Path(__file__).parents[2]
DUFFEL_SERVER = REPO_ROOT / "mcp_servers" / "duffel_server" / "server.py"

# Which flight tools change state (need approval) vs are safe to read.
READ_ONLY_FLIGHT_TOOLS = {"find_airport", "search_flights", "get_offer", "get_order"}
WRITE_FLIGHT_TOOLS = {"book_flight", "cancel_flight"}


def get_mcp_client(settings: Settings | None = None) -> MultiServerMCPClient:
    # Build the multi-server client. Each server runs as a subprocess.
    # We set cwd to the repo root and pass the Duffel key explicitly so the
    # subprocess can authenticate no matter where the app was started from.
    settings = settings or get_settings()

    return MultiServerMCPClient(
        {
            "duffel": {
                "command": sys.executable,
                "args": [str(DUFFEL_SERVER)],
                "transport": "stdio",
                "cwd": str(REPO_ROOT),
                "env": {"DUFFEL_API_KEY": settings.duffel_api_key},
            }
        }
    )


async def get_flight_tools(
    include_write: bool = False, settings: Settings | None = None
) -> list[BaseTool]:
    # Return the flight tools as LangChain tools.
    #   include_write=False (default): only read-only tools (safe for the agent).
    #   include_write=True: also book/cancel (used by the approval-gated path).
    settings = settings or get_settings()
    client = get_mcp_client(settings)
    all_tools = await client.get_tools(server_name="duffel")

    allowed = READ_ONLY_FLIGHT_TOOLS | (WRITE_FLIGHT_TOOLS if include_write else set())
    return [t for t in all_tools if t.name in allowed]
