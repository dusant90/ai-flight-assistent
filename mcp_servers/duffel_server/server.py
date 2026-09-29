# Duffel MCP server.
#
# This turns our Duffel client into an MCP server: a small program that exposes
# "tools" over the Model Context Protocol. Any MCP client (our LangGraph agent,
# Claude Desktop, the MCP Inspector) can connect, discover the tools, and call
# them - without knowing anything about Duffel or HTTP.
#
# FastMCP (from the official `mcp` SDK) does the protocol work. We just write
# normal Python functions and decorate them with @mcp.tool(). The function's
# name, type hints and docstring become the tool's schema that the LLM reads to
# decide how to call it - so the docstring here is effectively a prompt.
#
# Run it standalone (stdio transport):
#   uv run python mcp_servers/duffel_server/server.py

# FastMCP (mcp 1.x) does the protocol work. We pin mcp<2 because the
# langchain-mcp-adapters bridge does not yet support the mcp 2.x rename.
from mcp.server.fastmcp import FastMCP

from travel_assistant.flights import duffel_client as duffel

mcp = FastMCP("duffel-flights")


@mcp.tool()
def find_airport(query: str) -> list[dict]:
    """Look up the IATA code for a city or airport by name.

    ALWAYS use this to convert a city or airport name into an IATA code before
    searching flights. Do not guess IATA codes yourself. Returns matching places
    with their type (city/airport), iata_code, and full name.

    Args:
        query: A city or airport name, e.g. "Belgrade" or "Oslo".
    """
    return duffel.find_places(query)


@mcp.tool()
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    cabin_class: str = "economy",
    max_results: int = 5,
) -> list[dict]:
    """Search for flights and return the cheapest offers.

    Use this to find flight options for a trip. Returns a list of offers, each
    with an offer_id (needed later to book), price, airline, and the flight
    times for each leg.

    Args:
        origin: Departure airport IATA code, e.g. "LHR".
        destination: Arrival airport IATA code, e.g. "JFK".
        departure_date: Outbound date in YYYY-MM-DD format.
        return_date: Return date YYYY-MM-DD for a round trip; omit for one-way.
        adults: Number of adult passengers.
        cabin_class: One of economy, premium_economy, business, first.
        max_results: Maximum number of offers to return.
    """
    return duffel.search_flights(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        adults=adults,
        cabin_class=cabin_class,
        max_results=max_results,
    )


@mcp.tool()
def get_offer(offer_id: str) -> dict:
    """Re-check a specific flight offer before booking.

    Offers expire after a few minutes. Call this right before booking to get the
    CURRENT price and the number of passengers the offer expects. Returns the
    offer with its price, expiry time, and passenger slots.

    Args:
        offer_id: The offer_id from a previous search_flights result.
    """
    return duffel.get_offer(offer_id)


@mcp.tool()
def book_flight(offer_id: str, passengers: list[dict]) -> dict:
    """Book a flight offer (creates a confirmed order). TEST MODE: no real money.

    Only call this after the user has explicitly approved the specific offer.
    Re-prices the offer first, then books it. Returns the order with its booking
    reference (PNR).

    Args:
        offer_id: The offer_id to book.
        passengers: One dict per passenger, each with: given_name, family_name,
            born_on (YYYY-MM-DD), title (mr/ms/mrs/miss/dr), gender (m/f),
            email, phone_number (E.164 format, e.g. "+442080160508").
    """
    return duffel.create_order(offer_id, passengers)


@mcp.tool()
def get_order(order_id: str) -> dict:
    """Look up an existing flight booking by its order id.

    Args:
        order_id: The order_id returned by book_flight.
    """
    return duffel.get_order(order_id)


@mcp.tool()
def cancel_flight(order_id: str) -> dict:
    """Cancel a flight booking and get the refund details.

    Only call this after the user has explicitly asked to cancel.

    Args:
        order_id: The order_id of the booking to cancel.
    """
    return duffel.cancel_order(order_id)


if __name__ == "__main__":
    # stdio transport: the server talks over stdin/stdout, which is how a local
    # client launches and speaks to it as a subprocess.
    mcp.run()
