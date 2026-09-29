# Duffel API client (flights).
#
# This is plain library code: it talks HTTP to Duffel and returns small, clean
# Python dicts. It knows NOTHING about MCP or agents, which is exactly why it is
# easy to test on its own and to reuse. The MCP server (a thin wrapper) and any
# tests both call these functions.
#
# We use Duffel TEST mode (key starts with duffel_test_): the data is fictional
# ("Duffel Airways"), no real money moves, and orders are sandbox orders.

import httpx

from travel_assistant.config import Settings, get_settings

DUFFEL_API = "https://api.duffel.com"
# Duffel requires a version header so the response shape never changes under us.
DUFFEL_VERSION = "v2"


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.duffel_api_key}",
        "Duffel-Version": DUFFEL_VERSION,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _summarise_slice(sl: dict) -> dict:
    # Reduce a Duffel "slice" (one direction of travel) to the essentials an
    # LLM and a user actually care about. The raw payload is huge; sending all
    # of it to the model would waste tokens and bury the useful bits.
    segments = sl.get("segments", [])
    first, last = segments[0], segments[-1]
    return {
        "origin": first["origin"]["iata_code"],
        "destination": last["destination"]["iata_code"],
        "departing_at": first["departing_at"],
        "arriving_at": last["arriving_at"],
        "duration": sl.get("duration"),  # ISO 8601, e.g. "PT8H2M"
        "stops": len(segments) - 1,
    }


def _summarise_offer(offer: dict) -> dict:
    return {
        "offer_id": offer["id"],
        "price": offer["total_amount"],
        "currency": offer["total_currency"],
        "airline": offer["owner"]["name"],
        "slices": [_summarise_slice(sl) for sl in offer["slices"]],
    }


def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    cabin_class: str = "economy",
    max_results: int = 5,
    settings: Settings | None = None,
) -> list[dict]:
    # Search Duffel for flights and return the cheapest `max_results` offers,
    # each trimmed to the essentials.
    #
    # Args:
    #   origin/destination: IATA codes, e.g. "LHR", "JFK".
    #   departure_date:     "YYYY-MM-DD".
    #   return_date:        "YYYY-MM-DD" for a round trip; omit for one-way.
    #   cabin_class:        economy | premium_economy | business | first.
    settings = settings or get_settings()

    # A "slice" is one leg. One slice = one-way; two slices = round trip.
    slices = [{"origin": origin, "destination": destination, "departure_date": departure_date}]
    if return_date:
        slices.append(
            {"origin": destination, "destination": origin, "departure_date": return_date}
        )

    body = {
        "data": {
            "slices": slices,
            "passengers": [{"type": "adult"}] * adults,
            "cabin_class": cabin_class,
        }
    }

    # return_offers=true makes Duffel include priced offers in the response, so
    # we don't need a second request to fetch them.
    resp = httpx.post(
        f"{DUFFEL_API}/air/offer_requests?return_offers=true",
        headers=_headers(settings),
        json=body,
        timeout=60,
    )
    resp.raise_for_status()

    offers = resp.json()["data"].get("offers", [])
    # Duffel usually returns offers cheapest-first, but sort to be certain.
    offers.sort(key=lambda o: float(o["total_amount"]))
    return [_summarise_offer(o) for o in offers[:max_results]]


def find_places(query: str, settings: Settings | None = None) -> list[dict]:
    # Resolve a city or airport name to its IATA code(s) using Duffel's Places
    # API. This exists because a small LLM cannot be trusted to recall IATA
    # codes correctly - e.g. it guessed "BEO" for Belgrade instead of "BEG".
    # The agent must call this to look codes up rather than inventing them.
    settings = settings or get_settings()

    resp = httpx.get(
        f"{DUFFEL_API}/places/suggestions",
        headers=_headers(settings),
        params={"query": query},
        timeout=30,
    )
    resp.raise_for_status()

    places = []
    for p in resp.json()["data"][:6]:
        places.append(
            {
                "type": p["type"],  # "city" or "airport"
                "iata_code": p.get("iata_code"),
                "name": p["name"],
                "city_name": p.get("city_name"),
            }
        )
    return places


def get_offer(offer_id: str, settings: Settings | None = None) -> dict:
    # Re-fetch a single offer to get its CURRENT price and the passenger slots
    # it needs. Offers expire after a few minutes, so this must be called right
    # before booking - both to confirm the price hasn't changed and to read the
    # passenger IDs that the order must reference.
    settings = settings or get_settings()

    resp = httpx.get(
        f"{DUFFEL_API}/air/offers/{offer_id}",
        headers=_headers(settings),
        timeout=60,
    )
    resp.raise_for_status()
    offer = resp.json()["data"]

    summary = _summarise_offer(offer)
    summary["expires_at"] = offer.get("expires_at")
    # Each passenger in the offer has an id the order must reuse. We expose the
    # ids and their types so the booking step can pair traveller details to them.
    summary["passenger_ids"] = [p["id"] for p in offer.get("passengers", [])]
    summary["passenger_types"] = [p.get("type") for p in offer.get("passengers", [])]
    return summary


def _summarise_order(order: dict) -> dict:
    # Trim a Duffel order to what a user cares about.
    return {
        "order_id": order["id"],
        "booking_reference": order.get("booking_reference"),  # the PNR
        "price": order.get("total_amount"),
        "currency": order.get("total_currency"),
        "passengers": [
            f"{p.get('given_name')} {p.get('family_name')}"
            for p in order.get("passengers", [])
        ],
        "cancelled_at": order.get("cancelled_at"),
    }


def create_order(
    offer_id: str,
    passengers: list[dict],
    settings: Settings | None = None,
) -> dict:
    # Book a flight (creates a Duffel order). In test mode this is a sandbox
    # order: it returns a real booking reference but no money moves.
    #
    # `passengers` is a list of traveller details, one per passenger in the
    # offer, each a dict with: given_name, family_name, born_on (YYYY-MM-DD),
    # title (mr/ms/mrs/miss/dr), gender (m/f), email, phone_number (E.164).
    #
    # We re-fetch the offer first (get_offer) so we book at the CURRENT price
    # and can attach the correct passenger IDs - this is the re-price-before-book
    # safety check.
    settings = settings or get_settings()

    offer = get_offer(offer_id, settings)
    ids = offer["passenger_ids"]
    if len(passengers) != len(ids):
        raise ValueError(
            f"Offer needs {len(ids)} passenger(s) but got {len(passengers)}."
        )

    # Pair each traveller's details with the offer's passenger id (by order).
    order_passengers = []
    for pax, pax_id in zip(passengers, ids, strict=True):
        order_passengers.append({**pax, "id": pax_id})

    body = {
        "data": {
            "type": "instant",
            "selected_offers": [offer_id],
            "passengers": order_passengers,
            # In test mode we pay from the Duffel test balance.
            "payments": [
                {
                    "type": "balance",
                    "amount": offer["price"],
                    "currency": offer["currency"],
                }
            ],
        }
    }

    resp = httpx.post(
        f"{DUFFEL_API}/air/orders",
        headers=_headers(settings),
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    return _summarise_order(resp.json()["data"])


def get_order(order_id: str, settings: Settings | None = None) -> dict:
    # Look up a previously created order by its id.
    settings = settings or get_settings()
    resp = httpx.get(
        f"{DUFFEL_API}/air/orders/{order_id}",
        headers=_headers(settings),
        timeout=60,
    )
    resp.raise_for_status()
    return _summarise_order(resp.json()["data"])


def cancel_order(order_id: str, settings: Settings | None = None) -> dict:
    # Cancel an order. Duffel does this in two steps: create a cancellation
    # (which quotes any refund), then confirm it. Returns the refund details.
    settings = settings or get_settings()
    headers = _headers(settings)

    # Step 1: create the cancellation quote.
    create = httpx.post(
        f"{DUFFEL_API}/air/order_cancellations",
        headers=headers,
        json={"data": {"order_id": order_id}},
        timeout=60,
    )
    create.raise_for_status()
    cancellation = create.json()["data"]

    # Step 2: confirm it.
    confirm = httpx.post(
        f"{DUFFEL_API}/air/order_cancellations/{cancellation['id']}/actions/confirm",
        headers=headers,
        timeout=60,
    )
    confirm.raise_for_status()
    result = confirm.json()["data"]
    return {
        "order_id": order_id,
        "cancellation_id": result["id"],
        "refund_amount": result.get("refund_amount"),
        "refund_currency": result.get("refund_currency"),
        "confirmed_at": result.get("confirmed_at"),
    }
