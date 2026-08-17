"""
mcp/tools.py
------------
The actual TOOLS an agent can call. Each function is deliberately:
  - typed (so Gemini can build a JSON schema from it automatically)
  - documented with a docstring (Gemini reads this to know when/how to call it)
  - deterministic and offline (no real APIs, no API keys needed, no flaky
    network calls) so this demo runs the same way every time.

Swap any of these for a real API (Amadeus, Skyscanner, OpenWeather, etc.)
without changing anything else in the project -- that's the whole point of
having a tool layer.
"""
import hashlib


def _seed(*parts: str) -> int:
    """Turn text into a stable number so 'random' results are repeatable."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)


def search_flights(origin: str, destination: str, date: str, passengers: int = 1) -> dict:
    """Search round-trip flight options between two cities.

    Args:
        origin: departure city or airport code, e.g. "Mumbai".
        destination: arrival city or airport code, e.g. "Goa".
        date: travel date as YYYY-MM-DD.
        passengers: number of travelers.

    Returns:
        A dict with a list of flight options (airline, price in INR, duration).
    """
    s = _seed(origin, destination, date)
    airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara"]
    options = []
    for i in range(3):
        price = 2800 + ((s >> (i * 4)) % 4000) + i * 350
        options.append({
            "airline": airlines[(s + i) % len(airlines)],
            "price_inr_per_person": price,
            "total_price_inr": price * passengers,
            "duration_hours": round(1.4 + (i * 0.35), 1),
            "stops": 0 if i == 0 else 1,
        })
    return {"origin": origin, "destination": destination, "date": date,
            "passengers": passengers, "options": options}


def search_hotels(destination: str, checkin: str, checkout: str,
                   guests: int = 2, max_price_per_night_inr: int = 10000) -> dict:
    """Search hotels in a destination city within a nightly budget.

    Args:
        destination: city to stay in, e.g. "Goa".
        checkin: check-in date YYYY-MM-DD.
        checkout: check-out date YYYY-MM-DD.
        guests: number of guests.
        max_price_per_night_inr: don't return hotels pricier than this per night.

    Returns:
        A dict with a list of hotel options (name, rating, price per night).
    """
    s = _seed(destination, checkin, checkout)
    names = ["Beachside Cove Resort", "Palm Grove Inn", "Sunset Sands Hotel",
             "Old Town Boutique Stay"]
    options = []
    for i, name in enumerate(names):
        price = 2200 + ((s >> (i * 5)) % 6000)
        if price <= max_price_per_night_inr:
            options.append({
                "name": name,
                "rating": round(3.6 + ((s >> (i * 3)) % 14) / 10, 1),
                "price_per_night_inr": price,
                "distance_from_beach_km": round(0.2 + i * 0.6, 1),
            })
    return {"destination": destination, "checkin": checkin, "checkout": checkout,
            "guests": guests, "options": options or ["No hotels under budget found"]}


def get_weather(destination: str, date: str) -> dict:
    """Get an expected weather forecast for a destination and date.

    Args:
        destination: city name, e.g. "Goa".
        date: date as YYYY-MM-DD.

    Returns:
        A dict with condition, temperature range in Celsius, and rain chance.
    """
    s = _seed(destination, date)
    conditions = ["Sunny", "Partly cloudy", "Light showers", "Humid and clear"]
    return {
        "destination": destination, "date": date,
        "condition": conditions[s % len(conditions)],
        "temp_low_c": 22 + (s % 4),
        "temp_high_c": 30 + (s % 5),
        "rain_chance_pct": s % 40,
    }


def get_attractions(destination: str, interests: str = "") -> dict:
    """Look up top attractions/activities in a destination, optionally
    filtered by traveler interests.

    Args:
        destination: city name, e.g. "Goa".
        interests: comma-separated interests, e.g. "beaches,nightlife,seafood".

    Returns:
        A dict with a list of attractions tagged by category.
    """
    catalog = [
        {"name": "Baga Beach", "category": "beaches", "note": "Popular sunset spot"},
        {"name": "Anjuna Flea Market", "category": "shopping", "note": "Sat market, local crafts"},
        {"name": "Tito's Lane", "category": "nightlife", "note": "Clubs and live music"},
        {"name": "Fontainhas Latin Quarter", "category": "culture", "note": "Colourful old streets"},
        {"name": "Britto's Shack", "category": "seafood", "note": "Beachfront seafood grill"},
        {"name": "Dudhsagar Falls", "category": "nature", "note": "Half-day trip, waterfalls"},
        {"name": "Chapora Fort", "category": "sightseeing", "note": "Free, great viewpoint"},
    ]
    wanted = {i.strip().lower() for i in interests.split(",") if i.strip()}
    if wanted:
        filtered = [a for a in catalog if a["category"] in wanted]
    else:
        filtered = catalog
    return {"destination": destination, "attractions": filtered or catalog[:3]}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between currencies using fixed demo exchange rates.

    Args:
        amount: amount to convert.
        from_currency: 3-letter currency code, e.g. "INR".
        to_currency: 3-letter currency code, e.g. "USD".

    Returns:
        A dict with the converted amount and rate used.
    """
    rates_to_usd = {"INR": 1 / 83.0, "USD": 1.0, "EUR": 1.08, "GBP": 1.27}
    from_currency, to_currency = from_currency.upper(), to_currency.upper()
    if from_currency not in rates_to_usd or to_currency not in rates_to_usd:
        return {"error": f"Unsupported currency pair {from_currency}->{to_currency}"}
    usd = amount * rates_to_usd[from_currency]
    converted = usd / rates_to_usd[to_currency]
    return {"amount": amount, "from_currency": from_currency, "to_currency": to_currency,
            "converted_amount": round(converted, 2)}


def calculate_budget(flights_total_inr: float, hotel_per_night_inr: float,
                      nights: int, daily_spend_inr: float, days: int) -> dict:
    """Add up a rough trip budget in INR.

    Args:
        flights_total_inr: total cost of flights for all travelers.
        hotel_per_night_inr: hotel cost per night.
        nights: number of nights staying.
        daily_spend_inr: estimated food/activities/transport spend per day.
        days: number of days of the trip.

    Returns:
        A dict with a line-item breakdown and grand total in INR.
    """
    hotel_total = hotel_per_night_inr * nights
    spend_total = daily_spend_inr * days
    grand_total = flights_total_inr + hotel_total + spend_total
    return {
        "flights_inr": flights_total_inr,
        "hotel_inr": hotel_total,
        "daily_spend_inr": spend_total,
        "grand_total_inr": grand_total,
    }


# Every tool the agents are allowed to use, in one place.
ALL_TOOLS = [search_flights, search_hotels, get_weather, get_attractions,
             convert_currency, calculate_budget]
