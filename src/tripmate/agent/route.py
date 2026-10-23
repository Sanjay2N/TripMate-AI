"""Multi-city routing (US2.4), done in code: visiting order and how many days each city gets."""

from itertools import permutations

from tripmate.agent.budget import haversine_km
from tripmate.schemas import Segment, TripRequest


def order_cities(source: str, cities: list[str], places: dict) -> list[str]:
    """Shortest loop source -> cities -> source. Exhaustive, which is fine for a handful of cities."""
    if len(cities) <= 1:
        return list(cities)

    def loop_km(order) -> float:
        stops = [source, *order, source]
        return sum(haversine_km(places[a], places[b]) for a, b in zip(stops, stops[1:]))

    return list(min(permutations(cities), key=loop_km))


def split_days(cities: list[str], days: int, weights: dict[str, float]) -> dict[str, int]:
    """Every city gets a day; the rest go one at a time to the city with the most left to see."""
    if days < len(cities):
        raise ValueError(f"{len(cities)} cities need at least {len(cities)} days, got {days}.")
    alloc = {c: 1 for c in cities}
    for _ in range(days - len(cities)):
        best = max(cities, key=lambda c: weights.get(c, 1) / (alloc[c] + 1))
        alloc[best] += 1
    return alloc


def build_route(req: TripRequest, places: dict, weights: dict[str, float]) -> list[Segment]:
    cities = order_cities(req.source, list(dict.fromkeys(req.destinations)), places)
    fixed = req.days_per_city
    if set(fixed) == set(cities) and sum(fixed.values()) == req.days and min(fixed.values()) >= 1:
        alloc = fixed
    else:
        alloc = split_days(cities, req.days, weights)
    route, day = [], 1
    for city in cities:
        route.append(Segment(city=city, days=list(range(day, day + alloc[city]))))
        day += alloc[city]
    return route
