"""All money arithmetic lives here, in code. The model proposes unit prices; it never adds.

Intercity fares are distance heuristics, not quotes, and are labeled low-confidence in the UI.
"""

import math

from pydantic import BaseModel

from tripmate.schemas import Mode, Party, Plan, Segment, Tier, TripRequest

TIER_FRACTION = {Tier.budget: 0.6, Tier.balanced: 0.8, Tier.premium: 1.0}
ROAD_FACTOR = 1.3  # road/rail distance vs straight line
PER_KM = {Mode.bus: 1.8, Mode.train: 1.5}  # INR per person
CAR_PER_KM = 12.0  # INR per vehicle, up to 4 people
# Share of what is left after intercity transport, used as guidance caps in the prompt
SPLIT = {"stay": 0.45, "food": 0.25, "local": 0.10, "activities": 0.20}


class Budget(BaseModel):
    intercity: int
    local_transport: int
    stay: int
    food: int
    activities: int
    total: int
    target: int
    ceiling: int


def haversine_km(a: dict, b: dict) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["long"], b["lat"], b["long"]))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * math.asin(math.sqrt(h))


def leg_cost(req: TripRequest, a: str, b: str, places: dict) -> float:
    """One-way estimate for the whole party."""
    km = haversine_km(places[a], places[b])
    if req.mode == Mode.air:
        return (3000 + 2.5 * km) * req.travelers
    if req.mode == Mode.car:
        return CAR_PER_KM * km * ROAD_FACTOR * math.ceil(req.travelers / 4)
    return PER_KM[req.mode] * km * ROAD_FACTOR * req.travelers


def intercity_cost(req: TripRequest, route: list[Segment], places: dict) -> int:
    """Every leg of source -> city 1 -> ... -> city n -> source."""
    stops = [req.source, *(s.city for s in route), req.source]
    return round(sum(leg_cost(req, a, b, places) for a, b in zip(stops, stops[1:])) / 100) * 100


def rooms(req: TripRequest) -> int:
    return math.ceil(req.travelers / (3 if req.party == Party.family else 2))


def nights_by_city(route: list[Segment]) -> dict[str, int]:
    """You sleep in a city every day you are there, except the day you head home."""
    nights = {s.city: len(s.days) for s in route}
    if route:
        nights[route[-1].city] -= 1
    return nights


def target(req: TripRequest) -> int:
    return int(req.budget_inr * TIER_FRACTION[req.tier])


def envelope(req: TripRequest, route: list[Segment], places: dict) -> dict:
    """Pre-computed caps handed to the model so it never has to do division."""
    intercity = intercity_cost(req, route, places)
    remaining = target(req) - intercity
    total_nights = max(req.days - 1, 1)

    def cap(share: str, divisor: int) -> int:
        return max(int(remaining * SPLIT[share] / max(divisor, 1) / 50) * 50, 0)

    return {
        "intercity": intercity,
        "remaining": remaining,
        "feasible": remaining / (req.days * req.travelers) >= 500,
        "stay_per_night": cap("stay", rooms(req) * total_nights),
        "food_per_person_day": cap("food", req.travelers * req.days),
        "local_per_day": cap("local", req.days),
        "activities_per_person": cap("activities", req.travelers),
    }


def breakdown(req: TripRequest, route: list[Segment], plan: Plan, places: dict) -> Budget:
    intercity = intercity_cost(req, route, places)
    nights = nights_by_city(route)
    stay = food = local = 0
    for seg in route:
        costs = plan.cities.get(seg.city)
        if costs is None:
            continue
        stay += (costs.stay.per_night_inr if costs.stay else 0) * rooms(req) * nights[seg.city]
        food += (costs.food_per_day_inr or 0) * req.travelers * len(seg.days)
        local += (costs.local_transport_per_day_inr or 0) * len(seg.days)
    activities = sum(a.cost_inr for d in plan.days for a in d.activities) * req.travelers
    return Budget(
        intercity=intercity,
        local_transport=local,
        stay=stay,
        food=food,
        activities=activities,
        total=intercity + local + stay + food + activities,
        target=target(req),
        ceiling=req.budget_inr,
    )
