"""Deterministic checks on a plan. Each violation is a sentence the model can act on in a repair turn.

The same function gates the live agent, filters synthetic training data, and scores evals.
"""

import re

from tripmate.agent.budget import breakdown
from tripmate.schemas import Pace, Plan
from tripmate.state import TripState

MAX_PER_DAY = {Pace.relaxed: 3, Pace.moderate: 5, Pace.packed: 7}
# "I've added..." with no plan change tells the user something that did not happen.
CHANGE_CLAIM = re.compile(
    r"\b(i(?:'ve| have)? (?:added|swapped|replaced|moved|removed|switched|trimmed|lightened"
    r"|upgraded|booked)|i'll (?:add|swap|replace|move))\b",
    re.IGNORECASE,
)
DAY_START, DAY_END = 7 * 60, 22 * 60 + 30


def minutes(hhmm: str) -> int | None:
    m = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", hhmm.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def avoid_terms(constraints: list[str]) -> list[str]:
    """'no temples' / 'avoid museums' -> ['temple', 'museum'] (crude singular)."""
    terms = []
    for c in constraints:
        m = re.match(r"\s*(?:no|avoid)\s+(.+)", c, flags=re.I)
        if m:
            terms.append(re.sub(r"s$", "", m.group(1).strip().lower()))
    return terms


def validate(
    state: TripState, plan: Plan, places: dict, expected_days: set[int] | None = None
) -> list[str]:
    """Check `plan` against the request, route and known places in `state`.

    expected_days defaults to the whole trip; while a multi-city trip is still being planned
    city by city, pass only the days that should exist so far.
    """
    req, pois = state.request, state.pois
    expected = expected_days if expected_days is not None else set(range(1, req.days + 1))
    v: list[str] = []
    days = {d.day for d in plan.days}
    if days != expected:
        v.append(f"Plan must cover exactly days {sorted(expected)}, got {sorted(days)}.")

    for seg in state.route:
        if not expected & set(seg.days):
            continue
        costs = plan.cities.get(seg.city)
        if req.days > 1 and (costs is None or costs.stay is None):
            v.append(f"stay is missing for {seg.city}.")
        if (
            costs is None
            or costs.food_per_day_inr is None
            or costs.local_transport_per_day_inr is None
        ):
            v.append(
                f"food_per_day_inr and local_transport_per_day_inr must be set for {seg.city}."
            )

    avoid = avoid_terms(req.constraints)
    seen: set[str] = set()
    for d in plan.days:
        city = state.day_city(d.day)
        if not d.activities:
            v.append(f"Day {d.day} has no activities.")
        if len(d.activities) > MAX_PER_DAY[req.pace]:
            v.append(
                f"Day {d.day} has {len(d.activities)} activities; "
                f"max {MAX_PER_DAY[req.pace]} for a {req.pace.value} pace."
            )
        prev_end, prev_id = None, None
        for a in sorted(d.activities, key=lambda a: minutes(a.start) or 0):
            if a.poi not in pois:
                v.append(f"Day {d.day}: {a.poi} is not in PLACES.")
                continue
            if city and pois[a.poi].get("destination") != city:
                v.append(f"Day {d.day} is in {city}, but {a.poi} is in another city.")
            if a.poi in seen:
                v.append(f"{a.poi} is scheduled more than once.")
            seen.add(a.poi)
            text = (pois[a.poi]["name"] + " " + pois[a.poi]["text"]).lower()
            for term in avoid:
                if term in text:
                    v.append(f"Day {d.day}: {a.poi} conflicts with the constraint 'no {term}'.")
            start = minutes(a.start)
            if start is None:
                v.append(f"Day {d.day}: {a.poi} start '{a.start}' is not HH:MM.")
                continue
            if not 15 <= a.duration_min <= 600 or a.cost_inr < 0:
                v.append(f"Day {d.day}: {a.poi} has an implausible duration or cost.")
                continue
            end = start + a.duration_min
            if start < DAY_START or end > DAY_END:
                v.append(f"Day {d.day}: {a.poi} runs outside 07:00-22:30.")
            if prev_end is not None and start < prev_end:
                v.append(f"Day {d.day}: {a.poi} overlaps with {prev_id}.")
            prev_end, prev_id = end, a.poi
        ids = {a.poi for a in d.activities}
        if d.drop_if_short and d.drop_if_short not in ids:
            v.append(f"Day {d.day}: drop_if_short must be one of that day's activities or ''.")

    for d in plan.days:
        if d.add_if_extra and (d.add_if_extra not in pois or d.add_if_extra in seen):
            v.append(f"Day {d.day}: add_if_extra must be an unused id from PLACES or ''.")

    b = breakdown(req, state.route, plan, places)
    if b.total > b.target:
        v.append(
            f"Total cost ₹{b.total} exceeds the target ₹{b.target} "
            f"(stay ₹{b.stay}, food ₹{b.food}, local ₹{b.local_transport}, "
            f"activities ₹{b.activities}, intercity ₹{b.intercity}). Reduce costs."
        )
    return v
