"""What-if simulator (US2.3): try a different budget, length or pace on a copy of the trip.

Code decides the cheapest sufficient action. If the current plan still satisfies the new
numbers, nothing is sent to the model and the answer is instant; otherwise only the cities
whose days now break a rule are re-planned.
"""

import re
from dataclasses import dataclass

from tripmate.agent.orchestrator import Agent, TurnResult
from tripmate.agent.validator import validate
from tripmate.schemas import Pace, Segment
from tripmate.state import TripState, diff_plans


@dataclass
class WhatIf:
    state: TripState  # the simulated trip; the caller decides whether to adopt it
    summary: str
    changes: list[str]
    used_model: bool
    result: TurnResult | None = None


def resize_route(route: list[Segment], days: int) -> list[Segment]:
    """Add or remove days at the end of the trip, keeping at least one day per city."""
    current = sum(len(s.days) for s in route)
    lengths = [len(s.days) for s in route]
    lengths[-1] += max(days - current, 0)
    for _ in range(max(current - days, 0)):
        longest = max(range(len(lengths)), key=lambda i: (lengths[i], i))
        if lengths[longest] <= 1:
            raise ValueError(f"A {len(route)}-city trip needs at least {len(route)} days.")
        lengths[longest] -= 1
    resized, day = [], 1
    for seg, n in zip(route, lengths):
        resized.append(Segment(city=seg.city, days=list(range(day, day + n))))
        day += n
    return resized


def affected_segments(state: TripState, violations: list[str]) -> list[Segment]:
    """Cities whose days are named in a violation; trip-wide problems affect every city."""
    days: set[int] = set()
    for v in violations:
        if v.startswith("Plan must cover"):
            days |= {d for s in state.route for d in s.days} - state.planned_days()
        elif m := re.match(r"Day (\d+)", v):
            days.add(int(m.group(1)))
        else:
            return list(state.route)
    return [s for s in state.route if days & set(s.days)]


def what_if(
    agent: Agent,
    state: TripState,
    budget_pct: float = 0,
    days_delta: int = 0,
    pace: Pace | None = None,
) -> WhatIf:
    sim = state.model_copy(deep=True)
    req = sim.request
    asked = []
    if budget_pct:
        req.budget_inr = round(req.budget_inr * (1 + budget_pct / 100) / 100) * 100
        asked.append(f"budget ₹{req.budget_inr}")
    if days_delta:
        sim.route = resize_route(sim.route, req.days + days_delta)
        req.days += days_delta
        req.days_per_city = {s.city: len(s.days) for s in sim.route}
        asked.append(f"{req.days} days")
        if sim.plan:
            # Drop days that no longer exist. Days that moved to another city fail the
            # validator's city check below and get re-planned.
            sim.plan.days = [d for d in sim.plan.days if d.day <= req.days]
    if pace and pace != req.pace:
        req.pace = pace
        asked.append(f"{pace.value} pace")

    if sim.plan is None:
        return WhatIf(sim, "No plan yet to simulate against.", [], used_model=False)

    violations = validate(sim, sim.plan, agent.places)
    if not violations:
        b = agent.budget(sim)
        return WhatIf(
            sim,
            f"The current plan already fits {', '.join(asked) or 'these settings'}: "
            f"₹{b.total} of ₹{b.target}.",
            diff_plans(state.plan, sim.plan, sim.pois),
            used_model=False,
        )

    message = (
        f"What-if: the trip is now {', '.join(asked)}. Update the plan to fit. "
        "Problems to fix: " + " ".join(violations[:6])
    )
    result = agent.turn(sim, message, segments=affected_segments(sim, violations))
    # A re-plan that still breaks a check was not applied; say so instead of echoing the
    # model's claim that it adjusted the trip.
    summary = result.reply
    if result.violations:
        summary = "I could not make the plan fit these settings: " + " ".join(result.violations[:3])
    return WhatIf(
        sim,
        summary,
        diff_plans(state.plan, sim.plan, sim.pois),
        used_model=True,
        result=result,
    )
