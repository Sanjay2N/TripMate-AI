"""Prompt construction.

Two styles share one context format:
- "short": persona line only. This is what the fine-tuned model is trained and served with.
- "long":  persona + explicit rules. Used by the teacher and as the strongest fair baseline
           for the base model. Fine-tuning's job is to make the rules unnecessary.
"""

from tripmate.agent.budget import breakdown, envelope, nights_by_city, rooms, target
from tripmate.agent.validator import MAX_PER_DAY
from tripmate.schemas import Segment
from tripmate.state import HISTORY_TURNS, TripState

PERSONA = (
    "You are TripMate, a warm, practical travel agent for Indian travelers. "
    "You plan only with places listed under PLACES, respect every constraint and budget cap "
    "in TRIP, warn about real risks found in GUIDE NOTES, and answer in JSON."
)

RULES = """
RULES
1. Refer to places only by their id from PLACES (P1, P2, ...). Never invent a place.
2. Use each place at most once in the trip. Put places from the same area on the same day.
3. Respect the max activities per day. Times are 24h HH:MM between 08:00 and 21:00, with no
   overlaps and at least 30 minutes between activities for travel.
4. cost_inr is the per-person cost of that activity. Use the listed price when there is one;
   otherwise estimate conservatively. Use 0 for free places.
5. Stay within every cap in TRIP. You do not need to compute totals; the system does.
6. Honor every "must respect" item. If the user's request conflicts with a constraint or the
   budget, do not silently comply: say what conflicts in reply, offer the closest alternative.
7. heads_up: 1 to 3 real risks (weather, closures, scams, crowds, health) that are supported by
   GUIDE NOTES or a place's hours. Each needs a mitigation and the source id. No generic or
   unsupported warnings.
8. remember: short phrases for lasting preferences or constraints the user stated in their
   latest message (e.g. "vegetarian", "no early mornings"). Never repeat items already listed
   under "must respect". Otherwise [].
9. When changing an existing plan, return only the days that change, and null for stay, food
   and local transport unless they change. If nothing changes, plan is null.
10. drop_if_short: id of that day's lowest-priority activity. add_if_extra: id of an unused
    place nearby. Use "" if there is none.
11. ask: one question only if you cannot plan without the answer. Otherwise "". Travel dates
    are not required; never ask for them before planning.
12. reply: at most 80 words, first person, warm and concrete. Mention one trade-off you made.
    Call places by name in reply, risk and mitigation; ids like P3 are for the plan fields only.
""".strip()


def system_prompt(style: str) -> str:
    return PERSONA if style == "short" else f"{PERSONA}\n\n{RULES}"


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def place_line(alias: str, p: dict) -> str:
    about = p["content"] or p["alt"] or ""
    return (
        f"{alias} | {p['name']} | {p['kind']} | {p['area']} | "
        f"hours: {p['hours'] or 'unknown'} | price: {p['price'] or 'unknown'} | {_clip(about, 110)}"
    )


def render_plan(state: TripState, seg: Segment, places: dict) -> str:
    """The current plan for this city, plus the computed total for the whole trip."""
    plan = state.plan
    days = [d for d in plan.days if d.day in seg.days] if plan else []
    if not days:
        return "(none yet for this city)"
    lines = []
    for d in days:
        acts = "; ".join(
            f"{a.start} {a.poi} {state.pois[a.poi]['name'] if a.poi in state.pois else '?'} "
            f"{a.duration_min}m ₹{a.cost_inr}"
            for a in d.activities
        )
        lines.append(f"Day {d.day} ({d.theme}): {acts}")
    costs = plan.cities.get(seg.city)
    if costs and costs.stay:
        lines.append(f"Stay: {costs.stay.area}, ₹{costs.stay.per_night_inr}/night per room")
    if costs:
        lines.append(
            f"Food ₹{costs.food_per_day_inr}/person/day, "
            f"local transport ₹{costs.local_transport_per_day_inr}/day"
        )
    b = breakdown(state.request, state.route, plan, places)
    lines.append(f"Computed total for the whole trip so far: ₹{b.total} of target ₹{b.target}")
    return "\n".join(lines)


def render_trip(state: TripState, seg: Segment, places: dict) -> str:
    r = state.request
    env = envelope(r, state.route, places)
    stops = " -> ".join(
        [r.source] + [f"{s.city} (days {s.days[0]}-{s.days[-1]})" for s in state.route] + [r.source]
    )
    lines = [
        f"route: {stops} by {r.mode.value} | {r.days} days | "
        f"{r.travelers} travelers ({r.party.value}), {rooms(r)} room(s)",
        f"now planning: {seg.city}, days {seg.days[0]}-{seg.days[-1]}, "
        f"{nights_by_city(state.route)[seg.city]} night(s) here",
        f"pace: {r.pace.value} (max {MAX_PER_DAY[r.pace]} activities per day)",
        f"interests: {', '.join(r.interests) or 'general sightseeing'}",
        f"must respect: {'; '.join(r.constraints) or 'nothing specific'}",
        f"budget: ceiling ₹{r.budget_inr}; plan to at most ₹{target(r)} ({r.tier.value} variant). "
        f"Intercity transport is fixed at about ₹{env['intercity']}, leaving ₹{env['remaining']}.",
    ]
    typical = places.get(seg.city, {}).get("typical", {}).get(r.tier.value)
    if typical:
        lines.append(
            f"typical {r.tier.value} prices in {seg.city} (unverified guide): "
            f"stay ₹{typical['stay'][0]}-{typical['stay'][1]}/night; "
            f"food ₹{typical['food'][0]}-{typical['food'][1]}/person/day; "
            f"local transport ₹{typical['local'][0]}-{typical['local'][1]}/day"
        )
    if env["feasible"]:
        lines.append(
            f"caps: stay ≤ ₹{env['stay_per_night']}/night per room; "
            f"food ≤ ₹{env['food_per_person_day']}/person/day; "
            f"local transport ≤ ₹{env['local_per_day']}/day; "
            f"activities ≤ ₹{env['activities_per_person']}/person for the whole trip"
        )
    else:
        lines.append(
            "NOT FEASIBLE: the budget cannot cover this trip after intercity transport. "
            "Do not produce a plan; explain and suggest what to change."
        )
    return "\n".join(lines)


def shown_ids(state: TripState, seg: Segment, fresh: list[str]) -> list[str]:
    """Places shown this turn, all in this city: everything already planned here, then this
    turn's search hits, then older candidates, up to a cap that keeps the prompt small."""
    cap = min(6 + 4 * len(seg.days), 24)
    in_city = [a for a, p in state.pois.items() if p["destination"] == seg.city]
    ordered = list(dict.fromkeys(state.planned_ids(seg.days) + fresh + in_city))
    shown = {a for a in ordered if a in in_city}
    shown = set([a for a in ordered if a in shown][:cap])
    return [a for a in in_city if a in shown]  # stable order keeps prompts cache-friendly


def shown_guides(state: TripState, seg: Segment) -> list[str]:
    return [a for a, g in state.guides.items() if g["destination"] == seg.city]


def build_messages(
    state: TripState,
    seg: Segment,
    user_message: str,
    shown: list[str],
    places: dict,
    style: str,
) -> list[dict]:
    place_lines = [place_line(a, state.pois[a]) for a in shown]
    guide_lines = [
        f"{a} | {state.guides[a]['section']} ({state.guides[a]['area']}"
        f"{', unverified' if state.guides[a].get('verified') is False else ''}): "
        f"{_clip(state.guides[a]['text'].split(': ', 1)[-1], 320)}"
        for a in shown_guides(state, seg)
    ]
    context = "\n\n".join(
        [
            "TRIP\n" + render_trip(state, seg, places),
            "CURRENT PLAN\n" + render_plan(state, seg, places),
            "PLACES (id | name | kind | area | hours | price | about)\n" + "\n".join(place_lines),
            "GUIDE NOTES\n" + ("\n".join(guide_lines) or "(none)"),
            "USER\n" + user_message,
        ]
    )
    return [
        {"role": "system", "content": system_prompt(style)},
        *state.history[-HISTORY_TURNS:],
        {"role": "user", "content": context},
    ]


def repair_message(violations: list[str]) -> str:
    return (
        "Your plan failed these checks:\n- "
        + "\n- ".join(violations[:8])
        + "\nReturn the full corrected JSON."
    )
