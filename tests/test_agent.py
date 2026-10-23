import pytest

from tripmate.agent.budget import breakdown, envelope, intercity_cost, nights_by_city
from tripmate.agent.route import build_route, order_cities, split_days
from tripmate.agent.validator import avoid_terms, minutes, validate
from tripmate.agent.whatif import affected_segments, resize_route
from tripmate.profile import TravelProfile, load_profile, save_profile
from tripmate.schemas import (
    Activity,
    AgentTurn,
    CityCosts,
    DayPlan,
    Plan,
    PlanDraft,
    Segment,
    Stay,
    TripRequest,
    turn_schema,
)
from tripmate.state import TripState, diff_plans, experience_score, merge_plan

PLACES = {
    "Bengaluru": {"lat": 12.98, "long": 77.60},
    "Goa": {"lat": 15.50, "long": 73.83},
    "Gokarna": {"lat": 14.55, "long": 74.32},
    "Hampi": {"lat": 15.34, "long": 76.46},
}
POIS = {
    "P1": {"id": "a", "name": "Old Fort", "text": "A fort.", "destination": "Goa"},
    "P2": {"id": "b", "name": "Shiva Temple", "text": "An old temple.", "destination": "Goa"},
    "P3": {"id": "c", "name": "Beach", "text": "Sandy beach.", "destination": "Goa"},
    "P4": {"id": "d", "name": "Om Beach", "text": "A beach.", "destination": "Gokarna"},
}


def make_state(cities=("Goa",), **kw):
    base = dict(source="Bengaluru", destinations=list(cities), days=2, travelers=2, budget_inr=30000)
    req = TripRequest(**(base | kw))
    state = TripState(request=req, route=build_route(req, PLACES, {}))
    state.pois = dict(POIS)
    return state


def act(poi, start="09:00", dur=60, cost=100):
    return Activity(poi=poi, start=start, duration_min=dur, cost_inr=cost, why="x")


def day(n, acts):
    return DayPlan(day=n, theme="t", activities=acts, drop_if_short="", add_if_extra="")


def costs(stay=2000, food=500, local=300):
    return CityCosts(
        stay=Stay(area="centre", per_night_inr=stay, why="central"),
        food_per_day_inr=food,
        local_transport_per_day_inr=local,
    )


def plan(days, cities=("Goa",), **kw):
    return Plan(cities={c: costs(**kw) for c in cities}, days=[day(i, a) for i, a in enumerate(days, 1)])


# ---- budget ----


def test_budget_arithmetic():
    state = make_state()
    b = breakdown(state.request, state.route, plan([[act("P1")], [act("P3", cost=250)]]), PLACES)
    assert b.intercity == intercity_cost(state.request, state.route, PLACES)
    assert b.stay == 2000 * 1 * 1  # 1 room, 1 night
    assert b.food == 500 * 2 * 2
    assert b.local_transport == 300 * 2
    assert b.activities == (100 + 250) * 2
    assert b.total == b.intercity + b.stay + b.food + b.local_transport + b.activities
    assert b.target == 24000  # balanced = 80% of ceiling


def test_envelope_flags_infeasible_budget():
    state = make_state()
    assert envelope(state.request, state.route, PLACES)["feasible"]
    poor = make_state(budget_inr=3000)
    assert not envelope(poor.request, poor.route, PLACES)["feasible"]


# ---- multi-city routing ----


def test_route_orders_cities_to_avoid_backtracking():
    # Goa and Gokarna are neighbours on the coast; Hampi is inland on the way back.
    order = order_cities("Bengaluru", ["Gokarna", "Hampi", "Goa"], PLACES)
    assert order in (["Hampi", "Goa", "Gokarna"], ["Gokarna", "Goa", "Hampi"])


def test_split_days_gives_every_city_a_day_and_favours_bigger_cities():
    assert split_days(["Goa", "Gokarna"], 5, {"Goa": 30, "Gokarna": 5}) == {"Goa": 4, "Gokarna": 1}
    with pytest.raises(ValueError):
        split_days(["Goa", "Gokarna"], 1, {})


def test_build_route_assigns_consecutive_days_and_honours_fixed_split():
    state = make_state(cities=("Goa", "Gokarna"), days=4, days_per_city={"Goa": 3, "Gokarna": 1})
    by_city = {s.city: s.days for s in state.route}
    assert sorted(d for days in by_city.values() for d in days) == [1, 2, 3, 4]
    assert len(by_city["Goa"]) == 3 and len(by_city["Gokarna"]) == 1
    nights = nights_by_city(state.route)
    assert sum(nights.values()) == 3  # no night on the day you go home


def test_multi_city_budget_prices_every_leg():
    one = make_state(cities=("Goa",), days=4)
    two = make_state(cities=("Goa", "Gokarna"), days=4)
    assert intercity_cost(two.request, two.route, PLACES) > intercity_cost(one.request, one.route, PLACES)


# ---- validator ----


def test_valid_plan_has_no_violations():
    state = make_state()
    assert validate(state, plan([[act("P1"), act("P2", "11:00")], [act("P3")]]), PLACES) == []


def test_validator_catches_each_rule():
    def problems(p, **kw):
        return " ".join(validate(make_state(**kw), p, PLACES))

    assert "not in PLACES" in problems(plan([[act("P9")], [act("P3")]]))
    assert "more than once" in problems(plan([[act("P1")], [act("P1")]]))
    assert "overlaps" in problems(plan([[act("P1", "09:00", 120), act("P2", "10:00")], [act("P3")]]))
    assert "exactly days" in problems(plan([[act("P1")]]))
    assert "max 3" in problems(
        plan([[act("P1", "08:00"), act("P2", "10:00"), act("P3", "12:00"), act("P9", "14:00")], []]),
        pace="relaxed",
    )
    assert "exceeds the target" in problems(plan([[act("P1")], [act("P3")]], stay=50000))
    assert "no temple" in problems(plan([[act("P2")], [act("P3")]]), constraints=["No temples"])
    assert "not HH:MM" in problems(plan([[act("P1", "9am")], [act("P3")]]))
    assert "stay is missing" in problems(Plan(days=[day(1, [act("P1")]), day(2, [act("P3")])]))
    assert "another city" in problems(plan([[act("P4")], [act("P3")]]))


def test_validator_accepts_partial_plan_while_planning_city_by_city():
    state = make_state(cities=("Goa", "Gokarna"), days=2)
    first = state.route[0]
    poi = "P1" if first.city == "Goa" else "P4"
    partial = Plan(cities={first.city: costs()}, days=[day(first.days[0], [act(poi)])])
    assert validate(state, partial, PLACES, expected_days=set(first.days)) == []
    assert any("exactly days" in v for v in validate(state, partial, PLACES))


def test_time_and_constraint_parsing():
    assert minutes("09:30") == 570 and minutes("25:00") is None
    assert avoid_terms(["vegetarian", "no temples", "Avoid museums"]) == ["temple", "museum"]


# ---- state ----


def test_merge_plan_replaces_only_this_citys_days():
    seg = Segment(city="Goa", days=[1, 2])
    old = plan([[act("P1")], [act("P2")]])
    draft = PlanDraft(days=[day(2, [act("P3")]), day(3, [act("P4")])])
    merged = merge_plan(old, draft, seg)
    assert [d.day for d in merged.days] == [1, 2]  # day 3 is outside the segment: ignored
    assert merged.days[0].activities[0].poi == "P1"
    assert merged.days[1].activities[0].poi == "P3"
    assert merged.cities["Goa"].stay.per_night_inr == 2000  # null in the draft: kept


def test_diff_plans_describes_changes():
    old = plan([[act("P1"), act("P2", "11:00")], [act("P3")]])
    new = plan([[act("P1", "10:00")], [act("P3")]], stay=1500)
    changes = " | ".join(diff_plans(old, new, POIS))
    assert "removed Shiva Temple" in changes
    assert "Old Fort moved 09:00 -> 10:00" in changes
    assert "Goa stay: ₹2000 -> ₹1500" in changes
    assert diff_plans(old, old, POIS) == []


def test_state_ids_are_stable_and_memory_dedupes():
    state = make_state(interests=["beaches", "history"])
    state.pois = {}
    assert state.register_poi(POIS["P1"], "history") == "P1"
    assert state.register_poi(POIS["P3"], "beaches") == "P2"
    assert state.register_poi(POIS["P1"]) == "P1"
    state.remember(["Vegetarian", "vegetarian", " "])
    assert state.request.constraints == ["Vegetarian"]
    state.plan = plan([[act("P1")], []])
    assert experience_score(state) == round(100 * (0.7 * 0.5 + 0.3 * 1.0))


# ---- schema, what-if, profile ----


def test_turn_schema_narrows_ids_and_forbids_skipping_costs_on_first_plan():
    schema = turn_schema(["P1", "P2"], ["G1"], first_plan=True)
    defs = schema["$defs"]
    assert defs["Activity"]["properties"]["poi"]["enum"] == ["P1", "P2"]
    assert defs["HeadsUp"]["properties"]["source"]["enum"] == ["G1", "P1", "P2"]
    assert defs["PlanDraft"]["properties"]["stay"] == {"$ref": "#/$defs/Stay"}
    assert schema["properties"]["plan"] == {"$ref": "#/$defs/PlanDraft"}
    refine = turn_schema(["P1"], [], first_plan=False)
    assert "anyOf" in refine["properties"]["plan"]  # null allowed when nothing changes


def test_resize_route_and_affected_segments():
    route = [Segment(city="Goa", days=[1, 2, 3]), Segment(city="Gokarna", days=[4])]
    longer = resize_route(route, 6)
    assert [s.days for s in longer] == [[1, 2, 3], [4, 5, 6]]
    shorter = resize_route(route, 3)
    assert [s.days for s in shorter] == [[1, 2], [3]]
    with pytest.raises(ValueError):
        resize_route(route, 1)

    state = make_state(cities=("Goa", "Gokarna"), days=4)
    first, second = state.route
    hit = affected_segments(state, [f"Day {second.days[0]} has 5 activities; max 3."])
    assert [s.city for s in hit] == [second.city]
    assert affected_segments(state, ["Total cost ₹9 exceeds the target ₹1."]) == state.route


def test_profile_round_trip_and_learning(tmp_path):
    profile = TravelProfile(name="Asha K", interests=["food"], constraints=["vegetarian"])
    assert profile.learn(["Vegetarian", "no early mornings"]) == ["no early mornings"]
    save_profile(profile, tmp_path)
    loaded = load_profile("Asha K", tmp_path)
    assert loaded == profile
    assert load_profile("nobody", tmp_path) is None
    request = TripRequest(destinations=["Goa"], days=3, budget_inr=40000, **loaded.trip_defaults())
    assert request.constraints == ["vegetarian", "no early mornings"] and request.source == "Bengaluru"


def test_remember_returns_only_new_items_and_ids_are_named_in_prose():
    state = make_state(constraints=["vegetarian"])
    assert state.remember(["Vegetarian", "no early mornings"]) == ["no early mornings"]
    assert state.name_ids("Start at P1, then P3. P99 stays.") == "Start at Old Fort, then Beach. P99 stays."
    assert state.name_ids("near P1 Old Fort and P3 (Beach), P2: Shiva Temple.") == (
        "near Old Fort and Beach, Shiva Temple."
    )
    state.guides = {"G1": {"area": "Anjuna", "section": "Stay safe"}, "G2": {"area": "Goa", "section": "Seasons and weather"}}
    assert state.name_ids("Watch for scams (G1, G2). Per G2, avoid June [G1].") == (
        "Watch for scams. Per the Goa seasons and weather notes, avoid June."
    )
    assert turn_schema(["P1"], [], first_plan=True)["properties"]["remember"]["maxItems"] == 0


def test_parse_turn_accepts_fenced_json_without_a_grammar():
    from tripmate.agent.orchestrator import parse_turn

    body = AgentTurn(ask="", remember=[], plan=None, heads_up=[], reply="Here you go.").model_dump_json()
    assert parse_turn(f"Sure!\n```json\n{body}\n```").reply == "Here you go."
    assert parse_turn("no json here") is None


def test_reply_claiming_an_edit_without_a_plan_gets_a_plan_required_repair():
    from tripmate.agent.orchestrator import CLAIM_WITHOUT_PLAN, Agent
    from tripmate.config import get_settings
    from tripmate.llm.client import Completion

    class FakeLLM:
        def __init__(self, outputs):
            self.outputs, self.schemas = list(outputs), []

        def chat(self, messages, schema=None, **kw):
            self.schemas.append(schema)
            return Completion(self.outputs.pop(0), 0, 10, 0.1)

    class NoSearch:
        def search(self, *a, **kw):
            return []

    def envelope_json(reply, plan_draft):
        return AgentTurn(ask="", remember=[], plan=plan_draft, heads_up=[], reply=reply).model_dump_json()

    state = make_state(budget_inr=60000)
    for p in state.pois.values():
        p |= {"kind": "see", "area": "", "hours": "", "price": "", "content": p["text"], "alt": ""}
    state.plan = plan([[act("P1")], [act("P3")]])
    edited = PlanDraft(days=[day(2, [act("P3"), act("P2", start="11:00")])])
    llm = FakeLLM([envelope_json("I've added the temple to day 2.", None),
                   envelope_json("I've added the temple to day 2.", edited)])
    agent = Agent.__new__(Agent)
    agent.s, agent.retriever, agent.llm, agent.places = get_settings(), NoSearch(), llm, PLACES
    agent.style, agent.repair, agent.constrained = "short", True, True
    agent.recorder, agent.teacher_note = None, None

    result = agent.turn(state, "Add the temple on day 2")
    assert result.first_pass_violations == [CLAIM_WITHOUT_PLAN]
    assert llm.schemas[1]["properties"]["plan"] == {"$ref": "#/$defs/PlanDraft"}
    assert result.applied and [a.poi for a in state.plan.days[1].activities] == ["P3", "P2"]
