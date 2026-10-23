import json
import random

from tripmate.agent.budget import envelope
from tripmate.agent.route import build_route
from tripmate.datagen.generate import reject_reason, split, teacher_note
from tripmate.datagen.scenarios import conflict_terms, next_user_turn, sample_scenario
from tripmate.datagen.seeds import first_question, is_generic_question
from tripmate.kb.profiles import sections, typical_costs
from tripmate.schemas import AgentTurn, HeadsUp, PlanDraft
from tripmate.state import TripState

PLACES = {
    "Bengaluru": {"lat": 12.98, "long": 77.60},
    "Mumbai": {"lat": 19.03, "long": 72.85},
    "Goa": {"lat": 15.50, "long": 73.83},
    "Hampi": {"lat": 15.34, "long": 76.46},
}
CITIES = ["Goa", "Hampi"]


def turn(**kw):
    base = dict(
        ask="",
        remember=[],
        plan=PlanDraft(),
        reply="Here is your plan.",
        heads_up=[HeadsUp(risk="r", mitigation="m", source="G1")],
    )
    return AgentTurn(**(base | kw))


def record(**kw):
    return {"turn": turn(**kw), "violations": []}


def test_scenarios_are_deterministic_and_budgets_match_feasibility():
    a = [sample_scenario(i, 0, PLACES, CITIES) for i in range(40)]
    b = [sample_scenario(i, 0, PLACES, CITIES) for i in range(40)]
    assert [s.request for s in a] == [s.request for s in b]
    assert any(not s.feasible for s in a) and sum(s.feasible for s in a) > 30
    for s in a:
        route = build_route(s.request, PLACES, {})
        assert envelope(s.request, route, PLACES)["feasible"] == s.feasible
        assert s.request.source not in s.request.destinations
        assert (s.n_follow_ups == 0) == (not s.feasible)


def test_next_user_turn_fills_every_template():
    sc = sample_scenario(1, 0, PLACES, CITIES)
    state = TripState(request=sc.request, route=build_route(sc.request, PLACES, {}))
    rng = random.Random(0)
    for _ in range(60):
        intent, text = next_user_turn(rng, state, PLACES, ["Is travel insurance worth it?"], set())
        assert text and "{" not in text and intent


def test_conflict_focus_scenarios_always_set_up_a_pushback():
    scenarios = [sample_scenario(i, 0, PLACES, CITIES, focus="conflict") for i in range(40)]
    assert all(conflict_terms(s.request.constraints) for s in scenarios)
    assert sum(not s.feasible for s in scenarios) >= 4
    sc = next(s for s in scenarios if s.feasible)
    state = TripState(request=sc.request, route=build_route(sc.request, PLACES, {}))
    rng, used = random.Random(0), set()
    intents = []
    for _ in range(3):
        intent, text = next_user_turn(rng, state, PLACES, [], used, focus="conflict")
        used.add(intent)
        intents.append(intent)
        note = teacher_note(intent, text, state, PLACES)
        assert (note is not None) == intent.startswith("conflict")
        assert note is None or "plan must be null" in note
    assert set(intents[:2]) == {"conflict_constraint", "conflict_budget"}


def test_filters_keep_good_turns_and_name_the_problem():
    assert reject_reason(record(), "first", True) is None
    assert (
        reject_reason({"turn": turn(), "violations": ["x"]}, "first", True)
        == "validator violations"
    )
    assert "ids leaked" in reject_reason(record(reply="I moved P4 to day 2."), "relax_day", True)
    assert "heads-up" in reject_reason(record(heads_up=[]), "first", True)
    assert "asked" in reject_reason(record(ask="Which month?"), "first", True)
    assert "infeasible" in reject_reason(record(), "first", False)
    assert reject_reason(record(plan=None), "first", False) is None
    assert "should not" in reject_reason(record(), "question", True)
    budget_reply = "That resort would break your budget; your cap is ₹4,000 a night."
    assert reject_reason(record(plan=None, reply=budget_reply), "conflict_budget", True) is None
    assert "conflict" in reject_reason(record(plan=None), "conflict_budget", True)
    clash = record(plan=None, reply="A temple visit clashes with your wishes; try the fort?")
    assert reject_reason(clash, "conflict_constraint", True, ["no temples"]) is None
    assert "conflict" in reject_reason(
        record(plan=None), "conflict_constraint", True, ["no temples"]
    )
    assert "did not remember" in reject_reason(record(), "new_constraint", True)
    assert "did not say" in reject_reason(record(remember=["vegetarian"]), "relax_day", True)
    assert "repair" in reject_reason(
        record(reply="I've corrected the budget for you."), "first", True
    )
    assert "when asked" in reject_reason(record(plan=None, reply="Done."), "swap", True)
    claim = record(plan=None, reply="I've added the fort to day 2.")
    assert "did not make" in reject_reason(claim, "question", True)
    assert "too long" in reject_reason(record(reply="word " * 200), "relax_day", True)


def test_split_holds_out_destinations_and_keeps_conversations_together(tmp_path):
    rows = [
        {
            "messages": [{"role": "user", "content": str(i)}],
            "meta": {
                "scenario": i // 3,
                "destinations": ["Udaipur"] if i // 3 % 5 == 0 else ["Goa"],
            },
        }
        for i in range(150)
    ]
    src = tmp_path / "all.jsonl"
    for i, r in enumerate(rows):
        r["meta"] |= {"turn": i % 3, "city": "Goa", "intent": "question"}  # three turns per conversation
        r["messages"].append({"role": "assistant", "content": turn(plan=None).model_dump_json()})
    rows.append(rows[0])  # a duplicate written by an interrupted run is dropped
    src.write_text("\n".join(json.dumps(r) for r in rows))
    counts = split(src, tmp_path, {"Udaipur"})
    assert counts["test"] == 30 and sum(counts.values()) == 150 and counts["valid"] % 3 == 0
    line = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert list(line) == ["messages"]


def test_clean_sample_names_ids_using_the_prompt():
    from tripmate.datagen.generate import clean_sample

    prompt = (
        "PLACES (id | name | kind | area | hours | price | about)\n"
        "P1 | Old Fort | see | Panaji | hours: unknown | price: unknown | x\n"
        "GUIDE NOTES\nG1 | Stay safe (Anjuna, unverified): touts\n\nUSER\nhi"
    )
    out = turn(
        reply="Start at P1.",
        heads_up=[HeadsUp(risk="Touts", mitigation="Ignore them (Source: G1).", source="G1")],
    )
    row = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out.model_dump_json()},
        ]
    }
    cleaned = AgentTurn.model_validate_json(clean_sample(row)["messages"][-1]["content"])
    assert cleaned.reply == "Start at Old Fort."
    assert cleaned.heads_up[0].mitigation == "Ignore them." and cleaned.heads_up[0].source == "G1"


def test_seed_question_filter():
    assert is_generic_question("What should I pack for a week-long trip in the monsoon?")
    assert not is_generic_question("What are the best hotels in Paris for a family trip?")
    assert not is_generic_question("Write a comedy story about coworkers?")
    assert first_question("['How do I pack light?', 'answer']") == "How do I pack light?"
    assert first_question("not a list") is None


def test_profile_sections_skip_place_names():
    r = json.loads(open("RAG data/india_tourism_dataset.json").read())[0]
    text = " ".join(sections(r).values())
    assert "Post-Monsoon" in text and "petty theft" in text
    assert "Dudhsagar" not in text  # attractions are not chunked: they have no citable id
    assert typical_costs(r)["balanced"]["stay"] == [2000, 4000]
