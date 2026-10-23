from tripmate.eval.behavior import compare, figures_grounded, persona_ok, summarize


def row(**kw):
    base = {"scenario": 0, "focus": None, "turn": 1, "intent": "swap", "feasible": True,
            "ok": True, "parsed": True, "persona_ok": True, "heads_up": False,
            "plan_changed": True, "valid_plan": False, "reply": "", "seconds": 10.0, "tokens": 500}
    return base | kw


def test_summary_separates_goals_and_turn_depth():
    rows = [
        row(turn=0, intent="first", heads_up=True, valid_plan=True),
        row(turn=1, intent="conflict_budget", ok=False, plan_changed=True),
        row(turn=3, intent="question", plan_changed=False),
    ]
    table = dict(summarize(rows))
    assert table["constraints: first plan passes validator"] == "100% (1)"
    assert table["constraints: pushback on over-budget ask"] == "0% (1)"
    assert table["constraints: conflict left plan unchanged"] == "0% (1)"
    assert table["stability: turns 3+ pass"] == "100% (1)"
    assert "base" in compare({"base": rows, "ft": rows}).splitlines()[0]


def test_persona_rules():
    assert persona_ok("I kept day two light because of the heat.")
    assert not persona_ok("As an AI, I cannot travel.")
    assert not persona_ok("I swapped P4 for the fort.")
    assert not persona_ok("")


def test_rupee_figures_must_come_from_the_prompt_or_plan():
    prompt = "stay cap ₹7,250 per room per night; ceiling ₹84000"
    assert figures_grounded("Your cap is ₹7250 a night, total ₹84k.", prompt, "{}")
    assert not figures_grounded("That exceeds the cap of ₹8,750.", prompt, "{}")
    assert figures_grounded("No prices here.", prompt, "{}")
