"""Behavior eval: replay scripted conversations on held-out cities and score every turn.

The same checks that filter training data (validator, persona rules, pushback, memory) are
used as pass/fail metrics here, so "before" and "after" fine-tuning are measured on exactly
the behavior the data was built to teach. Repair is off, so this is first-pass behavior, and
there is no teacher note, so the model has to notice conflicts by itself.

One metric family per fine-tuning goal:
  persona       reply rules hold (non-empty, short, first person, no ids, no "as an AI")
  constraints   first plans pass the validator; conflicting requests get pushback, no change
  proactivity   first plans carry a grounded heads-up
  stability     pass rate on late turns (3+) compared with early turns
"""

import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from tripmate.agent.orchestrator import FIRST_MESSAGE, Agent
from tripmate.agent.validator import CHANGE_CLAIM
from tripmate.datagen.generate import (
    EDIT_INTENTS,
    MAX_REPLY_WORDS,
    REPAIR_TALK,
    reject_reason,
)
from tripmate.datagen.scenarios import Scenario, next_user_turn, sample_scenario

PERSONA_REASONS = (
    "reply empty or too long",
    "internal ids leaked into reply",
    "off-persona or asked for dates",
    "talked about its own repair",
)


@dataclass
class EvalScenario:
    scenario: Scenario
    focus: str | None


def eval_scenarios(
    n: int, places: dict, cities: list[str], seed: int = 7, follow_ups: int = 5
) -> list[EvalScenario]:
    """Half ordinary trips, half conflict trips; every feasible one gets `follow_ups` turns."""
    out = []
    for i in range(n):
        focus = "conflict" if i % 2 else None
        sc = sample_scenario(i, seed, places, cities, focus)
        sc.n_follow_ups = follow_ups if sc.feasible else 0
        out.append(EvalScenario(sc, focus))
    return out


def persona_ok(reply: str) -> bool:
    words = reply.split()
    return (
        0 < len(words) <= MAX_REPLY_WORDS
        and not re.search(r"\b[PG]\d+\b", reply)
        and "as an ai" not in reply.lower()
        and not REPAIR_TALK.search(reply)
    )


RUPEES = re.compile(r"₹\s?(\d[\d,]*(?:\.\d+)?)\s?(k|lakh)?", re.IGNORECASE)


def rupee_amounts(text: str) -> list[float]:
    scale = {"k": 1e3, "lakh": 1e5}
    return [
        float(n.replace(",", "")) * scale.get(unit.lower(), 1) for n, unit in RUPEES.findall(text)
    ]


def figures_grounded(reply: str, prompt: str, output: str) -> bool:
    """Every rupee amount in the reply appears (within 2%) in the prompt or in the model's
    own plan. Budget arithmetic belongs to code, so a figure from nowhere is a hallucination."""
    known = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", prompt + output)]
    return all(any(abs(a - k) <= 0.02 * max(a, 1) for k in known) for a in rupee_amounts(reply))


def run_eval(
    agent: Agent, scenarios: list[EvalScenario], questions: list[str], out: Path, log=print
):
    """Write one row per (turn, city segment) to `out` and return the rows.

    Unlike data generation, failed turns are kept (a failed first plan is still shown to the
    user), so later turns see whatever the model actually produced.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as f:
        for k, es in enumerate(scenarios, 1):
            sc = es.scenario
            rng = random.Random(f"eval-{sc.id}-{es.focus}")
            state = agent.start(sc.request)
            used: set[str] = set()
            for t in range(1 + sc.n_follow_ups):
                if t == 0:
                    intent, message = "first", FIRST_MESSAGE
                else:
                    intent, message = next_user_turn(
                        rng, state, agent.places, questions, used, es.focus
                    )
                used.add(intent)
                agent.recorder = []
                result = agent.turn(state, message)
                records = agent.recorder or [None]  # None: the response did not parse
                for rec in records:
                    reason = (
                        reject_reason(rec, intent, sc.feasible, state.request.constraints)
                        if rec
                        else "unparseable"
                    )
                    row = {
                        "scenario": sc.id,
                        "focus": es.focus,
                        "turn": t,
                        "intent": intent,
                        "feasible": sc.feasible,
                        "city": rec["city"] if rec else "",
                        "ok": reason is None,
                        "reason": reason,
                        "parsed": rec is not None,
                        "persona_ok": bool(rec) and persona_ok(rec["turn"].reply),
                        "figures_ok": bool(rec)
                        and figures_grounded(
                            rec["turn"].reply,
                            rec["messages"][-1]["content"],
                            rec["turn"].model_dump_json(),
                        ),
                        "heads_up": bool(rec) and bool(rec["turn"].heads_up),
                        "valid_plan": bool(rec)
                        and rec["turn"].plan is not None
                        and not rec["violations"],
                        "plan_changed": bool(rec) and rec["turn"].plan is not None,
                        "reply": rec["turn"].reply if rec else "",
                        "seconds": result.seconds / len(records),
                        "tokens": result.completion_tokens / len(records),
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            agent.recorder = None
            done = [r for r in rows if r["scenario"] == sc.id]
            log(
                f"[{k}/{len(scenarios)}] {'+'.join(sc.request.destinations)} "
                f"{'conflict' if es.focus else 'normal'}: {sum(r['ok'] for r in done)}/{len(done)} ok"
            )
    return rows


def _rate(rows: list[dict], key: str = "ok") -> str:
    if not rows:
        return "n/a"
    return f"{sum(bool(r[key]) for r in rows) / len(rows):.0%} ({len(rows)})"


def summarize(rows: list[dict]) -> list[tuple[str, str]]:
    rows = [rescore(r) for r in rows]
    first = [r for r in rows if r["intent"] == "first"]
    feasible_first = [r for r in first if r["feasible"]]
    conflicts = [r for r in rows if r["intent"].startswith("conflict")]
    follow = [r for r in rows if r["turn"] > 0]
    return [
        ("all turns pass every check", _rate(rows)),
        ("response parsed (schema)", _rate(rows, "parsed")),
        ("persona: reply rules hold", _rate(rows, "persona_ok")),
        ("grounding: rupee figures in reply are real", _rate([r for r in rows if "figures_ok" in r], "figures_ok")),
        ("constraints: first plan passes validator", _rate(feasible_first, "valid_plan")),
        ("constraints: infeasible trip declined", _rate([r for r in first if not r["feasible"]])),
        ("constraints: pushback on constraint clash", _rate([r for r in conflicts if r["intent"] == "conflict_constraint"])),
        ("constraints: pushback on over-budget ask", _rate([r for r in conflicts if r["intent"] == "conflict_budget"])),
        ("constraints: conflict left plan unchanged", f"{sum(not r['plan_changed'] for r in conflicts) / max(len(conflicts), 1):.0%} ({len(conflicts)})"),
        ("proactivity: first plan has heads-up", _rate(feasible_first, "heads_up")),
        ("edit requests change the plan", _rate([r for r in follow if r["intent"] in EDIT_INTENTS], "plan_changed")),
        ("reply never claims an edit it did not make", _rate(follow, "honest")),
        ("follow-ups pass (non-conflict)", _rate([r for r in follow if not r["intent"].startswith("conflict")])),
        ("stability: turns 1-2 pass", _rate([r for r in follow if r["turn"] <= 2])),
        ("stability: turns 3+ pass", _rate([r for r in follow if r["turn"] >= 3])),
        ("latency per call (s, median)", f"{statistics.median(r['seconds'] for r in rows):.1f}" if rows else "n/a"),
        ("output tokens per call (median)", f"{statistics.median(r['tokens'] for r in rows):.0f}" if rows else "n/a"),
    ]  # fmt: skip


def compare(runs: dict[str, list[dict]]) -> str:
    """Side-by-side table, one column per run."""
    tables = {name: dict(summarize(rows)) for name, rows in runs.items()}
    metrics = list(next(iter(tables.values())))
    width = max(map(len, metrics)) + 2
    lines = [f"{'metric':<{width}}" + "".join(f"{n:<16}" for n in tables)]
    for m in metrics:
        lines.append(f"{m:<{width}}" + "".join(f"{tables[n][m]:<16}" for n in tables))
    return "\n".join(lines)


def rescore(row: dict) -> dict:
    """Apply checks added after a run, from the fields every row keeps."""
    if row["parsed"] and row["turn"] > 0 and not row["plan_changed"]:
        if row["intent"] in EDIT_INTENTS:
            row = row | {
                "ok": False,
                "reason": row["reason"] or "did not change the plan when asked",
            }
        elif CHANGE_CLAIM.search(row["reply"]):
            row = row | {"ok": False, "reason": row["reason"] or "claimed a change it did not make"}
    false_claim = (
        row["parsed"] and not row["plan_changed"] and bool(CHANGE_CLAIM.search(row["reply"]))
    )
    return row | {"honest": not (row["turn"] > 0 and false_claim)}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
