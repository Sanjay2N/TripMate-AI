"""Generate fine-tuning data: the teacher model plays TripMate against scripted user turns.

The teacher runs through the real Agent (same retrieval, state, validator and repair loop as
production) with the long rule prompt. Each accepted model call becomes one training sample
whose prompt is the SHORT style, so the student learns the rules instead of reading them.
"""

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from tripmate.agent.budget import envelope
from tripmate.agent.orchestrator import FIRST_MESSAGE, Agent
from tripmate.agent.validator import CHANGE_CLAIM
from tripmate.datagen.scenarios import (
    CONFLICT_ASKS,
    Scenario,
    conflict_terms,
    next_user_turn,
    sample_scenario,
)
from tripmate.datagen.seeds import load_questions
from tripmate.schemas import AgentTurn
from tripmate.state import clean_turn

MAX_REPLY_WORDS = 110
NO_PLAN_INTENTS = ("question", "conflict_constraint", "conflict_budget")
EDIT_INTENTS = ("relax_day", "add_interest", "swap")
FOCUS_OFFSET = {"conflict": 10_000}  # scenario ids per focus, so they never collide
BUDGET_WORDS = ("budget", "₹", "afford", "cap", "limit")
# After a repair turn the teacher sometimes talks about fixing its own draft, which the user
# never saw ("I've corrected the budget..."). The student should not learn to say that.
REPAIR_TALK = re.compile(
    r"\b(i've|i have) (corrected|fixed|adjusted the plan to pass)|\bfailed (the )?checks?\b", re.I
)


def teacher_note(intent: str, message: str, state, places: dict) -> str | None:
    """Tell the teacher what the scripted user turn is testing. Only the teacher sees this."""
    if intent == "conflict_constraint":
        clash = next(
            (
                c
                for c in state.request.constraints
                if any(CONFLICT_ASKS[t] in message for t in conflict_terms([c]))
            ),
            "; ".join(state.request.constraints),
        )
        return (
            f"NOTE FOR THIS TURN: this request conflicts with the must-respect item "
            f"'{clash}'. Do not change the plan: plan must be null. In reply, say plainly "
            f"that it clashes with '{clash}', and offer the closest alternative from PLACES "
            "by name, asking whether they want it."
        )
    if intent == "conflict_budget":
        cap = envelope(state.request, state.route, places)["stay_per_night"]
        return (
            f"NOTE FOR THIS TURN: the stay the user asks for is far above the stay cap of "
            f"₹{cap:,} per room per night in TRIP. Do not change the plan: plan must be null. "
            "In reply, explain that it would break their budget, give the cap in rupees, and "
            "suggest the best kind of stay within it."
        )
    return None


def pushes_back(reply: str, intent: str, constraints: list[str]) -> bool:
    """Does the reply name the conflict, rather than just saying no or going along?"""
    text = reply.lower()
    if intent == "conflict_constraint":
        terms = conflict_terms(constraints)
        return any(t in text or CONFLICT_ASKS[t] in text for t in terms)
    return any(w in text for w in BUDGET_WORDS)


def reject_reason(
    record: dict, intent: str, feasible: bool, constraints: list[str] = ()
) -> str | None:
    """Why a recorded model call is not good enough to learn from (None = keep)."""
    turn = record["turn"]
    reply = turn.reply.strip()
    if record["violations"]:
        return "validator violations"
    if not reply or len(reply.split()) > MAX_REPLY_WORDS:
        return "reply empty or too long"
    if re.search(r"\b[PG]\d+\b", reply):
        return "internal ids leaked into reply"
    if REPAIR_TALK.search(reply):
        return "talked about its own repair"
    if "as an ai" in reply.lower() or "travel dates" in (reply + turn.ask).lower():
        return "off-persona or asked for dates"
    if intent == "first":
        if not feasible:
            return None if turn.plan is None else "planned an infeasible trip"
        if turn.plan is None:
            return "no plan on first turn"
        if turn.ask:
            return "asked instead of planning"
        if not turn.heads_up:
            return "no grounded heads-up"
    if intent in EDIT_INTENTS and turn.plan is None:
        return "did not change the plan when asked"
    if intent != "first" and turn.plan is None and CHANGE_CLAIM.search(reply):
        return "claimed a change it did not make"
    if intent in NO_PLAN_INTENTS and turn.plan is not None:
        return "changed the plan when it should not have"
    if intent.startswith("conflict") and not pushes_back(reply, intent, constraints):
        return "did not name the conflict"
    if intent == "new_constraint" and not turn.remember:
        return "did not remember the new constraint"
    if intent != "new_constraint" and turn.remember:
        return "remembered something the user did not say"
    return None


def run_scenario(
    agent: Agent, sc: Scenario, questions: list[str], seed: int, focus: str | None = None
) -> tuple[list, Counter]:
    """Play one conversation and return the accepted samples.

    A rejected turn is undone by restoring the state from before it, so no later sample is
    built on top of a bad exchange. The opening plan gets two tries; without it there is no
    conversation.
    """
    rng = random.Random(f"{seed}-{sc.id}-turns")
    samples, stats = [], Counter()
    state = agent.start(sc.request)
    used: set[str] = set()
    script = [("first", FIRST_MESSAGE), ("first", FIRST_MESSAGE)]
    turn_index = 0
    while script:
        intent, message = script.pop(0)
        snapshot = state.model_copy(deep=True)
        agent.recorder = []
        agent.teacher_note = teacher_note(intent, message, state, agent.places)
        agent.turn(state, message)
        agent.teacher_note = None
        records = agent.recorder
        constraints = state.request.constraints
        reasons = [reject_reason(r, intent, sc.feasible, constraints) for r in records]
        if not records or any(reasons):
            stats[f"rejected: {next((r for r in reasons if r), 'unparseable')}"] += 1
            state = snapshot
            if intent == "first" and not script:
                break  # both tries at the opening plan failed
        else:
            for record in records:
                samples.append(
                    {
                        "messages": [
                            *record["messages"],
                            {"role": "assistant", "content": record["turn"].model_dump_json()},
                        ],
                        "meta": {
                            "scenario": sc.id,
                            "turn": turn_index,
                            "intent": intent,
                            "city": record["city"],
                            "destinations": sc.request.destinations,
                            "repaired": record["repaired"],
                            "focus": focus,
                        },
                    }
                )
                stats[f"kept: {intent}"] += 1
            turn_index += 1
            if intent == "first":
                script = [None] * sc.n_follow_ups  # filled in lazily: they depend on the plan
        used.add(intent)
        if script and script[0] is None:
            script[0] = next_user_turn(rng, state, agent.places, questions, used, focus)
    agent.recorder = None
    return samples, stats


def generate(
    agent: Agent, out: Path, n: int, seed: int = 0, focus: str | None = None, log=print
) -> Counter:
    """Append samples for n scenarios to `out`, skipping scenarios already done.

    A focus run ("conflict") uses its own range of scenario ids, so it adds to the same file.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done_file = out.with_suffix(".done")
    done = set(map(int, done_file.read_text().split())) if done_file.exists() else set()
    questions = load_questions(agent.s.data_dir / "seeds" / "questions.txt")
    cities = agent.retriever.destinations
    total = Counter()
    offset = FOCUS_OFFSET.get(focus, 0)
    for i in range(n):
        if offset + i in done:
            continue
        sc = sample_scenario(i, seed, agent.places, cities, focus)
        sc.id = offset + i
        t0 = time.perf_counter()
        samples, stats = run_scenario(agent, sc, questions, seed, focus)
        with out.open("a") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        with done_file.open("a") as f:
            f.write(f"{sc.id}\n")
        total += stats
        log(
            f"[{i + 1}/{n}] {'+'.join(sc.request.destinations)} {sc.request.days}d "
            f"-> {len(samples)} samples in {time.perf_counter() - t0:.0f}s | "
            + ", ".join(f"{k}={v}" for k, v in stats.items() if not k.startswith("kept"))
        )
    return total


def prompt_ids(prompt: str) -> tuple[dict, dict]:
    """Recover the id -> place and id -> guide-note maps from a rendered prompt."""
    pois, guides = {}, {}
    for line in prompt.splitlines():
        if m := re.match(r"(P\d+) \| (.+?) \| ", line):
            pois[m.group(1)] = {"name": m.group(2)}
        elif m := re.match(r"(G\d+) \| (.+?) \((.+?)(?:, unverified)?\): ", line):
            guides[m.group(1)] = {"section": m.group(2), "area": m.group(3)}
    return pois, guides


def clean_sample(row: dict) -> dict:
    """Apply today's id clean-up to a sample generated by an older version of the agent."""
    *context, target = row["messages"]
    pois, guides = prompt_ids(context[-1]["content"])
    turn = clean_turn(AgentTurn.model_validate_json(target["content"]), pois, guides)
    return {**row, "messages": [*context, {"role": "assistant", "content": turn.model_dump_json()}]}


def split(
    source: Path,
    out_dir: Path,
    test_destinations: set[str],
    valid_share: float = 0.1,
    edit_weight: int = 1,
) -> dict:
    """Write train/valid/test in the chat format mlx-lm expects.

    Whole destinations are held out for test (generalization, not memorization), and whole
    conversations go to valid so turns of one conversation never straddle train and valid.
    """
    rows, seen = [], set()
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["meta"]["scenario"], row["meta"]["turn"], row["meta"]["city"])
        if key in seen:  # a scenario interrupted by a restart can be written twice
            continue
        seen.add(key)
        row = clean_sample(row)
        # Filters added after the first runs, applied to older samples here.
        if reject_reason(
            {
                "turn": AgentTurn.model_validate_json(row["messages"][-1]["content"]),
                "violations": [],
            },
            row["meta"]["intent"],
            feasible="NOT FEASIBLE" not in row["messages"][-2]["content"],
        ):
            continue
        rows.append(row)
    buckets: dict[str, list] = {"train": [], "valid": [], "test": []}
    for row in rows:
        meta = row["meta"]
        if test_destinations & set(meta["destinations"]):
            name = "test"
        elif random.Random(f"valid-{meta['scenario']}").random() < valid_share:
            name = "valid"
        else:
            name = "train"
        # Follow-ups that should edit the plan are outnumbered by ones that should not
        # (questions, pushback), and a small model then learns "follow-up -> plan: null".
        # Repeating the edits in train only rebalances that without touching valid/test.
        copies = edit_weight if name == "train" and meta["intent"] in EDIT_INTENTS else 1
        buckets[name].extend({"messages": row["messages"]} for _ in range(copies))
    for name, items in buckets.items():
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {name: len(items) for name, items in buckets.items()}
