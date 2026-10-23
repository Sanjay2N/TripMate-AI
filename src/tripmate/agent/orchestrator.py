"""One user turn: retrieve -> prompt -> model -> validate -> (repair once) -> update state.

A trip is planned one city segment at a time, so the model only ever sees one city's places.
"""

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from tripmate.agent.budget import Budget, breakdown, envelope
from tripmate.agent.prompts import build_messages, repair_message, shown_guides, shown_ids
from tripmate.agent.route import build_route
from tripmate.agent.validator import CHANGE_CLAIM, MAX_PER_DAY, validate
from tripmate.config import Settings, get_settings
from tripmate.llm.client import LLM
from tripmate.rag.retriever import Retriever
from tripmate.schemas import AgentTurn, HeadsUp, Segment, TripRequest, turn_schema
from tripmate.state import TripState, clean_turn, diff_plans, merge_plan

ACTIVITY_KINDS = ("see", "do", "buy", "eat", "drink")  # never schedule hotels as activities
GUIDE_QUERIES = ("safety scams warnings", "weather climate season best time to visit")
FIRST_MESSAGE = "Please plan this trip for me."
CLAIM_WITHOUT_PLAN = (
    "Your reply says you changed the plan, but plan is null. Return the changed days in plan, "
    "matching what your reply describes."
)
MAX_HEADS_UP = 6


@dataclass
class TurnResult:
    reply: str = ""
    ask: str = ""
    parsed: bool = True  # every model response matched the schema
    applied: bool = False  # at least one proposed change was written into state
    violations: list[str] = field(default_factory=list)  # still failing after repair
    first_pass_violations: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)  # what changed in the plan (PRD 6.6)
    attempts: int = 0
    seconds: float = 0.0
    completion_tokens: int = 0
    raw: list[str] = field(default_factory=list)


def load_places(settings: Settings) -> dict:
    return json.loads((settings.kb_dir / "places.json").read_text())


def parse_turn(text: str) -> AgentTurn | None:
    """Without grammar-constrained decoding a model may wrap the JSON in a code fence or
    add a sentence around it; take the outermost {...} so format is judged fairly."""
    start, end = text.find("{"), text.rfind("}")
    try:
        return AgentTurn.model_validate_json(text[start : end + 1] if start >= 0 else text)
    except ValidationError:
        return None


class Agent:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        settings: Settings | None = None,
        style: str = "long",
        repair: bool = True,
        constrained: bool = True,
    ):
        self.s = settings or get_settings()
        self.retriever = retriever
        self.llm = llm
        self.places = load_places(self.s)
        self.style = style
        self.repair = repair
        self.constrained = constrained
        # When set to a list, every model call is recorded as a training sample candidate.
        self.recorder: list | None = None
        # Data generation only: extra guidance appended to the teacher's prompt for one turn.
        # The recorded student prompt never contains it, so the student learns to spot the
        # situation on its own.
        self.teacher_note: str | None = None

    def city_weights(self) -> dict[str, float]:
        """Number of things to see and do per city; drives how days are split."""
        weights: dict[str, float] = {}
        for d in self.retriever.docs.values():
            if d.get("kind") in ("see", "do"):
                weights[d["destination"]] = weights.get(d["destination"], 0) + 1
        return weights

    def start(self, request: TripRequest) -> TripState:
        """Create the state: fix the route, then pre-load candidates for each city
        (one search per interest plus must-sees, and the risk/weather guide notes)."""
        state = TripState(request=request)
        state.route = build_route(request, self.places, self.city_weights())
        for seg in state.route:
            self.load_candidates(state, seg.city)
        return state

    def load_candidates(self, state: TripState, city: str) -> None:
        for interest in state.request.interests:
            for hit in self.retriever.search(interest, city, kinds=ACTIVITY_KINDS, k=6):
                state.register_poi(hit.doc, interest)
        for hit in self.retriever.search(
            f"top attractions and landmarks in {city}", city, kinds=("see", "do"), k=8
        ):
            state.register_poi(hit.doc)
        for query in GUIDE_QUERIES:
            for hit in self.retriever.search(query, city, doc_type="guide", k=2):
                state.register_guide(hit.doc)

    def budget(self, state: TripState) -> Budget | None:
        if not state.plan:
            return None
        return breakdown(state.request, state.route, state.plan, self.places)

    def targets(self, state: TripState, message: str) -> list[Segment]:
        """Which cities a message is about: unplanned ones first, else by 'day N' or city
        name mentioned, else all of them."""
        planned = state.planned_days()
        unplanned = [s for s in state.route if not planned & set(s.days)]
        if unplanned:
            return unplanned
        days = {int(n) for n in re.findall(r"\bday\s*(\d+)", message, flags=re.I)}
        by_day = [s for s in state.route if days & set(s.days)]
        by_name = [s for s in state.route if s.city.lower() in message.lower()]
        return by_day or by_name or list(state.route)

    def turn(
        self,
        state: TripState,
        user_message: str = FIRST_MESSAGE,
        segments: list[Segment] | None = None,
    ) -> TurnResult:
        before = state.plan.model_copy(deep=True) if state.plan else None
        result = TurnResult()
        replies, heads_up = [], []
        targets = segments or self.targets(state, user_message)
        for seg in targets:
            turn = self._segment_turn(state, seg, user_message, result)
            if turn is None:
                continue
            prefix = f"[{seg.city}] " if len(targets) > 1 else ""
            replies.append(prefix + turn.reply)
            result.ask = result.ask or turn.ask
            heads_up += turn.heads_up
        result.reply = "\n\n".join(replies)
        result.changes = diff_plans(before, state.plan, state.pois)
        if heads_up:
            # Newest first, one per source, so rephrasings of the same warning do not pile up.
            by_source: dict[str, HeadsUp] = {}
            for h in heads_up + state.heads_up:
                by_source.setdefault(h.source, h)
            state.heads_up = list(by_source.values())[:MAX_HEADS_UP]
        if replies:
            state.history += [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": result.reply},
            ]
        return result

    def _segment_turn(
        self, state: TripState, seg: Segment, user_message: str, result: TurnResult
    ) -> AgentTurn | None:
        first_plan = not state.planned_days() & set(seg.days)
        fresh = []
        if not first_plan:  # refinement: look for places matching the new request
            hits = self.retriever.search(user_message, seg.city, kinds=ACTIVITY_KINDS, k=6)
            fresh = [state.register_poi(h.doc) for h in hits]

        shown = shown_ids(state, seg, fresh)
        guides = shown_guides(state, seg)
        messages = build_messages(state, seg, user_message, shown, self.places, self.style)
        student_messages = build_messages(state, seg, user_message, shown, self.places, "short")
        if self.teacher_note:
            last = messages[-1]
            messages[-1] = {**last, "content": f"{last['content']}\n\n{self.teacher_note}"}
        feasible = envelope(state.request, state.route, self.places)["feasible"]

        def schema_for(require_plan: bool = False) -> dict | None:
            if not self.constrained:
                return None
            return turn_schema(
                shown,
                guides,
                first_plan=first_plan and feasible,
                n_days=len(seg.days),
                max_activities=MAX_PER_DAY[state.request.pace],
                require_plan=require_plan,
            )

        schema = schema_for()
        expected = state.planned_days() | set(seg.days)

        turn, candidate, violations = None, None, []
        for attempt in range(2 if self.repair else 1):
            done = self.llm.chat(messages, schema=schema)
            result.attempts += 1
            result.seconds += done.seconds
            result.completion_tokens += done.completion_tokens
            result.raw.append(done.text)
            turn = parse_turn(done.text)
            if turn is not None:
                turn = clean_turn(turn, state.pois, state.guides)
            if turn is None:
                candidate = None
                violations = ["Response was not valid JSON for the expected schema."]
            elif turn.plan is None:
                candidate, violations = None, []
            else:
                candidate = merge_plan(state.plan, turn.plan, seg)
                if candidate == state.plan:
                    # The model re-sent the plan unchanged (common when it is only answering
                    # a question): that is "no change", so record it as such.
                    turn.plan, candidate, violations = None, None, []
                else:
                    violations = validate(state, candidate, self.places, expected)
            if turn is not None and turn.plan is None and not first_plan:
                if CHANGE_CLAIM.search(turn.reply):
                    # A small model often says "I added X" and then emits no change. Hold it
                    # to its own words: the repair turn must contain a plan.
                    violations = [CLAIM_WITHOUT_PLAN]
                    schema = schema_for(require_plan=True)
            if attempt == 0:
                result.first_pass_violations += violations
            if not violations:
                break
            messages = messages + [
                {"role": "assistant", "content": done.text},
                {"role": "user", "content": repair_message(violations)},
            ]

        result.violations += violations
        if turn is None:
            result.parsed = False
            return None
        # Clean up what the model may get slightly wrong, so the UI, the history and any
        # training sample built from this turn all see the corrected version.
        turn.remember = state.remember(turn.remember)
        # A city's first plan is shown even with violations (flagged in the UI); a refinement
        # that breaks a constraint is rejected so a valid plan is never made worse.
        if candidate is not None and (not violations or first_plan):
            state.plan = candidate
            result.applied = True
        known = set(shown) | set(guides)
        turn.heads_up = [h for h in turn.heads_up if h.source in known]
        if self.recorder is not None:
            self.recorder.append(
                {
                    "messages": student_messages,
                    "turn": turn,
                    "city": seg.city,
                    "violations": violations,
                    "repaired": attempt > 0,
                }
            )
        return turn
