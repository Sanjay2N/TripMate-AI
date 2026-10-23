"""TripState is the agent's memory. It is rendered into every prompt, so the model never has
to recall a constraint from early in a long conversation; it only has to respect what it is shown.
"""

import re

from pydantic import BaseModel

from tripmate.schemas import AgentTurn, CityCosts, HeadsUp, Plan, PlanDraft, Segment, TripRequest

HISTORY_TURNS = 4  # most recent user/assistant messages replayed verbatim


class TripState(BaseModel):
    request: TripRequest
    route: list[Segment] = []
    plan: Plan | None = None
    heads_up: list[HeadsUp] = []
    history: list[dict] = []  # {"role": "user" | "assistant", "content": str}
    # Stable short ids. A place keeps its id for the whole conversation because the
    # current plan refers to it.
    pois: dict[str, dict] = {}  # "P1" -> place record
    guides: dict[str, dict] = {}  # "G1" -> guide chunk
    matched: dict[str, list[str]] = {}  # "P1" -> interests whose search returned it

    def register_poi(self, doc: dict, interest: str | None = None) -> str:
        alias = next((a for a, d in self.pois.items() if d["id"] == doc["id"]), None)
        if alias is None:
            alias = f"P{len(self.pois) + 1}"
            self.pois[alias] = doc
        if interest and interest not in self.matched.setdefault(alias, []):
            self.matched[alias].append(interest)
        return alias

    def register_guide(self, doc: dict) -> str:
        alias = next((a for a, d in self.guides.items() if d["id"] == doc["id"]), None)
        if alias is None:
            alias = f"G{len(self.guides) + 1}"
            self.guides[alias] = doc
        return alias

    def day_city(self, day: int) -> str | None:
        return next((s.city for s in self.route if day in s.days), None)

    def planned_days(self) -> set[int]:
        return {d.day for d in self.plan.days} if self.plan else set()

    def planned_ids(self, days: list[int] | None = None) -> list[str]:
        if not self.plan:
            return []
        return [
            a.poi for d in self.plan.days if days is None or d.day in days for a in d.activities
        ]

    def remember(self, items: list[str]) -> list[str]:
        """Add lasting constraints; returns only the ones that were actually new."""
        known = {c.lower() for c in self.request.constraints}
        added = []
        for item in items:
            item = item.strip()
            if item and item.lower() not in known:
                self.request.constraints.append(item)
                known.add(item.lower())
                added.append(item)
        return added

    def name_ids(self, text: str) -> str:
        return name_ids(text, self.pois, self.guides)


GUIDE_CITATION = re.compile(
    r"\s*[(\[]\s*(?:sources?\s*:?\s*|see\s+|per\s+)?G\d+(?:\s*(?:,|/|&|and)\s*G\d+)*\s*[)\]]",
    flags=re.IGNORECASE,
)


def name_ids(text: str, pois: dict[str, dict], guides: dict[str, dict]) -> str:
    """Replace ids that slipped into prose with what they stand for.

    Place ids become the place's name ("visit P3" -> "visit City Palace"); models often
    write the id next to the name ("P3 City Palace", "P3 (City Palace)"), so the repeated
    name is dropped. Guide-note ids are citations: in brackets they are removed
    ("scams (Source: G1)" -> "scams"), elsewhere they become "the Anjuna safety notes".
    """

    def guide_label(m: re.Match) -> str:
        g = guides.get(m.group())
        return f"the {g['area']} {g['section'].lower()} notes" if g else m.group()

    text = GUIDE_CITATION.sub("", text)
    text = re.sub(r"\bG\d+\b", guide_label, text)
    out, pos = [], 0
    for m in re.finditer(r"\bP\d+\b", text):
        if m.group() not in pois:
            continue
        name = pois[m.group()]["name"]
        out.append(text[pos : m.start()] + name)
        pos = m.end()
        words = name.split()
        echoes = [f" ({name})", f" - {name}", f": {name}"]
        echoes += [" " + " ".join(words[i:]) for i in range(len(words))]
        echo = next((e for e in echoes if text[pos:].lower().startswith(e.lower())), "")
        pos += len(echo)
    return "".join(out) + text[pos:]


def clean_turn(turn: AgentTurn, pois: dict[str, dict], guides: dict[str, dict]) -> AgentTurn:
    """Name every id the model left in free text: reply, question, warnings, themes, reasons.

    Id fields (activity.poi, heads_up.source, drop_if_short, ...) are left untouched.
    """
    turn = turn.model_copy(deep=True)
    turn.reply = name_ids(turn.reply, pois, guides)
    turn.ask = name_ids(turn.ask, pois, guides)
    for h in turn.heads_up:
        h.risk = name_ids(h.risk, pois, guides)
        h.mitigation = name_ids(h.mitigation, pois, guides)
    for d in turn.plan.days if turn.plan else []:
        d.theme = name_ids(d.theme, pois, guides)
        for a in d.activities:
            a.why = name_ids(a.why, pois, guides)
    return turn


def merge_plan(old: Plan | None, draft: PlanDraft, seg: Segment) -> Plan:
    """Apply a draft for one city: replace the days it returns, keep everything else.

    Days outside the segment are ignored, so planning one city can never disturb another.
    """
    merged = old.model_copy(deep=True) if old else Plan()
    by_day = {d.day: d for d in merged.days}
    by_day.update({d.day: d for d in draft.days if d.day in seg.days})
    merged.days = sorted(by_day.values(), key=lambda d: d.day)
    costs = merged.cities.setdefault(seg.city, CityCosts())
    for field in ("stay", "food_per_day_inr", "local_transport_per_day_inr"):
        if getattr(draft, field) is not None:
            setattr(costs, field, getattr(draft, field))
    return merged


def diff_plans(old: Plan | None, new: Plan | None, pois: dict[str, dict]) -> list[str]:
    """Human-readable list of what changed between two plans (PRD 6.6)."""
    if new is None:
        return []

    def name(alias: str) -> str:
        return pois[alias]["name"] if alias in pois else alias

    old_days = {d.day: d for d in old.days} if old else {}
    new_days = {d.day: d for d in new.days}
    changes = []
    for day in sorted(set(old_days) | set(new_days)):
        if day not in new_days:
            changes.append(f"Day {day}: removed")
            continue
        if day not in old_days:
            places = ", ".join(name(a.poi) for a in new_days[day].activities)
            changes.append(f"Day {day}: new - {places}")
            continue
        before = {a.poi: a for a in old_days[day].activities}
        after = {a.poi: a for a in new_days[day].activities}
        parts = [f"added {name(p)}" for p in after if p not in before]
        parts += [f"removed {name(p)}" for p in before if p not in after]
        parts += [
            f"{name(p)} moved {before[p].start} -> {after[p].start}"
            for p in after
            if p in before and before[p].start != after[p].start
        ]
        if parts:
            changes.append(f"Day {day}: " + "; ".join(parts))
    old_cities = old.cities if old else {}
    for city, costs in new.cities.items():
        prev = old_cities.get(city)
        if prev is None:
            continue
        if costs.stay and prev.stay and costs.stay.per_night_inr != prev.stay.per_night_inr:
            changes.append(
                f"{city} stay: ₹{prev.stay.per_night_inr} -> ₹{costs.stay.per_night_inr} per night"
            )
        if costs.food_per_day_inr != prev.food_per_day_inr:
            changes.append(
                f"{city} food: ₹{prev.food_per_day_inr} -> ₹{costs.food_per_day_inr} per person/day"
            )
    return changes


def experience_score(state: TripState) -> int | None:
    """How well the plan matches stated interests (0-100). Explainable by construction:
    70% = share of interests with at least one matching activity, 30% = share of activities
    that match any interest."""
    interests = state.request.interests
    planned = state.planned_ids()
    if not interests or not planned:
        return None
    covered = {i for p in planned for i in state.matched.get(p, [])}
    on_interest = sum(1 for p in planned if state.matched.get(p))
    return round(100 * (0.7 * len(covered) / len(interests) + 0.3 * on_interest / len(planned)))
