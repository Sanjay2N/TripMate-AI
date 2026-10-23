"""Sample trips and scripted user turns for synthetic conversations.

Everything is driven by a seeded RNG, so scenario i is the same on every run and a long
generation job can be stopped and resumed.
"""

import random
from dataclasses import dataclass

from tripmate.agent.budget import TIER_FRACTION, envelope, intercity_cost
from tripmate.agent.route import build_route
from tripmate.agent.validator import avoid_terms
from tripmate.schemas import Mode, Pace, Party, Tier, TripRequest
from tripmate.state import TripState

INTERESTS = [
    "beaches", "history", "temples", "food", "nature", "wildlife", "adventure", "shopping",
    "art and museums", "architecture", "viewpoints", "local culture", "markets", "gardens",
]  # fmt: skip

# constraint as stored -> how a traveler would say it mid-conversation
CONSTRAINTS = {
    "vegetarian": "we're all vegetarian",
    "no early mornings": "we don't do early mornings, nothing before 10 please",
    "no temples": "we'd rather skip temples altogether",
    "no museums": "museums bore the kids, so none of those",
    "avoid crowded places": "we really dislike crowded places",
    "minimal walking": "my mother can't walk long distances",
    "no alcohol": "none of us drink, so no bars or pubs",
    "travelling with a toddler": "we'll have our two-year-old with us",
    "no shopping": "we're not interested in shopping stops",
}

TEMPLATES = {
    "relax_day": [
        "Day {day} looks too packed. Can you make it lighter?",
        "Can we slow down day {day}? Fewer stops please.",
        "make day {day} more relaxed",
        "I think day {day} is a bit much for us. Trim it down?",
    ],
    "add_interest": [
        "Could you add something for {interest} on day {day}?",
        "We'd also love some {interest}. Fit it in on day {day} if you can.",
        "any chance of squeezing {interest} into day {day}?",
    ],
    "swap": [
        "I'm not keen on {place}. Swap it for something else.",
        "Skip {place}. What else could we do instead?",
        "we've already been to {place}, please replace it",
    ],
    "new_constraint": [
        "By the way, {said}. Please keep that in mind.",
        "One thing I forgot to mention: {said}.",
        "Oh, and {said}.",
    ],
    "conflict_constraint": [
        "Actually, add a {term} visit on day {day}.",
        "Can you put a {term} into day {day}?",
    ],
    "conflict_budget": [
        "Upgrade our stay to a luxury resort at around ₹{price} a night.",
        "Let's book a five-star place, something like ₹{price} per night.",
    ],
    "question": [
        "Is it safe there? Anything we should watch out for?",
        "What's the weather like, and is this a good time to go?",
        "Any local customs or dress rules we should know about?",
        "Do we need any permits, and are ATMs easy to find?",
        "What food should we not miss there?",
        "How is the mobile network and internet there?",
    ],
}
FOLLOW_UPS = ["relax_day", "add_interest", "swap", "new_constraint", "question", "question"]
CONFLICTS = ["conflict_constraint", "conflict_budget"]
# avoid-term -> what a traveler would ask for that breaks it ("no shopping" -> "shopping market")
CONFLICT_ASKS = {
    "temple": "temple",
    "museum": "museum",
    "shopping": "shopping market",
    "alcohol": "bar",
}
FOCUS_CONSTRAINTS = ["no temples", "no museums", "no shopping", "no alcohol"]


@dataclass
class Scenario:
    id: int
    request: TripRequest
    feasible: bool
    n_follow_ups: int


def sample_scenario(
    i: int, seed: int, places: dict, cities: list[str], focus: str | None = None
) -> Scenario:
    """focus="conflict" makes every trip carry a constraint the user will later contradict,
    and raises the share of trips whose budget cannot cover the travel at all."""
    rng = random.Random(f"{seed}-{i}" if focus is None else f"{seed}-{focus}-{i}")
    sources = [p for p in places if p not in cities]
    n_cities = 2 if rng.random() < 0.15 else 1
    destinations = rng.sample(cities, n_cities)
    days = rng.randint(2, 3) if n_cities == 1 else rng.randint(3, 4)
    party = rng.choice(list(Party))
    travelers = {Party.solo: 1, Party.couple: 2}.get(party) or rng.randint(3, 5)
    tier = rng.choice(list(Tier))
    request = TripRequest(
        source=rng.choice(sources),
        destinations=destinations,
        days=days,
        travelers=travelers,
        party=party,
        mode=rng.choice([Mode.train, Mode.train, Mode.bus, Mode.air, Mode.car]),
        budget_inr=1,
        tier=tier,
        pace=rng.choice(list(Pace)),
        interests=rng.sample(INTERESTS, rng.randint(1, 3)),
        constraints=rng.sample(list(CONSTRAINTS), rng.choice([0, 0, 1, 1, 2])),
    )
    if focus == "conflict" and not set(request.constraints) & set(FOCUS_CONSTRAINTS):
        request.constraints.append(rng.choice(FOCUS_CONSTRAINTS))
    route = build_route(request, places, {})
    intercity = intercity_cost(request, route, places)
    feasible = rng.random() >= (0.25 if focus == "conflict" else 0.08)
    if feasible:
        per_person_day = rng.randint(15, 60) * 100
        ceiling = (intercity + per_person_day * days * travelers) / TIER_FRACTION[tier]
    else:  # deliberately too small: the right answer is to push back, not to plan
        ceiling = intercity * rng.uniform(0.5, 1.0) / TIER_FRACTION[tier]
    request.budget_inr = max(round(ceiling / 1000) * 1000, 1000)
    feasible = envelope(request, route, places)["feasible"]
    follow_ups = (3 if focus == "conflict" else rng.randint(2, 4)) if feasible else 0
    return Scenario(i, request, feasible, follow_ups)


def conflict_terms(constraints: list[str]) -> list[str]:
    return [t for t in avoid_terms(constraints) if t in CONFLICT_ASKS]


def next_user_turn(
    rng: random.Random,
    state: TripState,
    places: dict,
    questions: list[str],
    used: set[str],
    focus: str | None = None,
) -> tuple[str, str]:
    """Pick the next scripted user message given the current plan. Returns (intent, text).

    With focus="conflict" both conflicts come first, then one ordinary follow-up, so the
    model also learns to carry on normally after pushing back."""
    req = state.request
    options = [i for i in FOLLOW_UPS if i not in used or i == "question"]
    if focus == "conflict" or rng.random() < 0.25:
        options = [c for c in CONFLICTS if c not in used] or options
    intent = rng.choice(options)
    day = rng.choice(sorted(state.planned_days()) or [1])
    planned = state.planned_ids()

    if intent == "swap" and not planned:
        intent = "relax_day"
    if intent == "conflict_constraint" and not conflict_terms(req.constraints):
        intent = "conflict_budget"
    free = [c for c in CONSTRAINTS if c not in req.constraints]
    if intent == "new_constraint" and not free:
        intent = "question"

    fields = {"day": day}
    if intent == "add_interest":
        fields["interest"] = rng.choice([i for i in INTERESTS if i not in req.interests])
    elif intent == "swap":
        fields["place"] = state.pois[rng.choice(planned)]["name"]
    elif intent == "new_constraint":
        fields["said"] = CONSTRAINTS[rng.choice(free)]
    elif intent == "conflict_constraint":
        fields["term"] = CONFLICT_ASKS[rng.choice(conflict_terms(req.constraints))]
    elif intent == "conflict_budget":
        cap = envelope(req, state.route, places)["stay_per_night"]
        fields["price"] = max(round(cap * rng.uniform(4, 8) / 1000) * 1000, 15000)
    if intent == "question" and questions and rng.random() < 0.3:
        return intent, rng.choice(questions)
    return intent, rng.choice(TEMPLATES[intent]).format(**fields)
