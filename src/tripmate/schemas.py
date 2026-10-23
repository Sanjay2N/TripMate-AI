from enum import StrEnum

from pydantic import BaseModel, Field


class Pace(StrEnum):
    relaxed = "relaxed"
    moderate = "moderate"
    packed = "packed"


class Party(StrEnum):
    solo = "solo"
    couple = "couple"
    family = "family"
    friends = "friends"


class Mode(StrEnum):
    bus = "bus"
    train = "train"
    air = "air"
    car = "car"


class Tier(StrEnum):
    """Plan variant (PRD US1.2): how much of the budget ceiling the plan may use."""

    budget = "budget"
    balanced = "balanced"
    premium = "premium"


class TripRequest(BaseModel):
    source: str
    destinations: list[str] = Field(min_length=1, description="One city, or several (US2.4)")
    days: int = Field(ge=1, le=14, description="Total days across all cities")
    days_per_city: dict[str, int] = Field(
        default={}, description="Optional fixed split; otherwise decided by the route planner"
    )
    travelers: int = Field(default=2, ge=1, le=12)
    party: Party = Party.couple
    mode: Mode = Mode.train
    budget_inr: int = Field(gt=0, description="Hard ceiling for the whole trip, all travelers")
    tier: Tier = Tier.balanced
    pace: Pace = Pace.moderate
    interests: list[str] = []
    constraints: list[str] = Field(
        default=[], description="Non-negotiables, e.g. 'vegetarian', 'no temples'"
    )


# ---- What the model writes. Field order is generation order: plan first, prose last. ----


class Activity(BaseModel):
    poi: str = Field(description="Place id from PLACES, e.g. P3")
    start: str = Field(description="24h HH:MM")
    duration_min: int
    cost_inr: int = Field(description="Per person")
    why: str


class DayPlan(BaseModel):
    day: int
    theme: str
    activities: list[Activity]
    drop_if_short: str = Field(description="Id of the activity to drop if short on time, or ''")
    add_if_extra: str = Field(
        description="Id of an unused place to add if there is spare time, or ''"
    )


class Stay(BaseModel):
    area: str
    per_night_inr: int = Field(description="Per room")
    why: str


class PlanDraft(BaseModel):
    """What the model writes for the city it is currently planning."""

    stay: Stay | None = None
    food_per_day_inr: int | None = Field(default=None, description="Per person")
    local_transport_per_day_inr: int | None = Field(default=None, description="Whole group")
    days: list[DayPlan] = []


class HeadsUp(BaseModel):
    risk: str
    mitigation: str
    source: str = Field(description="Id of the PLACES or GUIDE NOTES entry this is based on")


class AgentTurn(BaseModel):
    """The response envelope for every turn.

    reply comes first on purpose: the model says what it is doing before it writes the plan,
    so whether to edit is decided after reading and answering the user, not at one early
    token where the majority case ("no change") wins. That ordering also keeps the reply and
    the plan consistent.
    """

    reply: str
    ask: str = Field(description="One clarifying question if essential info is missing, else ''")
    remember: list[str] = Field(description="New lasting constraints the user just stated")
    plan: PlanDraft | None = Field(
        description="Full plan, or only the changed days; null if no change"
    )
    heads_up: list[HeadsUp]


# ---- What the system keeps. ----


class Segment(BaseModel):
    """A consecutive run of days spent in one city."""

    city: str
    days: list[int]


class CityCosts(BaseModel):
    stay: Stay | None = None
    food_per_day_inr: int | None = None
    local_transport_per_day_inr: int | None = None


class Plan(BaseModel):
    """The whole trip: per-city costs plus every day, merged from the model's drafts."""

    cities: dict[str, CityCosts] = {}
    days: list[DayPlan] = []


def turn_schema(
    poi_ids: list[str],
    guide_ids: list[str],
    first_plan: bool = False,
    n_days: int = 14,
    max_activities: int = 7,
    require_plan: bool = False,
) -> dict:
    """JSON schema for one turn, with id fields narrowed to the ids shown in the prompt.

    Under grammar-constrained decoding this makes an invented place id unrepresentable.
    On a city's first plan neither the plan nor its cost fields may be null, so the model
    cannot answer with a question instead of planning, or skip stay and food.
    Array sizes and text lengths are bounded so a weak model cannot ramble past the token limit.
    """
    schema = AgentTurn.model_json_schema()
    defs = schema["$defs"]
    draft = defs["PlanDraft"]
    draft["required"] = list(draft["properties"])
    draft["properties"]["days"] |= {"maxItems": n_days, "minItems": n_days if first_plan else 0}
    defs["DayPlan"]["properties"]["activities"] |= {"minItems": 1, "maxItems": max_activities}
    defs["DayPlan"]["properties"]["theme"]["maxLength"] = 60
    defs["Activity"]["properties"]["why"]["maxLength"] = 160
    defs["Activity"]["properties"]["duration_min"] |= {"minimum": 15, "maximum": 480}
    defs["Activity"]["properties"]["cost_inr"] |= {"minimum": 0, "maximum": 20000}
    defs["Stay"]["properties"]["why"]["maxLength"] = 160
    for key in ("risk", "mitigation"):
        defs["HeadsUp"]["properties"][key]["maxLength"] = 200
    schema["properties"]["heads_up"]["maxItems"] = 3
    # The opening request is written by the system, so it cannot contain a new constraint.
    schema["properties"]["remember"]["maxItems"] = 0 if first_plan else 4
    schema["properties"]["ask"]["maxLength"] = 200
    schema["properties"]["reply"]["maxLength"] = 700
    if first_plan or require_plan:  # require_plan: a repair after "I changed it" with no plan
        schema["properties"]["plan"] = {"$ref": "#/$defs/PlanDraft"}
        draft["properties"]["days"]["minItems"] = n_days if first_plan else 1
    if first_plan:
        draft["properties"]["stay"] = {"$ref": "#/$defs/Stay"}
        for key in ("food_per_day_inr", "local_transport_per_day_inr"):
            draft["properties"][key] = {"type": "integer"}
    defs["Activity"]["properties"]["poi"]["enum"] = poi_ids
    for key in ("drop_if_short", "add_if_extra"):
        defs["DayPlan"]["properties"][key]["enum"] = ["", *poi_ids]
    defs["HeadsUp"]["properties"]["source"]["enum"] = [*guide_ids, *poi_ids]
    return schema
