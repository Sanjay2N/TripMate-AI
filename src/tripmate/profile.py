"""Travel profile (US2.2): preferences saved once and reused to pre-fill every new trip.

No accounts: a profile is a small JSON file. Locally it lives under data/profiles/; on a host
with an ephemeral disk the UI offers download and upload instead.
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from tripmate.llm.client import LLM
from tripmate.schemas import Mode, Pace, Party, Tier


class TravelProfile(BaseModel):
    name: str = "me"
    home_city: str = "Bengaluru"
    party: Party = Party.couple
    travelers: int = Field(default=2, ge=1, le=12)
    mode: Mode = Mode.train
    tier: Tier = Tier.balanced
    pace: Pace = Pace.moderate
    interests: list[str] = []
    constraints: list[str] = Field(default=[], description="Lasting non-negotiables")
    notes: str = Field(default="", description="Free text about past trips and tastes")

    def trip_defaults(self) -> dict:
        """Fields that pre-fill a TripRequest."""
        return {
            "source": self.home_city,
            "party": self.party,
            "travelers": self.travelers,
            "mode": self.mode,
            "tier": self.tier,
            "pace": self.pace,
            "interests": list(self.interests),
            "constraints": list(self.constraints),
        }

    def learn(self, constraints: list[str]) -> list[str]:
        """Promote constraints picked up during a trip. Returns the ones that were new."""
        known = {c.lower() for c in self.constraints}
        new = [c for c in constraints if c.lower() not in known]
        self.constraints += new
        return new


def _path(directory: Path, name: str) -> Path:
    return directory / (re.sub(r"[^a-z0-9_-]+", "_", name.lower()) + ".json")


def save_profile(profile: TravelProfile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = _path(directory, profile.name)
    path.write_text(profile.model_dump_json(indent=2))
    return path


def load_profile(name: str, directory: Path) -> TravelProfile | None:
    path = _path(directory, name)
    return TravelProfile.model_validate_json(path.read_text()) if path.exists() else None


class ProfileFacts(BaseModel):
    """What the model extracts from a free-text description (PRD 6.1)."""

    pace: Pace
    interests: list[str]
    constraints: list[str]


EXTRACT_PROMPT = (
    "Extract travel preferences from the traveler's description of themselves and past trips. "
    "pace: relaxed, moderate or packed. interests: short topics they enjoy (e.g. beaches, "
    "history, food). constraints: lasting non-negotiables only (e.g. vegetarian, no early "
    "mornings). Do not invent anything that is not stated or clearly implied. Answer in JSON."
)


def extract_profile(llm: LLM, text: str) -> ProfileFacts | None:
    done = llm.chat(
        [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text}],
        schema=ProfileFacts.model_json_schema(),
        temperature=0,
        max_tokens=300,
    )
    try:
        return ProfileFacts.model_validate_json(done.text)
    except ValueError:
        return None
