"""Destination profiles from the Kaggle "Indian Tourism Dataset" (Apache-2.0).

One record per destination: seasons, safety, customs, connectivity, typical daily costs.
It complements Wikivoyage, which has the individual places but little structured
destination-level information.

Trust level: the dataset cites unresolvable "web:NNN" sources and every record carries the
same date, so it reads as machine-compiled. Chunks are therefore tagged verified=False and
shown as unverified in the UI. Fields that name places (attractions, hidden gems, suggested
itinerary) are deliberately left out: the model may only plan with places it can cite by id.
"""

import json
from pathlib import Path

import yaml

SOURCE_URL = "https://www.kaggle.com/datasets/sushanthnaidu24/indian-tourism-dataset"
TIERS = {
    "budget": "budget_category",
    "balanced": "mid_range_category",
    "premium": "luxury_category",
}


def _join(value) -> str:
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if isinstance(value, dict):
        return "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in value.items())
    return str(value)


def sections(r: dict) -> dict[str, str]:
    permits = r["permits_details"] if r["permits_required"] else "none required"
    return {
        "Seasons and weather": (
            f"Best seasons: {_join(r['best_seasons'])}. Avoid: {_join(r['avoid_seasons'])}. "
            f"Peak tourist season: {r['peak_tourist_season']}. Off season: {r['off_season']}. "
            f"Temperatures: {_join(r['average_temperature'])}. Rainfall: {r['rainfall_pattern']}. "
            f"Note: {r['special_considerations']}"
        ),
        "Safety and practicalities": (
            f"Safety: {r['safety_notes']} (rated {r['safety_rating']}/10). Permits: {permits}. "
            f"ATMs: {r['atm_availability']} Internet: {r['internet_connectivity']} "
            f"Mobile networks: {_join(r['mobile_network'])}. "
            f"Accessibility: {r['accessibility']}; roads: {r['road_connectivity']}."
        ),
        "Customs and culture": (
            f"Customs: {r['local_customs']} Culture: {r['local_culture']} "
            f"Festivals: {r['festivals_events']} Languages: {_join(r['language_spoken'])}."
        ),
        "Getting there": (
            f"Nearest airport: {_join(r['nearest_airport'])}. "
            f"Nearest railway station: {_join(r['nearest_railway_station'])}. "
            f"Nearest major city: {_join(r['nearest_major_city'])}."
        ),
        "Food and shopping": (
            f"Food: {r['food_scene']} Must try: {_join(r['local_cuisine_must_try'])}. "
            f"Shopping: {_join(r['shopping_highlights'])}"
        ),
        "Who it suits": (
            f"{_join(r['ideal_for_why'])}. Typical stay: {r['minimum_days']}-{r['maximum_days']} "
            f"days, ideally {r['ideal_days']}."
        ),
    }


def typical_costs(r: dict) -> dict:
    """Daily price ranges per plan tier, used as priors in the prompt (never as quotes)."""
    return {
        tier: {
            "stay": r[key]["accommodation_range"],
            "food": r[key]["food_range"],
            "local": r[key]["local_transport_range"],
            "activities": r[key]["activities_range"],
        }
        for tier, key in TIERS.items()
    }


def build_profiles(dataset: Path, destinations_file: Path, kb_dir: Path, log=print) -> int:
    """Write kb/profiles.jsonl and add typical costs to kb/places.json for matched cities."""
    records = {}
    for r in json.loads(dataset.read_text()):
        records.setdefault(r["destination_name"], r)  # the dataset repeats a few names
    cfg = yaml.safe_load(destinations_file.read_text())
    places_path = kb_dir / "places.json"
    places = json.loads(places_path.read_text())

    chunks = []
    for dest in cfg["destinations"]:
        r = records.get(dest.get("profile", dest["name"]))
        if r is None:
            log(f"  ! no profile for {dest['name']}")
            continue
        for i, (section, text) in enumerate(sections(r).items()):
            chunks.append(
                {
                    "id": f"00000000-0000-4000-8000-{r['id']:06d}{i:06d}",
                    "doc_type": "guide",
                    "source": "kaggle",
                    "verified": False,
                    "destination": dest["name"],
                    "area": dest["name"],
                    "section": section,
                    "url": SOURCE_URL,
                    "text": f"{dest['name']} - {section}: {text}",
                }
            )
        places[dest["name"]]["typical"] = typical_costs(r)

    with (kb_dir / "profiles.jsonl").open("w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    places_path.write_text(json.dumps(places, indent=2, ensure_ascii=False))
    log(f"profiles: {len(chunks)} chunks for {len(chunks) // 6} destinations")
    return len(chunks)
