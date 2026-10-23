"""Real traveler questions from the HF dataset soniawmeyer/travel-conversations-finetuning.

Only the questions are used, as wording for the simulated user. The answers are not: many are
generic chatbot text ("As an AI language model...") written without any retrieved evidence,
which is the opposite of what TripMate should learn. Rows are sampled through the dataset
viewer API, so the 973 MB file is never downloaded.
"""

import ast
import re
import time
from pathlib import Path

import httpx

API = "https://datasets-server.huggingface.co/rows"
DATASET = "soniawmeyer/travel-conversations-finetuning"
TRAVEL_WORDS = re.compile(
    r"\b(travel|trip|pack|packing|luggage|flight|hotel|hostel|budget|itinerary|vacation|"
    r"holiday|tourist|sightseeing|backpack|solo|safe|safety|visa|jet lag|souvenir|"
    r"street food|local food|public transport|travel insurance)\b",
    re.I,
)


def is_generic_question(q: str) -> bool:
    """A short travel question that names no specific place, so it fits any trip."""
    if not (25 <= len(q) <= 160) or not q.endswith("?") or "\n" in q:
        return False
    if not TRAVEL_WORDS.search(q):
        return False
    # Capital letters after the first word mean a named place or brand: skip those.
    return not re.search(r"\s[A-Z][a-z]", q.replace(" I ", " i ").replace(" I'", " i'"))


def first_question(cell: str) -> str | None:
    try:
        turns = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return None
    return turns[0].strip() if turns and isinstance(turns[0], str) else None


def fetch_questions(out: Path, pages: int = 40, total_rows: int = 44734, log=print) -> int:
    questions: dict[str, None] = {}
    step = total_rows // pages
    with httpx.Client(timeout=30) as client:
        for page in range(pages):
            params = {
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "offset": page * step,
                "length": 100,
            }
            r = client.get(API, params=params)
            if r.status_code != 200:
                log(f"  ! page {page}: HTTP {r.status_code}")
                time.sleep(3)
                continue
            for row in r.json()["rows"]:
                q = first_question(row["row"]["data"])
                if q and is_generic_question(q):
                    questions[q] = None
            time.sleep(0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(questions) + "\n")
    log(f"kept {len(questions)} generic travel questions -> {out}")
    return len(questions)


def load_questions(path: Path) -> list[str]:
    return [q for q in path.read_text().splitlines() if q.strip()] if path.exists() else []
