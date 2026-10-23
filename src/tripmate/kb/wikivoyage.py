"""Fetch Wikivoyage pages and turn them into structured places and prose guide chunks.

Wikivoyage marks every place with a listing template, e.g.
    {{see | name=Basilica of Bom Jesus | hours=9AM-6:30PM | price=Free | lat=.. | content=..}}
so one listing becomes one retrievable chunk with real metadata. Prose sections such as
"Stay safe" become guide chunks. Content is CC BY-SA; every chunk keeps its source URL.
"""

import html
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import mwparserfromhell
import yaml

API = "https://en.wikivoyage.org/w/api.php"
USER_AGENT = "TripMateAI/0.1 (https://github.com/Sanjay2N/TripMate-AI; personal learning project)"
REQUEST_GAP_S = 1.0  # stay well under Wikimedia rate limits

LISTING_KINDS = {"see", "do", "buy", "eat", "drink", "sleep"}
GUIDE_SECTIONS = {
    "understand",
    "get in",
    "get around",
    "stay safe",
    "stay healthy",
    "respect",
    "cope",
    "climate",
    "talk",
}
MAX_GUIDE_CHARS = 1200
MIN_GUIDE_CHARS = 200


@dataclass
class ParsedPage:
    title: str
    geo: tuple[float, float] | None = None
    listings: list[dict] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)  # (heading, plain text)


def fetch_wikitext(
    client: httpx.Client, page: str, cache_dir: Path | None = None
) -> tuple[str, str] | None:
    """Return (resolved title, wikitext), or None if the page does not exist.

    Raw responses are cached on disk so re-running ingestion does not hit the API again.
    """
    cache = cache_dir / f"{page.replace('/', '_')}.json" if cache_dir else None
    if cache and cache.exists():
        data = json.loads(cache.read_text())
    else:
        params = {
            "action": "parse",
            "page": page,
            "prop": "wikitext",
            "format": "json",
            "redirects": 1,
        }
        for attempt in range(5):
            r = client.get(API, params=params)
            if r.status_code != 429:
                break
            retry_after = r.headers.get("retry-after", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else 5 * (attempt + 1))
        r.raise_for_status()
        data = r.json()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        time.sleep(REQUEST_GAP_S)
    if "error" in data:
        return None
    return data["parse"]["title"], data["parse"]["wikitext"]["*"]


def clean(value) -> str:
    """Wikitext fragment -> plain text."""
    code = mwparserfromhell.parse(str(value))
    for link in code.filter_wikilinks():
        if str(link.title).strip().lower().startswith(("file:", "image:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    text = html.unescape(code.strip_code())
    return re.sub(r"[ \t\xa0]+", " ", text).strip()


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def parse_page(title: str, wikitext: str) -> ParsedPage:
    code = mwparserfromhell.parse(wikitext)
    page = ParsedPage(title=title)

    for tpl in code.filter_templates():
        name = str(tpl.name).strip().lower()
        if name == "geo" and page.geo is None and len(tpl.params) >= 2:
            lat, lon = (
                _to_float(str(tpl.params[0].value).strip()),
                _to_float(str(tpl.params[1].value).strip()),
            )
            if lat is not None and lon is not None:
                page.geo = (lat, lon)
            continue

        def param(key: str) -> str:
            return clean(tpl.get(key).value) if tpl.has(key) else ""

        kind = param("type").lower() if name == "listing" else name
        if kind not in LISTING_KINDS or not param("name"):
            continue
        page.listings.append(
            {
                "kind": kind,
                "name": param("name"),
                "alt": param("alt"),
                "address": param("address"),
                "directions": param("directions"),
                "hours": param("hours"),
                "price": param("price"),
                "lat": _to_float(param("lat")),
                "long": _to_float(param("long")),
                "content": param("content"),
                "lastedit": param("lastedit"),
            }
        )

    for section in code.get_sections(levels=[2]):
        headings = section.filter_headings()
        if not headings:
            continue
        heading = clean(headings[0].title)
        if heading.lower() in GUIDE_SECTIONS:
            body = clean(str(section).split("\n", 1)[1] if "\n" in str(section) else "")
            if body:
                page.sections.append((heading, body))
    return page


def chunk_text(text: str, max_chars: int = MAX_GUIDE_CHARS) -> list[str]:
    """Pack whole paragraphs into chunks of at most max_chars (long paragraphs are split)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars) + 1 or max_chars
            paragraphs_head, p = p[:cut], p[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraphs_head)
        if current and len(current) + len(p) + 1 > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def _id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts)))


def _url(title: str) -> str:
    return "https://en.wikivoyage.org/wiki/" + title.replace(" ", "_")


def poi_text(p: dict) -> str:
    """The string that gets embedded and BM25-indexed for a place."""
    head = f"{p['name']} ({p['kind']}) in {p['area']}, {p['destination']}."
    extras = " ".join(x for x in (p["alt"], p["content"], p["directions"]) if x)
    return f"{head} {extras}".strip()


def ingest(destinations_file: Path, out_dir: Path, log=print) -> dict:
    cfg = yaml.safe_load(destinations_file.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir.parent / "raw"
    pois, guides, places = [], [], {}

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        for dest in cfg["destinations"]:
            seen_names: set[str] = set()
            n_pois = n_guides = 0
            for page_name in dest["pages"]:
                fetched = fetch_wikitext(client, page_name, raw_dir)
                if fetched is None:
                    log(f"  ! page not found: {page_name}")
                    continue
                title, wikitext = fetched
                page = parse_page(title, wikitext)
                if page.geo and dest["name"] not in places:
                    places[dest["name"]] = {"lat": page.geo[0], "long": page.geo[1]}

                for listing in page.listings:
                    key = listing["name"].lower()
                    if key in seen_names:
                        continue
                    seen_names.add(key)
                    poi = {
                        "id": _id(dest["name"], title, listing["name"]),
                        "doc_type": "poi",
                        "destination": dest["name"],
                        "area": title,
                        "url": _url(title),
                        **listing,
                    }
                    poi["text"] = poi_text(poi)
                    pois.append(poi)
                    n_pois += 1

                for heading, body in page.sections:
                    for i, chunk in enumerate(chunk_text(body)):
                        if len(chunk) < MIN_GUIDE_CHARS:
                            continue
                        guides.append(
                            {
                                "id": _id(dest["name"], title, heading, str(i)),
                                "doc_type": "guide",
                                "destination": dest["name"],
                                "area": title,
                                "section": heading,
                                "url": _url(title),
                                "text": f"{title} - {heading}: {chunk}",
                            }
                        )
                        n_guides += 1
            log(f"{dest['name']:<12} places={n_pois:<4} guide_chunks={n_guides}")

        for city in cfg.get("sources", []):
            fetched = fetch_wikitext(client, city, raw_dir)
            geo = parse_page(*fetched).geo if fetched else None
            if geo:
                places[city] = {"lat": geo[0], "long": geo[1]}
            else:
                log(f"  ! no coordinates for source city: {city}")

    for name, rows in (("pois", pois), ("guides", guides)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "places.json").write_text(json.dumps(places, indent=2, ensure_ascii=False))
    return {"pois": len(pois), "guides": len(guides), "places": len(places)}
