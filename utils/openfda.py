"""
OpenFDA drug-label client for Step 3.

Given a FAERS active ingredient (prod_ai, e.g. "TACROLMEDROXYPROGESTERONE ACETATE"), fetch that drug's
structured product label from https://api.fda.gov/drug/label.json and return the
sections Step 3 reasons over: indications, adverse reactions, boxed/other
warnings, contraindications etc.

Labels are cached on disk (data/labels/<slug>.json) so re-runs doesn't re-hit the API.
A "not found" result is cached too, so it doesn't retry drugs OpenFDA simply has no SPL
for -- but ONLY when OpenFDA actually answered "no match" (HTTP 404). A request that
never got an answer (network down, rate limit, 5xx) raises OpenFDAError instead: it says
nothing about the drug, and caching it as a not-found would mark a labeled drug unlabeled
forever.
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import json
import re
import time
from html.parser import HTMLParser

import requests

from . import config

_ENDPOINT = "https://api.fda.gov/drug/label.json"

# Retry budget for a request that never gets an ANSWER (network down, 429, 5xx).
# A 404 is an answer ("no such label") and is never retried.
MAX_ATTEMPTS = int(config._env("OPENFDA_MAX_ATTEMPTS", "10") or "10")
_BACKOFF = 3   # seconds, multiplied by the attempt number -> 3, 6, 9, ... ~2 min total

# Label sections kept. 
# Each *_table field is the SPL's HTML incidence table for the section above it, flattened to text by _html_to_text()
# For initial testing, please keep or remove sections or cap the section lengths to a reasonable amount.
_SECTIONS = {
    "indications_and_usage": 50000,
    "boxed_warning": 50000,
    "adverse_reactions": 50000,
    "adverse_reactions_table": 50000,
    "warnings_and_cautions": 50000,
    "warnings_and_precautions": 50000,
    "warnings_and_cautions_table": 50000,
    "warnings": 50000,
    "warnings_table": 50000,
    "precautions": 50000,
    "stop_use": 50000,
    "contraindications": 50000,
}

# _SECTIONS = {
#     "indications_and_usage": 4000,
#     "boxed_warning": 6000,
#     "adverse_reactions": 20000,
#     "adverse_reactions_table": 10000,
#     "warnings_and_cautions": 16000,
#     "warnings_and_precautions": 16000,
#     "warnings_and_cautions_table": 4000,
#     "warnings": 16000,
#     "warnings_table": 4000,
#     "precautions": 12000,
#     "stop_use": 1000,
#     "contraindications": 3000,
# }

############
# 2. Helpers
############
class _TableText(HTMLParser):
    """Flatten an SPL HTML table to readable text: cells joined by ' | ', rows by
    newlines. Every tag attribute (styleCode, width, ID, ...) is dropped."""

    def __init__(self) -> None:
        super().__init__()  # convert_charrefs=True -> entities decoded in handle_data
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in ("br", "p"):
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _html_to_text(html: str) -> str:
    """Strip SPL table markup to plain text, keeping row/column structure so an
    adverse-event name stays next to its incidence numbers."""
    parser = _TableText()
    parser.feed(html)
    lines = (" ".join(ln.split()).strip(" |") for ln in "".join(parser.parts).splitlines())
    return "\n".join(ln for ln in lines if ln)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def _cache_path(name: str, brand: str | None = None):
    config.LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(name)
    if brand:
        slug = f"{slug}__{_slug(brand)}"
    return config.LABEL_CACHE_DIR / f"{slug}.json"


class OpenFDAError(RuntimeError):
    """OpenFDA never answered (network down, rate limit exhausted, 5xx).

    Distinct from "this drug has no label": a request that never got an answer says
    NOTHING about the drug, and must never be cached as a not-found -- that would
    permanently mark a labeled drug as unlabeled.
    """


def _get(params: dict) -> requests.Response | None:
    """GET the label endpoint, retrying transient failures.

    Returns the response, or None when OpenFDA answers 404 -- the one reply that
    genuinely means "the search matched nothing". Raises OpenFDAError once the retry
    budget is spent without ever getting an answer.

    Backoff is linear (3s, 6s, 9s, ...), so the default 10 attempts ride out roughly
    2 minutes of a flapping resolver or a rate-limit window.
    """
    if config.OPENFDA_API_KEY:
        params = {**params, "api_key": config.OPENFDA_API_KEY}

    last = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(_ENDPOINT, params=params, timeout=30)
        except requests.RequestException as err:
            last = str(err)
            wait = _BACKOFF * attempt
        else:
            if resp.status_code == 404:
                return None
            if resp.ok:
                return resp
            last = f"HTTP {resp.status_code} -- {resp.text[:200]}"
            # 429 = rate limited; the window is per-minute, so back off harder
            wait = (_BACKOFF * 2 if resp.status_code == 429 else _BACKOFF) * attempt
        if attempt < MAX_ATTEMPTS:               # don't sleep after the last try
            time.sleep(wait)
    raise OpenFDAError(f"[openfda] no answer after {MAX_ATTEMPTS} attempts -- {last}")


def _query(search: str) -> dict | None:
    # sort by effective_time desc so results[0] is the MOST RECENT label revision.
    resp = _get({"search": search, "limit": 1, "sort": "effective_time:desc"})
    if resp is None:
        return None
    results = resp.json().get("results", [])
    return results[0] if results else None


def _extract(result: dict) -> dict:
    sections = {}
    for field, budget in _SECTIONS.items():
        value = result.get(field)
        if not value:
            continue
        text = " ".join(value) if isinstance(value, list) else str(value)
        if field.endswith("_table") or "<table" in text[:200]:
            text = _html_to_text(text)
        text = text.strip()
        if text:
            # Say so when a section is cut: the dropped tail is risk text the model
            # never sees, and an AE described only there comes back a false "NOVEL".
            if len(text) > budget:
                print(f"[openfda] truncated {field}: {len(text):,} -> {budget:,} chars")
            sections[field] = text[:budget]
    openfda = result.get("openfda", {})
    return {
        "found": bool(sections),
        "brand_name": openfda.get("brand_name", []),
        "generic_name": openfda.get("generic_name", []),
        "manufacturer_name": openfda.get("manufacturer_name", []),
        "effective_time": result.get("effective_time", ""),   # SPL revision date (YYYYMMDD)
        "sections": sections,
    }


###############
# 3. Public API
###############
def fetch_label(
    active_ingredient: str,
    brand_name: str | None = None,
    *,
    refresh: bool = False,
) -> dict:
    """Return label sections for a drug, using the disk cache.

    Matching is STRICT (openFDA `.exact` only): the drug must match substance_name
    OR generic_name exactly. Pass `brand_name` to pin the comparison to ONE product
    -- it adds an exact brand_name match. If no exact label exists, this PRINTS a
    warning and returns {"found": False, ...} rather than falling back to a looser
    or different product. `brand_name` must be the exact registered string; use
    list_brands(active_ingredient) to discover it.
    """
    path = _cache_path(active_ingredient, brand_name)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    name = active_ingredient.strip().replace('"', "")
    brand_clause = ""
    if brand_name:
        brand = brand_name.strip().replace('"', "")
        brand_clause = f' AND openfda.brand_name.exact:"{brand}"'

    # Exact substance OR exact generic (both strict); + exact brand when given.
    result = (
        _query(f'openfda.substance_name.exact:"{name}"{brand_clause}')
        or _query(f'openfda.generic_name.exact:"{name}"{brand_clause}')
    )
    if result is None:
        print(f"[openfda] no exact label for {name!r}"
              + (f" brand={brand_name!r}" if brand_name else ""))
        label = {"found": False, "sections": {}, "reason": "no exact match",
                 "query": {"drug": name, "brand_name": brand_name}}
    else:
        label = _extract(result)
    path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    return label


def list_brands(active_ingredient: str, limit: int = 25) -> list[tuple[str, int]]:
    """List (brand_name, n_labels) for a drug, most labels first -- so you can see the
    EXACT brand strings to pass to fetch_label(..., brand_name=...).

    Searches substance_name then generic_name (same fields as fetch_label).
    """
    name = active_ingredient.strip().replace('"', "")
    for field in ("substance_name", "generic_name"):
        resp = _get({                      # raises OpenFDAError if the API never answers
            "search": f'openfda.{field}.exact:"{name}"',
            "count": "openfda.brand_name.exact",
            "limit": limit,
        })
        if resp is None:
            continue                       # this field matched nothing -- try the next
        results = resp.json().get("results", [])
        if results:
            return [(r["term"], r["count"]) for r in results]
    return []                              # genuinely no brands on either field


def check_brand(active_ingredient: str, brand_name: str) -> dict:
    """Verify `brand_name` is a REAL registered brand for a drug before relying on it.

    fetch_label matches brands with openFDA `.exact`, so a near-miss (e.g. "Provera"
    when the registered strings are "Depo-Provera" / "Depo-SubQ Provera") silently
    yields found=False -> status=no_label. This checks the string against the drug's
    actual brand list and, on a miss, suggests the closest registered ones.

    Returns {"exact": bool, "n_labels": int, "suggestions": [(brand, n_labels), ...]}.
    """
    brands = list_brands(active_ingredient, limit=100)
    counts = dict(brands)
    if brand_name in counts:
        return {"exact": True, "n_labels": counts[brand_name], "suggestions": []}
    q = brand_name.strip().lower()
    close = [(b, n) for b, n in brands if q in b.lower() or b.lower() in q]
    return {"exact": False, "n_labels": 0, "suggestions": close or brands[:10]}


def label_meta(label: dict) -> dict:
    """Provenance for a fetched label: representative brand / manufacturer / SPL
    revision date (effective_time). Used to stamp results with which label a
    novel-vs-labeled or ranking decision was made against."""
    def first(xs):
        return xs[0] if isinstance(xs, list) and xs else (xs if isinstance(xs, str) else "")
    return {
        "label_brand": first(label.get("brand_name", [])),
        "label_manufacturer": first(label.get("manufacturer_name", [])),
        "label_effective_time": label.get("effective_time", ""),
    }


def label_to_text(label: dict) -> str:
    """Render cached sections into a labelled block for the prompt."""
    parts = []
    for field, text in label.get("sections", {}).items():
        heading = field.replace("_", " ").upper()
        parts.append(f"## {heading}\n{text}")
    return "\n\n".join(parts)
