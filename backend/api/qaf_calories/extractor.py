from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FoodCandidate:
    item_id: str
    confidence_item: float
    uncertain: bool
    method: str  # exact|fuzzy|heuristic
    matched_alias: str | None = None
    reasons: list[str] | None = None


_STOPWORDS = {
    "con",
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "a",
    "al",
    "y",
    "en",
    "para",
    "por",
    "una",
    "un",
}


def _normalize_key(value: str) -> str:
    key = re.sub(r"\s+", " ", (value or "").strip().lower())
    key = re.sub(r"[^a-záéíóúñü0-9/ ]", "", key)
    return key.strip()


def _tokenize(value: str) -> list[str]:
    key = _normalize_key(value)
    return [t for t in key.split(" ") if t and t not in _STOPWORDS]


def _best_fuzzy_match(query: str, candidates: list[str]) -> tuple[str | None, float]:
    if not query or not candidates:
        return None, 0.0
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=0.0)
    if not matches:
        return None, 0.0
    best = matches[0]
    score = difflib.SequenceMatcher(a=query, b=best).ratio()
    return best, float(score)


def extract_foods(vision: dict[str, Any], *, aliases: dict[str, str], fuzzy_cutoff: float = 0.86) -> list[FoodCandidate]:
    raw_items = vision.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    out: list[FoodCandidate] = []

    def _add(item: FoodCandidate | None):
        if not item:
            return
        if all(x.item_id != item.item_id for x in out):
            out.append(item)

    for it in raw_items:
        raw = str(it)
        key = _normalize_key(raw)
        if not key:
            continue

        # exact
        if key in aliases:
            _add(
                FoodCandidate(
                    item_id=aliases[key],
                    confidence_item=1.0,
                    uncertain=False,
                    method="exact",
                    matched_alias=key,
                    reasons=[],
                )
            )
            continue

        # fuzzy
        best, score = _best_fuzzy_match(key, list(aliases.keys()))
        if best and score >= fuzzy_cutoff:
            conf = max(0.0, min(0.95, score))
            _add(
                FoodCandidate(
                    item_id=aliases[best],
                    confidence_item=conf,
                    uncertain=conf < 0.9,
                    method="fuzzy",
                    matched_alias=best,
                    reasons=["fuzzy_match"],
                )
            )

        # n-grams
        toks = _tokenize(key)
        if len(toks) >= 2:
            for n in (3, 2, 1):
                if len(toks) < n:
                    continue
                for i in range(0, len(toks) - n + 1):
                    phrase = " ".join(toks[i : i + n])
                    if phrase in aliases:
                        _add(
                            FoodCandidate(
                                item_id=aliases[phrase],
                                confidence_item=0.92,
                                uncertain=False,
                                method="exact",
                                matched_alias=phrase,
                                reasons=["ngram_exact"],
                            )
                        )

        # heuristics
        if any(t.startswith("arroz") for t in toks):
            _add(FoodCandidate(item_id="arroz_blanco_cocido", confidence_item=0.65, uncertain=True, method="heuristic", reasons=["token_arroz"]))
        if any(t.startswith("pollo") for t in toks):
            _add(FoodCandidate(item_id="pollo_pechuga_asada", confidence_item=0.65, uncertain=True, method="heuristic", reasons=["token_pollo"]))

    return out
