from __future__ import annotations

import csv
import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LOCALE = "es-CO"


@dataclass(frozen=True)
class CalorieItem:
    item_id: str
    kcal_per_100g: float | None
    kcal_per_unit: float | None
    unit_grams: float | None
    default_serving_grams: float | None


@dataclass(frozen=True)
class NormalizedItem:
    item_id: str
    confidence: float
    method: str  # exact|fuzzy|heuristic
    matched_alias: str | None = None


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


def load_aliases(csv_path: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not csv_path.exists():
        return aliases
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = _normalize_key(row.get("alias") or "")
            item_id = (row.get("item_id") or "").strip()
            if alias and item_id:
                aliases[alias] = item_id
    return aliases


def load_items_meta(csv_path: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    if not csv_path.exists():
        return meta
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue
            meta[item_id] = {
                "display_name_es": (row.get("display_name_es") or "").strip() or item_id,
                "display_name_en": (row.get("display_name_en") or "").strip() or item_id,
            }
    return meta


def load_calorie_db(csv_path: Path) -> dict[str, CalorieItem]:
    items: dict[str, CalorieItem] = {}
    if not csv_path.exists():
        return items

    def _to_float(value: str | None) -> float | None:
        raw = (value or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue
            items[item_id] = CalorieItem(
                item_id=item_id,
                kcal_per_100g=_to_float(row.get("kcal_per_100g")),
                kcal_per_unit=_to_float(row.get("kcal_per_unit")),
                unit_grams=_to_float(row.get("unit_grams")),
                default_serving_grams=_to_float(row.get("default_serving_grams")),
            )
    return items


def display_name(item_id: str, items_meta: dict[str, dict[str, str]] | None, locale: str) -> str:
    meta = (items_meta or {}).get(item_id) or {}
    if str(locale or "").lower().startswith("en"):
        return meta.get("display_name_en") or item_id
    return meta.get("display_name_es") or item_id


def normalize_item(raw: str, *, aliases: dict[str, str], fuzzy_cutoff: float = 0.86) -> NormalizedItem | None:
    key = _normalize_key(raw)
    if not key:
        return None

    if key in aliases:
        return NormalizedItem(item_id=aliases[key], confidence=1.0, method="exact", matched_alias=key)

    best, score = _best_fuzzy_match(key, list(aliases.keys()))
    if best and score >= fuzzy_cutoff:
        conf = max(0.0, min(0.95, score))
        return NormalizedItem(item_id=aliases[best], confidence=conf, method="fuzzy", matched_alias=best)

    tokens = _tokenize(key)
    if any("arroz" == t or t.startswith("arroz") for t in tokens):
        return NormalizedItem(item_id="arroz_blanco_cocido", confidence=0.65, method="heuristic")
    if any("pollo" == t or t.startswith("pollo") for t in tokens):
        return NormalizedItem(item_id="pollo_pechuga_asada", confidence=0.65, method="heuristic")

    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _portion_confidence(method: str) -> float:
    return {
        "explicit_grams": 1.0,
        "explicit_units": 0.8,
        "explicit_ml": 0.9,
        "cup": 0.7,
        "slice": 0.75,
        "qualitative": 0.55,
        "default": 0.5,
        "unknown": 0.2,
    }.get(method, 0.2)


def parse_portion_grams(item_id: str, portion_text: str | None, calorie_db: dict[str, CalorieItem]) -> tuple[float | None, str]:
    text = (portion_text or "").strip().lower()
    if not text:
        return None, "unknown"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*g\b", text)
    if m:
        grams = _safe_float(m.group(1).replace(",", "."))
        if grams and grams > 0:
            return float(grams), "explicit_grams"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*ml\b", text)
    if m:
        ml = _safe_float(m.group(1).replace(",", "."))
        if ml and ml > 0:
            return float(ml), "explicit_ml"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(unidad|unidades)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if not n or n <= 0:
            return None, "unknown"
        ci = calorie_db.get(item_id)
        if ci and ci.unit_grams:
            return float(n) * float(ci.unit_grams), "explicit_units"
        if item_id == "huevo":
            return float(n) * 50.0, "explicit_units"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(rebanada|rebanadas)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if n and n > 0:
            ci = calorie_db.get(item_id)
            if ci and ci.unit_grams:
                return float(n) * float(ci.unit_grams), "slice"

    if "taza" in text:
        if item_id == "arroz_blanco_cocido":
            return 160.0, "cup"
        return 150.0, "cup"

    return None, "unknown"


def build_portion_candidates(
    item_id: str,
    *,
    calorie_db: dict[str, CalorieItem],
    locale: str = DEFAULT_LOCALE,
    memory_hint_grams: float | None = None,
    center_grams: float | None = None,
    center_source: str | None = None,
) -> list[dict[str, Any]]:
    ci = calorie_db.get(item_id)

    def _mk(label: str, grams: float, conf: float) -> dict[str, Any]:
        return {"label": label, "grams": round(float(grams), 2), "confidence_hint": round(float(conf), 4)}

    if center_grams and center_grams > 0 and (center_source in {"explicit", "inferred"}):
        g = float(center_grams)
        return [_mk("Menos", g * 0.9, 0.85), _mk("Estimado", g, 0.9), _mk("Más", g * 1.1, 0.85)]

    if memory_hint_grams and memory_hint_grams > 0:
        base = float(memory_hint_grams)
        return [_mk("Repetir última porción", base, 0.9), _mk("Menos", base * 0.9, 0.8), _mk("Más", base * 1.1, 0.8)]

    if ci and ci.unit_grams:
        ug = float(ci.unit_grams)
        return [_mk("1 unidad", 1.0 * ug, 0.85), _mk("2 unidades", 2.0 * ug, 0.8), _mk("3 unidades", 3.0 * ug, 0.75)]

    if ci and ci.default_serving_grams:
        c = float(ci.default_serving_grams)
        return [_mk("Porción pequeña", max(10.0, c * 0.75), 0.65), _mk("Porción típica", c, 0.75), _mk("Porción grande", c * 1.25, 0.65)]

    # Heurísticas MVP
    if item_id == "arroz_blanco_cocido":
        return [_mk("1/2 taza", 80.0, 0.7), _mk("1 taza", 160.0, 0.75), _mk("1.5 tazas", 240.0, 0.7)]
    if item_id == "pollo_pechuga_asada":
        return [_mk("100 g", 100.0, 0.7), _mk("150 g", 150.0, 0.75), _mk("200 g", 200.0, 0.7)]

    return [_mk("Porción pequeña", 100.0, 0.5), _mk("Porción media", 200.0, 0.55), _mk("Porción grande", 300.0, 0.5)]


def shannon_entropy(probs: list[float]) -> float:
    p = [x for x in probs if x and x > 0]
    if not p:
        return 0.0
    s = sum(p)
    p = [x / s for x in p]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def qaf_estimate(
    vision: dict[str, Any],
    *,
    calorie_db: dict[str, CalorieItem],
    aliases: dict[str, str],
    items_meta: dict[str, dict[str, str]] | None = None,
    locale: str = DEFAULT_LOCALE,
    memory_hint_by_item: dict[str, float] | None = None,
    goal_kcal_meal: float | None = None,
    confirmed_portions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    memory_hint_by_item = memory_hint_by_item or {}

    raw_items = vision.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    normalized_items: list[NormalizedItem] = []

    def _add(n: NormalizedItem | None):
        if not n:
            return
        if all(x.item_id != n.item_id for x in normalized_items):
            normalized_items.append(n)

    for it in raw_items:
        raw = str(it)
        _add(normalize_item(raw, aliases=aliases))
        toks = _tokenize(raw)
        if len(toks) >= 2:
            for n in (3, 2, 1):
                if len(toks) < n:
                    continue
                for i in range(0, len(toks) - n + 1):
                    _add(normalize_item(" ".join(toks[i : i + n]), aliases=aliases))

    normalized_ids = [x.item_id for x in normalized_items]

    portion_text = str(vision.get("portion_estimate") or "").strip() or None

    confirmed_map: dict[str, float] = {}
    if confirmed_portions:
        for cp in confirmed_portions:
            if not isinstance(cp, dict):
                continue
            iid = str(cp.get("item_id") or "").strip()
            try:
                g = float(cp.get("grams"))
            except Exception:
                continue
            if iid and g > 0:
                confirmed_map[iid] = g

    items_out = []
    total_kcal_raw = 0.0
    missing: list[str] = []
    reasons: list[str] = []

    per_item_conf: dict[str, dict[str, Any]] = {}

    for nitem in normalized_items:
        item_id = nitem.item_id
        ci = calorie_db.get(item_id)

        grams, portion_method = parse_portion_grams(item_id, portion_text, calorie_db)
        portion_match = None

        selected_portion = None
        if item_id in confirmed_map:
            grams = float(confirmed_map[item_id])
            portion_method = "explicit_grams"
            portion_match = "confirmed"
            selected_portion = {"grams": round(float(grams), 2), "source": "user"}

        used_default = False
        if grams is None and ci and ci.default_serving_grams:
            grams = float(ci.default_serving_grams)
            portion_method = "default"
            used_default = True

        kcal = None
        if ci:
            if grams is not None and ci.kcal_per_100g is not None:
                kcal = (float(grams) / 100.0) * float(ci.kcal_per_100g)
            elif ci.kcal_per_unit is not None:
                if grams is not None and ci.unit_grams:
                    kcal = (float(grams) / float(ci.unit_grams)) * float(ci.kcal_per_unit)
                else:
                    kcal = float(ci.kcal_per_unit)
                    grams = grams or ci.unit_grams

        if kcal is None:
            missing.append(item_id)
            reasons.append(f"missing_item_in_db:{item_id}")
            continue

        total_kcal_raw += float(kcal)

        conf_item = float(nitem.confidence)
        conf_portion = _portion_confidence(portion_method)
        conf_combined = max(0.0, min(1.0, 0.65 * conf_item + 0.35 * conf_portion))

        if nitem.method != "exact":
            reasons.append(f"item_normalized_{nitem.method}:{item_id}")
        if used_default:
            reasons.append(f"used_default_serving:{item_id}")

        per_item_conf[item_id] = {
            "confidence_item": round(conf_item, 4),
            "confidence_portion": round(conf_portion, 4),
            "confidence": round(conf_combined, 4),
            "item_method": nitem.method,
            "portion_method": portion_method,
            "matched_alias": nitem.matched_alias,
            "portion_match": portion_match,
        }

        center = None
        center_source = None
        if grams is not None and portion_method in {"explicit_grams", "explicit_ml", "explicit_units"}:
            center = float(grams)
            center_source = "explicit"
        elif grams is not None and portion_method in {"cup", "slice", "qualitative"}:
            center = float(grams)
            center_source = "inferred"

        candidates = build_portion_candidates(
            item_id,
            calorie_db=calorie_db,
            locale=locale,
            memory_hint_grams=memory_hint_by_item.get(item_id),
            center_grams=center,
            center_source=center_source,
        )

        items_out.append(
            {
                "item_id": item_id,
                "display_name": display_name(item_id, items_meta, locale),
                "grams": grams,
                "calories": round(float(kcal), 2),
                "portion_candidates": candidates,
                "selected_portion": selected_portion,
                "item_confidence": round(conf_item, 4),
                "portion_confidence": round(conf_portion, 4),
            }
        )

    probs = [1.0 for _ in normalized_ids]
    ent = shannon_entropy(probs)

    total_conf = 0.0
    if items_out and total_kcal_raw > 0:
        for it in items_out:
            iid = str(it.get("item_id") or "")
            kcal = float(it.get("calories") or 0.0)
            w = kcal / total_kcal_raw
            c = float((per_item_conf.get(iid) or {}).get("confidence") or 0.0)
            total_conf += w * c

    uncertainty_score = max(0.0, min(1.0, 1.0 - total_conf))

    # Rango por candidates
    low = 0.0
    high = 0.0
    best_estimate = 0.0
    per_item_spread: dict[str, float] = {}

    for it in items_out:
        iid = str(it.get("item_id") or "")
        ci = calorie_db.get(iid)
        cands = it.get("portion_candidates") or []
        if not ci or not cands:
            kcal = float(it.get("calories") or 0.0)
            low += kcal
            high += kcal
            best_estimate += kcal
            continue

        def _kcal_for_grams(g: float) -> float:
            if ci.kcal_per_100g is not None:
                return (float(g) / 100.0) * float(ci.kcal_per_100g)
            if ci.kcal_per_unit is not None and ci.unit_grams:
                return (float(g) / float(ci.unit_grams)) * float(ci.kcal_per_unit)
            return float(it.get("calories") or 0.0)

        grams_kcal = []
        for c in cands:
            try:
                g = float(c.get("grams") or 0.0)
            except Exception:
                continue
            grams_kcal.append((g, _kcal_for_grams(g)))
        grams_kcal = [(g, k) for (g, k) in grams_kcal if k >= 0]
        grams_kcal.sort(key=lambda x: x[0])
        kcal_opts = [k for (_g, k) in grams_kcal]
        if not kcal_opts:
            kcal = float(it.get("calories") or 0.0)
            low += kcal
            high += kcal
            best_estimate += kcal
            continue

        item_low = min(kcal_opts)
        item_high = max(kcal_opts)
        item_mid = kcal_opts[len(kcal_opts) // 2]

        low += item_low
        high += item_high

        pm = str((per_item_conf.get(iid) or {}).get("portion_method") or "")
        current = float(it.get("calories") or 0.0)
        best_estimate += float(item_mid if pm in {"default", "unknown"} else (current or item_mid))

        per_item_spread[iid] = float(item_high - item_low)

    range_driver = None
    if per_item_spread:
        range_driver = sorted(per_item_spread.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    range_width = max(0.0, high - low)
    rel_width = (range_width / best_estimate) if best_estimate else 1.0

    crosses_goal = False
    if goal_kcal_meal is not None:
        try:
            gk = float(goal_kcal_meal)
            crosses_goal = (low <= gk <= high)
        except Exception:
            crosses_goal = False

    needs_confirmation = bool(missing) or (rel_width >= 0.35) or (range_width >= 180.0) or crosses_goal

    follow_up_questions: list[dict[str, Any]] = []
    if needs_confirmation and range_driver:
        driver_item = next((x for x in items_out if x.get("item_id") == range_driver), None)
        if driver_item:
            prompt = f"¿Cuánta porción de {display_name(range_driver, items_meta, locale)} fue?"
            follow_up_questions.append(
                {
                    "type": "confirm_portion",
                    "item_id": range_driver,
                    "prompt": prompt,
                    "options": driver_item.get("portion_candidates") or [],
                }
            )

    defaults_used = sum(1 for _iid, info in per_item_conf.items() if str(info.get("portion_method")) == "default")
    D = (defaults_used / len(per_item_conf)) if per_item_conf else 0.0
    M = 1.0 if missing else 0.0
    U = float(uncertainty_score)
    A = 1.0 if follow_up_questions else 0.0

    w_uncertainty, w_default, w_missing, w_ask = 1.0, 0.4, 1.2, 0.3
    decision_score = (w_uncertainty * U) + (w_default * D) + (w_missing * M) + (w_ask * A)

    if missing:
        decision = "needs_confirmation"
        decision_reason = "missing_items_in_db"
    elif needs_confirmation and len(items_out) > 1:
        decision = "partial"
        decision_reason = "confirm_range_driver"
    elif needs_confirmation:
        decision = "needs_confirmation"
        decision_reason = "high_uncertainty"
    else:
        decision = "accepted"
        decision_reason = "low_uncertainty"

    explainability = [
        f"Estimado: {round(best_estimate, 0):.0f} kcal (rango {round(low, 0):.0f}–{round(high, 0):.0f})",
    ]
    if range_driver:
        explainability.append(f"Lo que más mueve el rango: {display_name(range_driver, items_meta, locale)}")
    explainability = explainability[:2]

    return {
        "items": items_out,
        "missing_items": missing,
        "total_calories": round(float(best_estimate), 2),
        "total_calories_range": {"low": round(float(low), 2), "high": round(float(high), 2)},
        "uncertainty": {
            "entropy": round(float(ent), 4),
            "uncertainty_score": round(float(uncertainty_score), 4),
            "portion_text": portion_text,
        },
        "needs_confirmation": bool(needs_confirmation),
        "confidence": {"total": round(float(total_conf), 4), "per_item": per_item_conf},
        "reasons": sorted(set([r for r in reasons if r])),
        "follow_up_questions": follow_up_questions,
        "suggested_questions": follow_up_questions,
        "range_driver": range_driver,
        "explainability": explainability,
        "decision": decision,
        "decision_reason": decision_reason,
        "decision_score": round(float(decision_score), 4),
        "goal_kcal_meal": goal_kcal_meal,
    }


def render_professional_summary(qaf: dict[str, Any]) -> str:
    """Texto corto y profesional para incrustar en el chat como fallback."""
    if not isinstance(qaf, dict):
        return ""
    try:
        from .response_builder import build_professional_chat_text

        return build_professional_chat_text(qaf)
    except Exception:
        # Fallback ultra-defensivo (evita romper el chat por import errors)
        total = qaf.get("total_calories")
        rng = qaf.get("total_calories_range") or {}
        low = rng.get("low")
        high = rng.get("high")
        driver = qaf.get("range_driver")

        lines = []
        if total is not None and low is not None and high is not None:
            lines.append(
                f"Estimado: {round(float(total), 0):.0f} kcal (rango {round(float(low), 0):.0f}–{round(float(high), 0):.0f})"
            )
        if driver:
            lines.append(f"Lo que más mueve el rango: {driver}")

        if qaf.get("needs_confirmation") and (qaf.get("follow_up_questions") or []):
            q = (qaf.get("follow_up_questions") or [])[0]
            prompt = str(q.get("prompt") or "").strip()
            if prompt:
                lines.append(prompt)
        return "\n".join(lines).strip()


def load_nutrition_db(csv_path: Path) -> dict[str, Any]:
    """Loader v2 (re-export) para nutrition_db.csv."""
    from .catalogs import load_nutrition_db as _load

    return _load(csv_path)


def load_micros_db(csv_path: Path) -> dict[str, Any]:
    """Loader v2 (re-export) para micros_db.csv."""
    from .catalogs import load_micros_db as _load

    return _load(csv_path)


def qaf_estimate_v2(
    vision: dict[str, Any],
    *,
    calorie_db: dict[str, CalorieItem],
    aliases: dict[str, str],
    items_meta: dict[str, dict[str, str]] | None = None,
    locale: str = DEFAULT_LOCALE,
    memory_hint_by_item: dict[str, float] | None = None,
    goal_kcal_meal: float | None = None,
    confirmed_portions: list[dict[str, Any]] | None = None,
    nutrition_db: dict[str, Any] | None = None,
    micros_db: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Pipeline v2: extractor -> portion_engine -> nutrition_engine -> governor -> response_builder."""

    from .extractor import extract_foods
    from .governor import decide
    from .nutrition_engine import macro_calorie_percentages, macros_from_grams, micros_highlights
    from .portion_engine import build_portion_candidates, parse_portion, portion_confidence
    from .response_builder import build_ui_blocks

    memory_hint_by_item = memory_hint_by_item or {}
    nutrition_db = nutrition_db or {}
    micros_db = micros_db or {}

    # Confirmaciones desde el frontend
    confirmed_map: dict[str, float] = {}
    if confirmed_portions:
        for cp in confirmed_portions:
            if not isinstance(cp, dict):
                continue
            iid = str(cp.get("item_id") or "").strip()
            try:
                g = float(cp.get("grams"))
            except Exception:
                continue
            if iid and g > 0:
                confirmed_map[iid] = g

    portion_text = str(vision.get("portion_estimate") or "").strip() or None
    scale_hints = vision.get("scale_hints") if isinstance(vision.get("scale_hints"), dict) else None

    foods = extract_foods(vision, aliases=aliases)

    items_out: list[dict[str, Any]] = []
    missing_items: list[str] = []
    missing_nutrition: list[str] = []
    reasons: list[str] = []

    per_item_conf: dict[str, dict[str, Any]] = {}
    per_item_spread: dict[str, float] = {}

    total_kcal_current = 0.0

    # Totales macros/micros
    macros_total = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    micros_pool: dict[str, str] = {}  # micro -> best level
    level_rank = {"alto": 3, "medio": 2, "bajo": 1}

    def _kcal_from_sources(item_id: str, grams: float | None) -> float | None:
        if grams is None or grams <= 0:
            return None
        n = nutrition_db.get(item_id)
        if n is not None:
            try:
                if getattr(n, "kcal_per_100g", None) is not None:
                    return (float(grams) / 100.0) * float(getattr(n, "kcal_per_100g"))
            except Exception:
                pass
        ci = calorie_db.get(item_id)
        if not ci:
            return None
        if ci.kcal_per_100g is not None:
            return (float(grams) / 100.0) * float(ci.kcal_per_100g)
        if ci.kcal_per_unit is not None:
            if ci.unit_grams and ci.unit_grams > 0:
                return (float(grams) / float(ci.unit_grams)) * float(ci.kcal_per_unit)
            return float(ci.kcal_per_unit)
        return None

    # Selección del range_driver: preferimos el ítem con mayor spread NO confirmado (si existe)
    spread_candidates: list[tuple[str, float, bool]] = []  # (item_id, spread, is_confirmed)

    for f in foods:
        item_id = f.item_id
        ci = calorie_db.get(item_id)

        grams, portion_method, portion_match = parse_portion(item_id, portion_text, calorie_db)
        selected_portion = None

        if item_id in confirmed_map:
            grams = float(confirmed_map[item_id])
            portion_method = "explicit_grams"
            portion_match = "confirmed"
            selected_portion = {"grams": round(float(grams), 2), "source": "user"}

        if grams is None and item_id in memory_hint_by_item and float(memory_hint_by_item[item_id] or 0.0) > 0:
            grams = float(memory_hint_by_item[item_id])
            portion_method = "memory"
            portion_match = "memory"

        used_default = False
        if grams is None and ci and ci.default_serving_grams:
            grams = float(ci.default_serving_grams)
            portion_method = "default"
            portion_match = portion_match or "default"
            used_default = True

        kcal = _kcal_from_sources(item_id, grams)
        if kcal is None:
            missing_items.append(item_id)
            reasons.append(f"missing_item_in_db:{item_id}")
            continue

        total_kcal_current += float(kcal)

        conf_item = float(f.confidence_item)
        conf_portion = float(portion_confidence(portion_method))
        conf_combined = max(0.0, min(1.0, 0.65 * conf_item + 0.35 * conf_portion))

        if f.method != "exact":
            reasons.append(f"item_normalized_{f.method}:{item_id}")
        if used_default:
            reasons.append(f"used_default_serving:{item_id}")
        if f.uncertain:
            reasons.append(f"item_uncertain:{item_id}")

        per_item_conf[item_id] = {
            "confidence_item": round(conf_item, 4),
            "confidence_portion": round(conf_portion, 4),
            "confidence": round(conf_combined, 4),
            "item_method": f.method,
            "portion_method": portion_method,
            "matched_alias": f.matched_alias,
            "portion_match": portion_match,
        }

        center = None
        center_source = None
        if grams is not None and portion_method in {"explicit_grams", "explicit_ml", "explicit_units"}:
            center = float(grams)
            center_source = "explicit"
        elif grams is not None and portion_method in {"cup", "slice", "qualitative"}:
            center = float(grams)
            center_source = "inferred"

        candidates = build_portion_candidates(
            item_id,
            calorie_db=calorie_db,
            locale=locale,
            memory_hint_grams=memory_hint_by_item.get(item_id),
            center_grams=center,
            center_source=center_source,
            scale_hints=scale_hints,
        )

        # Macros
        n = nutrition_db.get(item_id)
        item_macros = None
        if n is not None and grams is not None:
            try:
                item_macros = macros_from_grams(n, float(grams))
                for k in macros_total.keys():
                    macros_total[k] += float(item_macros.get(k) or 0.0)
            except Exception:
                item_macros = None

        if n is None:
            missing_nutrition.append(item_id)

        # Micros
        m_list = micros_db.get(item_id) or []
        for m in m_list:
            micro = str(m.get("micro") or "").strip()
            level = str(m.get("level") or "").strip().lower()
            if not micro or not level:
                continue
            prev = micros_pool.get(micro)
            if not prev or level_rank.get(level, 0) > level_rank.get(prev, 0):
                micros_pool[micro] = level

        items_out.append(
            {
                "item_id": item_id,
                "display_name": display_name(item_id, items_meta, locale),
                "grams": grams,
                "calories": round(float(kcal), 2),
                "portion_candidates": candidates,
                "selected_portion": selected_portion,
                "item_confidence": round(conf_item, 4),
                "portion_confidence": round(conf_portion, 4),
                "macros": item_macros,
                "micros": m_list,
            }
        )

        # Spread por ítem (kcal) basado en candidates
        if candidates:
            kcal_opts = []
            for c in candidates:
                try:
                    g = float(c.get("grams") or 0.0)
                except Exception:
                    continue
                k = _kcal_from_sources(item_id, g)
                if k is not None and k >= 0:
                    kcal_opts.append(float(k))
            if kcal_opts:
                spread = max(kcal_opts) - min(kcal_opts)
                per_item_spread[item_id] = float(spread)
                spread_candidates.append((item_id, float(spread), item_id in confirmed_map))

    # Totales y rango
    low = 0.0
    high = 0.0
    best_estimate = 0.0

    for it in items_out:
        iid = str(it.get("item_id") or "")
        cands = it.get("portion_candidates") or []
        current_kcal = float(it.get("calories") or 0.0)

        if not cands:
            low += current_kcal
            high += current_kcal
            best_estimate += current_kcal
            continue

        kcal_opts = []
        for c in cands:
            try:
                g = float(c.get("grams") or 0.0)
            except Exception:
                continue
            k = _kcal_from_sources(iid, g)
            if k is not None and k >= 0:
                kcal_opts.append(float(k))
        if not kcal_opts:
            low += current_kcal
            high += current_kcal
            best_estimate += current_kcal
            continue

        item_low = min(kcal_opts)
        item_high = max(kcal_opts)
        item_mid = sorted(kcal_opts)[len(kcal_opts) // 2]
        low += item_low
        high += item_high

        pm = str((per_item_conf.get(iid) or {}).get("portion_method") or "")
        best_estimate += float(item_mid if pm in {"default", "unknown"} else (current_kcal or item_mid))

    # range_driver (mejor candidato no confirmado; si no hay, el de mayor spread)
    range_driver = None
    if spread_candidates:
        spread_candidates.sort(key=lambda x: x[1], reverse=True)
        best_unconfirmed = next((iid for (iid, _sp, is_conf) in spread_candidates if not is_conf), None)
        range_driver = best_unconfirmed or spread_candidates[0][0]

    # Confianza total ponderada por kcal
    total_conf = 0.0
    if items_out and best_estimate > 0:
        for it in items_out:
            iid = str(it.get("item_id") or "")
            kcal = float(it.get("calories") or 0.0)
            w = (kcal / best_estimate) if best_estimate else 0.0
            c = float((per_item_conf.get(iid) or {}).get("confidence") or 0.0)
            total_conf += w * c

    uncertainty_score = max(0.0, min(1.0, 1.0 - float(total_conf)))

    gov = decide(
        missing_items=missing_items,
        low=float(low),
        high=float(high),
        estimate=float(best_estimate),
        range_driver=range_driver,
        items_count=len(items_out),
        uncertainty_score=float(uncertainty_score),
        goal_kcal_meal=goal_kcal_meal,
    )

    follow_up_questions: list[dict[str, Any]] = []
    if gov.get("needs_confirmation") and range_driver:
        driver_item = next((x for x in items_out if x.get("item_id") == range_driver), None)
        if driver_item:
            prompt = f"¿Cuánta porción de {display_name(range_driver, items_meta, locale)} fue?"
            follow_up_questions.append(
                {
                    "type": "confirm_portion",
                    "item_id": range_driver,
                    "prompt": prompt,
                    "options": driver_item.get("portion_candidates") or [],
                }
            )

    defaults_used = sum(1 for _iid, info in per_item_conf.items() if str(info.get("portion_method")) == "default")
    D = (defaults_used / len(per_item_conf)) if per_item_conf else 0.0
    M = 1.0 if missing_items else 0.0
    U = float(uncertainty_score)
    A = 1.0 if follow_up_questions else 0.0
    w_uncertainty, w_default, w_missing, w_ask = 1.0, 0.4, 1.2, 0.3
    decision_score = (w_uncertainty * U) + (w_default * D) + (w_missing * M) + (w_ask * A)

    # Micros highlights (orden: alto -> medio -> bajo)
    micros_sorted = sorted(micros_pool.items(), key=lambda kv: level_rank.get(kv[1], 0), reverse=True)
    micros_out = [f"{micro} ({level})" for (micro, level) in micros_sorted if micro and level]

    macros_pct = macro_calorie_percentages(macros_total)

    explainability = [
        f"Estimado: {round(float(best_estimate), 0):.0f} kcal (rango {round(float(low), 0):.0f}–{round(float(high), 0):.0f})",
    ]
    if range_driver:
        explainability.append(f"Lo que más mueve el rango: {display_name(range_driver, items_meta, locale)}")
    explainability = explainability[:2]

    out = {
        "items": items_out,
        "missing_items": missing_items,
        "missing_nutrition_items": sorted(set(missing_nutrition)),
        "total_calories": round(float(best_estimate), 2),
        "total_calories_range": {"low": round(float(low), 2), "high": round(float(high), 2)},
        "uncertainty": {
            "entropy": 0.0,
            "uncertainty_score": round(float(uncertainty_score), 4),
            "portion_text": portion_text,
        },
        "needs_confirmation": bool(gov.get("needs_confirmation")),
        "confidence": {"total": round(float(total_conf), 4), "per_item": per_item_conf},
        "reasons": sorted(set([r for r in reasons if r])),
        "follow_up_questions": follow_up_questions,
        "suggested_questions": follow_up_questions,
        "range_driver": range_driver,
        "range_driver_display": display_name(range_driver, items_meta, locale) if range_driver else None,
        "explainability": explainability,
        "decision": str(gov.get("decision") or ""),
        "decision_reason": str(gov.get("decision_reason") or ""),
        "decision_score": round(float(decision_score), 4),
        "goal_kcal_meal": goal_kcal_meal,
        "macros_total": {k: round(float(v), 2) for k, v in macros_total.items()},
        "macros_pct": {k: round(float(v), 2) for k, v in macros_pct.items()},
        "micros_highlights": micros_out[:8],
    }

    out["ui_blocks"] = build_ui_blocks(out)
    return out
