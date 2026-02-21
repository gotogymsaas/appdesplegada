from __future__ import annotations

import csv
import difflib
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
class NutritionItem:
    item_id: str
    kcal_per_100g: float | None
    protein_g_per_100g: float | None
    carbs_g_per_100g: float | None
    fat_g_per_100g: float | None
    fiber_g_per_100g: float | None


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


def base_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def paths() -> dict[str, Path]:
    base = base_data_dir()
    return {
        "base": base,
        "aliases": base / "aliases.csv",
        "items_meta": base / "items_meta.csv",
        "calorie_db": base / "calorie_db.csv",
        "nutrition_db": base / "nutrition_db.csv",
        "micros_db": base / "micros_db.csv",
    }


def _to_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


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
    out: dict[str, str] = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = _normalize_key(row.get("alias") or "")
            item_id = (row.get("item_id") or "").strip()
            if alias and item_id:
                out[alias] = item_id
    return out


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


def display_name(item_id: str, items_meta: dict[str, dict[str, str]] | None, locale: str) -> str:
    meta = (items_meta or {}).get(item_id) or {}
    if str(locale or "").lower().startswith("en"):
        return meta.get("display_name_en") or item_id
    return meta.get("display_name_es") or item_id


def load_calorie_db(csv_path: Path) -> dict[str, CalorieItem]:
    items: dict[str, CalorieItem] = {}
    if not csv_path.exists():
        return items
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


def load_nutrition_db(csv_path: Path) -> dict[str, NutritionItem]:
    items: dict[str, NutritionItem] = {}
    if not csv_path.exists():
        return items
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue
            items[item_id] = NutritionItem(
                item_id=item_id,
                kcal_per_100g=_to_float(row.get("kcal_per_100g")),
                protein_g_per_100g=_to_float(row.get("protein_g_per_100g")),
                carbs_g_per_100g=_to_float(row.get("carbs_g_per_100g")),
                fat_g_per_100g=_to_float(row.get("fat_g_per_100g")),
                fiber_g_per_100g=_to_float(row.get("fiber_g_per_100g")),
            )
    return items


def load_micros_db(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            micro = (row.get("micro") or "").strip()
            level = (row.get("level") or "").strip()
            if not (item_id and micro and level):
                continue
            out.setdefault(item_id, []).append({"micro": micro, "level": level})
    return out


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def parse_portion(item_id: str, portion_text: str | None, calorie_db: dict[str, CalorieItem]) -> tuple[float | None, str]:
    text = (portion_text or "").strip().lower()
    if not text:
        return None, "unknown"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*g\b", text)
    if m:
        grams = _safe_float(m.group(1).replace(",", "."))
        if grams and grams > 0:
            return float(grams), "explicit_grams"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(unidad|unidades)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if not n or n <= 0:
            return None, "unknown"
        ci = calorie_db.get(item_id)
        if ci and ci.unit_grams:
            return float(n) * float(ci.unit_grams), "explicit_units"

    if "taza" in text and item_id == "arroz_blanco_cocido":
        return 160.0, "cup"

    if item_id == "ensalada_mixta" and "grande" in text:
        return 250.0, "qualitative"

    return None, "unknown"


def portion_confidence(method: str) -> float:
    return {
        "explicit_grams": 1.0,
        "explicit_units": 0.8,
        "cup": 0.7,
        "qualitative": 0.55,
        "default": 0.5,
        "unknown": 0.2,
        "memory": 0.85,
    }.get(method, 0.2)


def build_portion_candidates(
    item_id: str,
    *,
    calorie_db: dict[str, CalorieItem],
    memory_hint_grams: float | None = None,
    center_grams: float | None = None,
) -> list[dict[str, Any]]:
    def _mk(label: str, grams: float, conf: float) -> dict[str, Any]:
        return {"label": label, "grams": round(float(grams), 2), "confidence_hint": round(float(conf), 4)}

    if memory_hint_grams and memory_hint_grams > 0:
        g = float(memory_hint_grams)
        return [_mk("Repetir", g, 0.9), _mk("Menos", g * 0.9, 0.8), _mk("Más", g * 1.1, 0.8)]

    if center_grams and center_grams > 0:
        g = float(center_grams)
        return [_mk("Menos", g * 0.9, 0.85), _mk("Estimado", g, 0.9), _mk("Más", g * 1.1, 0.85)]

    ci = calorie_db.get(item_id)
    if ci and ci.unit_grams:
        ug = float(ci.unit_grams)
        return [_mk("1 unidad", ug, 0.85), _mk("2 unidades", ug * 2.0, 0.8), _mk("3 unidades", ug * 3.0, 0.75)]

    if ci and ci.default_serving_grams:
        c = float(ci.default_serving_grams)
        return [_mk("Pequeña", max(10.0, c * 0.75), 0.65), _mk("Típica", c, 0.75), _mk("Grande", c * 1.25, 0.65)]

    return [_mk("Pequeña", 100.0, 0.5), _mk("Media", 200.0, 0.55), _mk("Grande", 300.0, 0.5)]


def _kcal_for_grams(item_id: str, grams: float | None, *, calorie_db: dict[str, CalorieItem], nutrition_db: dict[str, NutritionItem]) -> float | None:
    if grams is None or grams <= 0:
        return None

    n = nutrition_db.get(item_id)
    if n and n.kcal_per_100g is not None:
        return (float(grams) / 100.0) * float(n.kcal_per_100g)

    ci = calorie_db.get(item_id)
    if not ci:
        return None
    if ci.kcal_per_100g is not None:
        return (float(grams) / 100.0) * float(ci.kcal_per_100g)
    if ci.kcal_per_unit is not None and ci.unit_grams:
        return (float(grams) / float(ci.unit_grams)) * float(ci.kcal_per_unit)
    if ci.kcal_per_unit is not None:
        return float(ci.kcal_per_unit)

    return None


def _macros_for_grams(n: NutritionItem, grams: float) -> dict[str, float]:
    g = float(grams)
    out = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    if n.protein_g_per_100g is not None:
        out["protein_g"] = (g / 100.0) * float(n.protein_g_per_100g)
    if n.carbs_g_per_100g is not None:
        out["carbs_g"] = (g / 100.0) * float(n.carbs_g_per_100g)
    if n.fat_g_per_100g is not None:
        out["fat_g"] = (g / 100.0) * float(n.fat_g_per_100g)
    if n.fiber_g_per_100g is not None:
        out["fiber_g"] = (g / 100.0) * float(n.fiber_g_per_100g)
    return out


def normalize_item(raw: str, *, aliases: dict[str, str], fuzzy_cutoff: float = 0.86) -> str | None:
    key = _normalize_key(raw)
    if not key:
        return None
    if key in aliases:
        return aliases[key]

    best, score = _best_fuzzy_match(key, list(aliases.keys()))
    if best and score >= fuzzy_cutoff:
        return aliases[best]

    toks = _tokenize(key)
    if any(t.startswith("arroz") for t in toks):
        return "arroz_blanco_cocido"
    if any(t.startswith("pollo") for t in toks):
        return "pollo_pechuga_asada"
    return None


def shannon_entropy(probs: list[float]) -> float:
    p = [x for x in probs if x and x > 0]
    if not p:
        return 0.0
    s = sum(p)
    p = [x / s for x in p]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def qaf_estimate_v2(
    vision: dict[str, Any],
    *,
    calorie_db: dict[str, CalorieItem],
    nutrition_db: dict[str, NutritionItem],
    micros_db: dict[str, list[dict[str, Any]]],
    aliases: dict[str, str],
    items_meta: dict[str, dict[str, str]] | None = None,
    locale: str = DEFAULT_LOCALE,
    memory_hint_by_item: dict[str, float] | None = None,
    confirmed_portions: list[dict[str, Any]] | None = None,
    goal_kcal_meal: float | None = None,
) -> dict[str, Any]:
    memory_hint_by_item = memory_hint_by_item or {}

    raw_items = vision.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    portion_text = str(vision.get("portion_estimate") or "").strip() or None

    confirmed_map: dict[str, float] = {}
    if isinstance(confirmed_portions, list):
        for cp in confirmed_portions:
            if not isinstance(cp, dict):
                continue
            iid = str(cp.get("item_id") or "").strip()
            g = _safe_float(cp.get("grams"))
            if iid and g and g > 0:
                confirmed_map[iid] = float(g)

    normalized_ids: list[str] = []
    for it in raw_items:
        raw = str(it)
        iid = normalize_item(raw, aliases=aliases)
        if iid and iid not in normalized_ids:
            normalized_ids.append(iid)

    items_out: list[dict[str, Any]] = []
    missing_items: list[str] = []
    reasons: list[str] = []

    per_item_conf: dict[str, dict[str, Any]] = {}

    # Totales
    macros_total = {"protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    micros_pool: dict[str, str] = {}
    level_rank = {"alto": 3, "medio": 2, "bajo": 1}

    for item_id in normalized_ids:
        ci = calorie_db.get(item_id)

        grams, portion_method = parse_portion(item_id, portion_text, calorie_db)
        portion_match = None
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
            used_default = True

        kcal = _kcal_for_grams(item_id, grams, calorie_db=calorie_db, nutrition_db=nutrition_db)
        if kcal is None:
            missing_items.append(item_id)
            reasons.append(f"missing_item_in_db:{item_id}")
            continue

        conf_item = 1.0
        conf_portion = portion_confidence(portion_method)
        conf_combined = max(0.0, min(1.0, 0.65 * conf_item + 0.35 * conf_portion))

        if used_default:
            reasons.append(f"used_default_serving:{item_id}")

        per_item_conf[item_id] = {
            "confidence_item": round(conf_item, 4),
            "confidence_portion": round(conf_portion, 4),
            "confidence": round(conf_combined, 4),
            "item_method": "catalog",
            "portion_method": portion_method,
            "matched_alias": None,
            "portion_match": portion_match,
        }

        center = None
        if grams is not None and portion_method in {"explicit_grams", "explicit_units", "cup", "qualitative"}:
            center = float(grams)

        candidates = build_portion_candidates(
            item_id,
            calorie_db=calorie_db,
            memory_hint_grams=memory_hint_by_item.get(item_id),
            center_grams=center,
        )

        # Macros
        n = nutrition_db.get(item_id)
        item_macros = None
        if n is not None and grams is not None:
            item_macros = _macros_for_grams(n, float(grams))
            for k in macros_total.keys():
                macros_total[k] += float(item_macros.get(k) or 0.0)

        # Micros
        for m in micros_db.get(item_id) or []:
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
                "micros": micros_db.get(item_id) or [],
            }
        )

    # Rango y estimate
    total_kcal_raw = sum(float(x.get("calories") or 0.0) for x in items_out)

    low = 0.0
    high = 0.0
    best = 0.0
    per_item_spread: dict[str, float] = {}

    for it in items_out:
        iid = str(it.get("item_id") or "")
        cands = it.get("portion_candidates") or []
        current = float(it.get("calories") or 0.0)
        if not cands:
            low += current
            high += current
            best += current
            continue

        kcal_opts = []
        for c in cands:
            g = _safe_float(c.get("grams"))
            if not g or g <= 0:
                continue
            k = _kcal_for_grams(iid, float(g), calorie_db=calorie_db, nutrition_db=nutrition_db)
            if k is None or k < 0:
                continue
            kcal_opts.append(float(k))

        if not kcal_opts:
            low += current
            high += current
            best += current
            continue

        item_low = min(kcal_opts)
        item_high = max(kcal_opts)
        item_mid = sorted(kcal_opts)[len(kcal_opts) // 2]

        low += item_low
        high += item_high

        pm = str((per_item_conf.get(iid) or {}).get("portion_method") or "")
        best += float(item_mid if pm in {"default", "unknown"} else (current or item_mid))

        per_item_spread[iid] = float(item_high - item_low)

    range_driver = None
    if per_item_spread:
        range_driver = sorted(per_item_spread.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    total_conf = 0.0
    if items_out and best > 0:
        for it in items_out:
            iid = str(it.get("item_id") or "")
            kcal = float(it.get("calories") or 0.0)
            w = kcal / best if best else 0.0
            c = float((per_item_conf.get(iid) or {}).get("confidence") or 0.0)
            total_conf += w * c

    uncertainty_score = max(0.0, min(1.0, 1.0 - float(total_conf)))

    range_width = max(0.0, float(high) - float(low))
    rel_width = (range_width / float(best)) if best else 1.0

    crosses_goal = False
    if goal_kcal_meal is not None:
        try:
            gk = float(goal_kcal_meal)
            crosses_goal = float(low) <= gk <= float(high)
        except Exception:
            crosses_goal = False

    needs_confirmation = bool(missing_items) or (rel_width >= 0.35) or (range_width >= 180.0) or crosses_goal

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

    # Micros highlights (alto->medio->bajo)
    micros_sorted = sorted(micros_pool.items(), key=lambda kv: level_rank.get(kv[1], 0), reverse=True)
    micros_out = [f"{micro} ({level})" for (micro, level) in micros_sorted if micro and level]

    probs = [1.0 for _ in normalized_ids]
    ent = shannon_entropy(probs)

    out = {
        "items": items_out,
        "missing_items": missing_items,
        "total_calories": round(float(best), 2),
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
        "range_driver_display": display_name(range_driver, items_meta, locale) if range_driver else None,
        "goal_kcal_meal": goal_kcal_meal,
        "macros_total": {k: round(float(v), 2) for k, v in macros_total.items()},
        "micros_highlights": micros_out[:8],
    }

    out["ui_blocks"] = build_ui_blocks(out)
    return out


def build_ui_blocks(qaf: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if not isinstance(qaf, dict):
        return blocks

    total = qaf.get("total_calories")
    rng = qaf.get("total_calories_range") or {}
    low = rng.get("low")
    high = rng.get("high")
    driver_name = qaf.get("range_driver_display") or qaf.get("range_driver")

    if total is not None and low is not None and high is not None:
        blocks.append(
            {
                "type": "summary",
                "text": f"Estimado: {round(float(total), 0):.0f} kcal (rango {round(float(low), 0):.0f}–{round(float(high), 0):.0f})",
            }
        )
    if driver_name:
        blocks.append({"type": "range_driver", "text": f"Lo que más mueve el rango: {driver_name}"})

    macros = qaf.get("macros_total") or {}
    if macros:
        p = float(macros.get("protein_g") or 0.0)
        c = float(macros.get("carbs_g") or 0.0)
        f = float(macros.get("fat_g") or 0.0)
        blocks.append({"type": "macros", "text": f"Macros aprox: P {p:.0f}g | C {c:.0f}g | G {f:.0f}g"})

    micros = qaf.get("micros_highlights") or []
    if micros:
        blocks.append({"type": "micros", "text": "Micros probables: " + ", ".join(micros[:5])})

    return blocks


def render_professional_summary(qaf: dict[str, Any]) -> str:
    if not isinstance(qaf, dict):
        return ""

    lines = []
    for b in build_ui_blocks(qaf):
        if b.get("type") in {"summary", "range_driver", "macros"}:
            t = str(b.get("text") or "").strip()
            if t:
                lines.append(t)

    if qaf.get("needs_confirmation") and (qaf.get("follow_up_questions") or []):
        q = (qaf.get("follow_up_questions") or [])[0]
        prompt = str(q.get("prompt") or "").strip()
        if prompt:
            lines.append(prompt)

    return "\n".join(lines).strip()
