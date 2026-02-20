from __future__ import annotations

import re
from typing import Any

from .catalogs import CalorieItem

DEFAULT_LOCALE = "es-CO"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def portion_confidence(method: str) -> float:
    return {
        "explicit_grams": 1.0,
        "explicit_units": 0.8,
        "explicit_ml": 0.9,
        "cup": 0.7,
        "slice": 0.75,
        "qualitative": 0.55,
        "default": 0.5,
        "unknown": 0.2,
        "memory": 0.85,
    }.get(method, 0.2)


def parse_portion(item_id: str, portion_text: str | None, calorie_db: dict[str, CalorieItem]) -> tuple[float | None, str, str | None]:
    """Retorna (grams, method, match_phrase)."""
    text = (portion_text or "").strip().lower()
    if not text:
        return None, "unknown", None

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*g\b", text)
    if m:
        grams = _safe_float(m.group(1).replace(",", "."))
        if grams and grams > 0:
            return float(grams), "explicit_grams", f"{m.group(1)}g"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*ml\b", text)
    if m:
        ml = _safe_float(m.group(1).replace(",", "."))
        if ml and ml > 0:
            return float(ml), "explicit_ml", f"{m.group(1)}ml"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(unidad|unidades)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if not n or n <= 0:
            return None, "unknown", None
        ci = calorie_db.get(item_id)
        if ci and ci.unit_grams:
            return float(n) * float(ci.unit_grams), "explicit_units", f"{m.group(1)} unidades"
        if item_id == "huevo":
            return float(n) * 50.0, "explicit_units", f"{m.group(1)} unidades"

    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(rebanada|rebanadas)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if n and n > 0:
            ci = calorie_db.get(item_id)
            if ci and ci.unit_grams:
                return float(n) * float(ci.unit_grams), "slice", f"{m.group(1)} rebanadas"

    if item_id == "ensalada_mixta" and "ensalada" in text and re.search(r"\bgrande\b", text):
        return 250.0, "qualitative", "ensalada grande"

    if "taza" in text:
        if item_id == "arroz_blanco_cocido":
            return 160.0, "cup", "taza"
        return 150.0, "cup", "taza"

    return None, "unknown", None


def build_portion_candidates(
    item_id: str,
    *,
    calorie_db: dict[str, CalorieItem],
    locale: str = DEFAULT_LOCALE,
    memory_hint_grams: float | None = None,
    center_grams: float | None = None,
    center_source: str | None = None,
    scale_hints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Devuelve exactamente 3 opciones. scale_hints ajusta levemente el centro (si existe)."""

    ci = calorie_db.get(item_id)

    def _mk(label: str, grams: float, conf: float) -> dict[str, Any]:
        return {"label": label, "grams": round(float(grams), 2), "confidence_hint": round(float(conf), 4)}

    # Ajuste leve por scale_hints (MVP): si dicen plato grande -> +10% al centro.
    adj = 1.0
    if isinstance(scale_hints, dict):
        plate = str(scale_hints.get("plate") or "").lower().strip()
        if plate in {"large", "grande"}:
            adj = 1.1
        elif plate in {"small", "pequeño", "pequeno"}:
            adj = 0.9

    if center_grams and center_grams > 0 and (center_source in {"explicit", "inferred"}):
        g = float(center_grams) * adj
        return [_mk("Menos", g * 0.9, 0.85), _mk("Estimado", g, 0.9), _mk("Más", g * 1.1, 0.85)]

    if memory_hint_grams and memory_hint_grams > 0:
        g = float(memory_hint_grams) * adj
        return [_mk("Repetir última porción", g, 0.9), _mk("Menos", g * 0.9, 0.8), _mk("Más", g * 1.1, 0.8)]

    if ci and ci.unit_grams:
        ug = float(ci.unit_grams)
        return [_mk("1 unidad", 1.0 * ug, 0.85), _mk("2 unidades", 2.0 * ug, 0.8), _mk("3 unidades", 3.0 * ug, 0.75)]

    if ci and ci.default_serving_grams:
        c = float(ci.default_serving_grams) * adj
        return [_mk("Porción pequeña", max(10.0, c * 0.75), 0.65), _mk("Porción típica", c, 0.75), _mk("Porción grande", c * 1.25, 0.65)]

    if item_id == "arroz_blanco_cocido":
        return [_mk("1/2 taza", 80.0, 0.7), _mk("1 taza", 160.0, 0.75), _mk("1.5 tazas", 240.0, 0.7)]
    if item_id == "pollo_pechuga_asada":
        return [_mk("100 g", 100.0, 0.7), _mk("150 g", 150.0, 0.75), _mk("200 g", 200.0, 0.7)]

    return [_mk("Porción pequeña", 100.0, 0.5), _mk("Porción media", 200.0, 0.55), _mk("Porción grande", 300.0, 0.5)]
