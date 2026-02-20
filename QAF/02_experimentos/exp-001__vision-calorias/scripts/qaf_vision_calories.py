from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalorieItem:
    item_id: str
    kcal_per_100g: float | None
    kcal_per_unit: float | None
    unit_grams: float | None
    default_serving_grams: float | None


def load_calorie_db(csv_path: Path) -> dict[str, CalorieItem]:
    items: dict[str, CalorieItem] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue

            def _to_float(value: str | None) -> float | None:
                raw = (value or "").strip()
                if not raw:
                    return None
                try:
                    return float(raw)
                except Exception:
                    return None

            items[item_id] = CalorieItem(
                item_id=item_id,
                kcal_per_100g=_to_float(row.get("kcal_per_100g")),
                kcal_per_unit=_to_float(row.get("kcal_per_unit")),
                unit_grams=_to_float(row.get("unit_grams")),
                default_serving_grams=_to_float(row.get("default_serving_grams")),
            )
    return items


_NORMALIZATION_MAP: dict[str, str] = {
    "manzana": "manzana",
    "apple": "manzana",
    "arroz": "arroz_blanco_cocido",
    "arroz blanco": "arroz_blanco_cocido",
    "arroz blanco cocido": "arroz_blanco_cocido",
    "pollo": "pollo_pechuga_asada",
    "pechuga": "pollo_pechuga_asada",
    "pechuga de pollo": "pollo_pechuga_asada",
    "ensalada": "ensalada_mixta",
    "ensalada mixta": "ensalada_mixta",
    "papas fritas": "papa_frita",
    "papa frita": "papa_frita",
    "huevo": "huevo",
    "banano": "banano",
    "banana": "banano",
    "pan": "pan_rebanada",
    "rebanada de pan": "pan_rebanada",
    "leche": "leche_entera",
    "leche entera": "leche_entera",
}


def normalize_item(raw: str) -> str | None:
    key = re.sub(r"\s+", " ", (raw or "").strip().lower())
    key = re.sub(r"[^a-záéíóúñü0-9 ]", "", key)
    key = key.strip()
    if not key:
        return None
    if key in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[key]

    # heurística mínima: si contiene 'arroz' o 'pollo'
    if "arroz" in key:
        return "arroz_blanco_cocido"
    if "pollo" in key:
        return "pollo_pechuga_asada"
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def parse_portion_grams(item_id: str, portion_text: str | None, calorie_db: dict[str, CalorieItem]) -> float | None:
    """Heurística MVP para porciones.

    Soporta:
    - "150g" / "150 g"
    - "2 unidades" (si hay unit_grams)
    - "1 taza" (default 160g para arroz cocido si aplica)
    """

    text = (portion_text or "").strip().lower()
    if not text:
        return None

    # 150g
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*g\b", text)
    if m:
        value = m.group(1).replace(",", ".")
        grams = _safe_float(value)
        if grams and grams > 0:
            return grams

    # 2 unidades
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(unidad|unidades)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if not n or n <= 0:
            return None
        ci = calorie_db.get(item_id)
        if ci and ci.unit_grams:
            return float(n) * float(ci.unit_grams)
        # fallback por si no hay unit_grams
        if item_id == "huevo":
            return float(n) * 50.0

    # 1 taza (solo MVP)
    if "taza" in text:
        if item_id == "arroz_blanco_cocido":
            return 160.0
        # default genérico
        return 150.0

    return None


def shannon_entropy(probs: list[float]) -> float:
    p = [x for x in probs if x and x > 0]
    if not p:
        return 0.0
    s = sum(p)
    p = [x / s for x in p]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def qaf_infer_from_vision(
    vision: dict[str, Any],
    calorie_db: dict[str, CalorieItem],
) -> dict[str, Any]:
    raw_items = vision.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    normalized: list[str] = []
    for it in raw_items:
        norm = normalize_item(str(it))
        if norm and norm not in normalized:
            normalized.append(norm)

    portion_text = str(vision.get("portion_estimate") or "").strip() or None

    # Probabilidades: si no vienen, uniformes.
    probs = [1.0 for _ in normalized]
    ent = shannon_entropy(probs)

    items_out = []
    total_kcal = 0.0
    missing = []

    for item_id in normalized:
        ci = calorie_db.get(item_id)
        grams = parse_portion_grams(item_id, portion_text, calorie_db)
        if grams is None and ci and ci.default_serving_grams:
            grams = float(ci.default_serving_grams)

        kcal = None
        if ci:
            if grams is not None and ci.kcal_per_100g is not None:
                kcal = (float(grams) / 100.0) * float(ci.kcal_per_100g)
            elif ci.kcal_per_unit is not None:
                kcal = float(ci.kcal_per_unit)
                grams = grams or ci.unit_grams

        if kcal is None:
            missing.append(item_id)
            continue

        total_kcal += float(kcal)
        items_out.append(
            {
                "item_id": item_id,
                "grams": grams,
                "calories": round(float(kcal), 2),
            }
        )

    # Incertidumbre: MVP.
    # Si hay items desconocidos o no se pudo inferir gramos, pedimos confirmación.
    needs_confirmation = bool(missing) or any(x.get("grams") is None for x in items_out)
    # Entropía alta (para cuando más adelante haya probabilidades reales): umbral genérico.
    if ent >= 1.0 and len(normalized) >= 3:
        needs_confirmation = True

    # Rango (heurístico): +/- 25% si falta porción; +/- 15% si todo ok.
    base_pct = 0.25 if any(x.get("grams") is None for x in items_out) else 0.15
    low = max(0.0, total_kcal * (1.0 - base_pct))
    high = total_kcal * (1.0 + base_pct)

    return {
        "is_food": bool(vision.get("is_food")) if "is_food" in vision else None,
        "items": items_out,
        "missing_items": missing,
        "total_calories": round(total_kcal, 2),
        "total_calories_range": {"low": round(low, 2), "high": round(high, 2)},
        "uncertainty": {
            "entropy": round(ent, 4),
            "portion_text": portion_text,
        },
        "needs_confirmation": bool(needs_confirmation),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calorie-db", required=True)
    parser.add_argument("--vision-json", required=True, help="JSON string o ruta a .json")
    args = parser.parse_args()

    calorie_db = load_calorie_db(Path(args.calorie_db))

    vision_arg = args.vision_json
    if Path(vision_arg).exists():
        vision = json.loads(Path(vision_arg).read_text(encoding="utf-8"))
    else:
        vision = json.loads(vision_arg)

    result = qaf_infer_from_vision(vision, calorie_db)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
