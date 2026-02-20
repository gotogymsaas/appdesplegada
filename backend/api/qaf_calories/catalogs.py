from __future__ import annotations

import csv
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


def load_aliases(csv_path: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not csv_path.exists():
        return aliases
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = (row.get("alias") or "").strip().lower()
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
    """micros_db.csv: item_id,micro,level (alto|medio|bajo)"""
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
