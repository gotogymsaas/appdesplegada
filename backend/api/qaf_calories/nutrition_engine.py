from __future__ import annotations

from typing import Any

from .catalogs import NutritionItem


def kcal_from_grams(n: NutritionItem, grams: float) -> float:
    if n.kcal_per_100g is None:
        return 0.0
    return (float(grams) / 100.0) * float(n.kcal_per_100g)


def macros_from_grams(n: NutritionItem, grams: float) -> dict[str, float]:
    g = float(grams)
    out = {
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "fiber_g": 0.0,
    }
    if n.protein_g_per_100g is not None:
        out["protein_g"] = (g / 100.0) * float(n.protein_g_per_100g)
    if n.carbs_g_per_100g is not None:
        out["carbs_g"] = (g / 100.0) * float(n.carbs_g_per_100g)
    if n.fat_g_per_100g is not None:
        out["fat_g"] = (g / 100.0) * float(n.fat_g_per_100g)
    if n.fiber_g_per_100g is not None:
        out["fiber_g"] = (g / 100.0) * float(n.fiber_g_per_100g)
    return out


def macro_calorie_percentages(total: dict[str, float]) -> dict[str, float]:
    # 4/4/9 kcal per g
    kcal_p = float(total.get("protein_g") or 0.0) * 4.0
    kcal_c = float(total.get("carbs_g") or 0.0) * 4.0
    kcal_f = float(total.get("fat_g") or 0.0) * 9.0
    total_kcal = kcal_p + kcal_c + kcal_f
    if total_kcal <= 0:
        return {"protein_pct": 0.0, "carbs_pct": 0.0, "fat_pct": 0.0}
    return {
        "protein_pct": (kcal_p / total_kcal) * 100.0,
        "carbs_pct": (kcal_c / total_kcal) * 100.0,
        "fat_pct": (kcal_f / total_kcal) * 100.0,
    }


def micros_highlights(micros: list[dict[str, Any]] | None) -> list[str]:
    # devuelve strings tipo "Vitamina A (alto)"
    out = []
    for m in micros or []:
        micro = str(m.get("micro") or "").strip()
        level = str(m.get("level") or "").strip()
        if micro and level:
            out.append(f"{micro} ({level})")
    return out
