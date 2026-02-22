from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from ..qaf_calories.engine import (
    DEFAULT_LOCALE,
    load_items_meta,
    load_micros_db,
    load_nutrition_db,
    paths as qaf_paths,
)


VarietyLevel = Literal["simple", "normal", "high"]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _grams_for_kcal(item_id: str, kcal: float, nutrition_db: dict[str, Any]) -> float:
    n = nutrition_db.get(item_id)
    kcal100 = float(getattr(n, "kcal_per_100g", 0.0) or 0.0) if n else 0.0
    if kcal100 <= 0:
        return 0.0
    return max(0.0, (float(kcal) / kcal100) * 100.0)


def _kcal_for_grams(item_id: str, grams: float, nutrition_db: dict[str, Any]) -> float:
    n = nutrition_db.get(item_id)
    kcal100 = float(getattr(n, "kcal_per_100g", 0.0) or 0.0) if n else 0.0
    if kcal100 <= 0:
        return 0.0
    return (float(grams) / 100.0) * kcal100


def _display_name(item_id: str, items_meta: dict[str, dict[str, str]] | None, locale: str) -> str:
    meta = (items_meta or {}).get(item_id) or {}
    if str(locale or "").lower().startswith("en"):
        return (meta.get("display_name_en") or meta.get("display_name_es") or item_id).strip() or item_id
    return (meta.get("display_name_es") or meta.get("display_name_en") or item_id).strip() or item_id


def _classify_pools(nutrition_db: dict[str, Any], exclude: set[str]) -> dict[str, list[str]]:
    pools = {"protein": [], "carb": [], "veg": [], "fat": [], "fruit": []}

    all_items: list[str] = []
    for iid, n in nutrition_db.items():
        if iid in exclude:
            continue
        kcal = float(getattr(n, "kcal_per_100g", 0.0) or 0.0)
        if kcal <= 0:
            continue
        all_items.append(iid)
        p = float(getattr(n, "protein_g_per_100g", 0.0) or 0.0)
        c = float(getattr(n, "carbs_g_per_100g", 0.0) or 0.0)
        f = float(getattr(n, "fat_g_per_100g", 0.0) or 0.0)
        fb = float(getattr(n, "fiber_g_per_100g", 0.0) or 0.0)

        if p >= 15 and c <= 8:
            pools["protein"].append(iid)
        if c >= 20 and f <= 8:
            pools["carb"].append(iid)
        if fb >= 2 and kcal <= 80:
            pools["veg"].append(iid)
        if f >= 10 and c <= 10:
            pools["fat"].append(iid)
        if c >= 10 and fb >= 1 and kcal <= 90:
            pools["fruit"].append(iid)

    for k in list(pools.keys()):
        if not pools[k]:
            pools[k] = all_items[:]
    return pools


def _micro_coverage(plan: dict[str, Any], micros_db: dict[str, list[dict[str, Any]]]) -> float:
    found: set[str] = set()
    for day in plan.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                for m in micros_db.get(iid) or []:
                    lvl = str(m.get("level") or "").lower()
                    if lvl in ("alto", "medio"):
                        micro = str(m.get("micro") or "").strip()
                        if micro:
                            found.add(micro)
    return min(1.0, len(found) / 16.0)


def _variety_penalty(plan: dict[str, Any]) -> float:
    ids: list[str] = []
    for day in plan.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                if iid:
                    ids.append(iid)
    if not ids:
        return 1.0
    c = Counter(ids)
    repeats = sum(max(0, v - 1) for v in c.values())
    return min(1.0, repeats / max(1, len(ids)))


def _friction_penalty(plan: dict[str, Any]) -> float:
    ids: set[str] = set()
    for day in plan.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                if iid:
                    ids.add(iid)
    return min(1.0, max(0.0, (len(ids) - 18) / 18.0))


@dataclass(frozen=True)
class PlanScore:
    kcal_loss: float
    micro_coverage: float
    variety_penalty: float
    friction_penalty: float
    total: float


def _score_plan(plan: dict[str, Any], *, kcal_day_target: float, micros_db: dict[str, Any], variety_level: VarietyLevel) -> PlanScore:
    days = plan.get("days") or []
    kcal_err2 = 0.0
    for day in days:
        kcal = float(day.get("total_kcal") or 0.0)
        e = (kcal - float(kcal_day_target)) / max(1.0, float(kcal_day_target))
        kcal_err2 += e * e
    kcal_loss = kcal_err2 / max(1, len(days))

    micro = _micro_coverage(plan, micros_db)
    var_pen = _variety_penalty(plan)
    fr_pen = _friction_penalty(plan)

    if variety_level == "high":
        wv, wf = 0.9, 0.4
    elif variety_level == "simple":
        wv, wf = 0.3, 0.8
    else:
        wv, wf = 0.6, 0.6

    total = (1.2 * kcal_loss) + (0.9 * (1.0 - micro)) + (wv * var_pen) + (wf * fr_pen)
    return PlanScore(kcal_loss=kcal_loss, micro_coverage=micro, variety_penalty=var_pen, friction_penalty=fr_pen, total=total)


def _build_day(*, rng: random.Random, nutrition_db: dict[str, Any], items_meta: dict[str, Any], pools: dict[str, list[str]], kcal_day: float, meals_per_day: int, locale: str) -> dict[str, Any]:
    if meals_per_day == 4:
        shares = [0.25, 0.35, 0.30, 0.10]
        slots = ["desayuno", "almuerzo", "cena", "snack"]
    else:
        shares = [0.30, 0.40, 0.30]
        slots = ["desayuno", "almuerzo", "cena"]

    meals: list[dict[str, Any]] = []
    total_kcal = 0.0
    for slot, share in zip(slots, shares):
        target = float(kcal_day) * float(share)
        if slot == "snack":
            chosen = [rng.choice(pools["fruit"]), rng.choice(pools["fat"])]
            split = [0.65, 0.35]
        else:
            chosen = [rng.choice(pools["protein"]), rng.choice(pools["carb"]), rng.choice(pools["veg"])]
            split = [0.40, 0.40, 0.20]

        items = []
        kcal_meal = 0.0
        for iid, frac in zip(chosen, split):
            grams = _grams_for_kcal(iid, target * frac, nutrition_db)
            grams = round(grams / 5.0) * 5.0
            kcal = _kcal_for_grams(iid, grams, nutrition_db)
            kcal_meal += kcal
            items.append(
                {
                    "item_id": iid,
                    "name": _display_name(iid, items_meta, locale),
                    "grams": float(grams),
                    "kcal": round(float(kcal), 1),
                }
            )
        total_kcal += kcal_meal
        meals.append({"slot": slot, "items": items, "kcal": round(float(kcal_meal), 1)})
    return {"meals": meals, "total_kcal": round(float(total_kcal), 1)}


def generate_week_plan(
    *,
    kcal_day: float,
    meals_per_day: int,
    variety_level: VarietyLevel,
    exclude_item_ids: list[str] | None,
    seed: int,
    locale: str,
) -> dict[str, Any]:
    p = qaf_paths()
    items_meta = load_items_meta(p["items_meta"])
    nutrition_db = load_nutrition_db(p["nutrition_db"])
    micros_db = load_micros_db(p["micros_db"])

    exclude = {str(x).strip() for x in (exclude_item_ids or []) if str(x).strip()}
    pools = _classify_pools(nutrition_db, exclude)

    rng = random.Random(int(seed) & 0xFFFFFFFF)

    pop_n = 24
    gens = 18
    elite = 6

    def new_candidate() -> dict[str, Any]:
        return {
            "days": [
                _build_day(
                    rng=rng,
                    nutrition_db=nutrition_db,
                    items_meta=items_meta,
                    pools=pools,
                    kcal_day=float(kcal_day),
                    meals_per_day=int(meals_per_day),
                    locale=locale,
                )
                for _ in range(7)
            ]
        }

    def mutate(plan: dict[str, Any]) -> dict[str, Any]:
        out = json.loads(json.dumps(plan))
        days = out.get("days") or []
        if not days:
            return out
        d = rng.choice(days)
        meals = d.get("meals") or []
        if not meals:
            return out
        meal = rng.choice(meals)
        items = meal.get("items") or []
        if not items:
            return out
        idx = rng.randrange(0, len(items))

        iid_old = str(items[idx].get("item_id") or "")
        pool_key = "veg"
        if iid_old in pools.get("protein", []):
            pool_key = "protein"
        elif iid_old in pools.get("carb", []):
            pool_key = "carb"
        elif iid_old in pools.get("fat", []):
            pool_key = "fat"
        elif iid_old in pools.get("fruit", []):
            pool_key = "fruit"

        iid_new = rng.choice(pools.get(pool_key) or pools["carb"])
        kcal_target = float(items[idx].get("kcal") or 0.0)
        grams = _grams_for_kcal(iid_new, kcal_target, nutrition_db)
        grams = round(grams / 5.0) * 5.0
        kcal_new = _kcal_for_grams(iid_new, grams, nutrition_db)

        items[idx] = {
            "item_id": iid_new,
            "name": _display_name(iid_new, items_meta, locale),
            "grams": float(grams),
            "kcal": round(float(kcal_new), 1),
        }

        meal["kcal"] = round(float(sum(float(it.get("kcal") or 0.0) for it in items)), 1)
        d["total_kcal"] = round(float(sum(float(m.get("kcal") or 0.0) for m in meals)), 1)
        return out

    population = [new_candidate() for _ in range(pop_n)]
    best = None
    best_s = None
    for _ in range(gens):
        scored = [(p, _score_plan(p, kcal_day_target=float(kcal_day), micros_db=micros_db, variety_level=variety_level)) for p in population]
        scored.sort(key=lambda x: x[1].total)
        if best is None or scored[0][1].total < (best_s.total if best_s else 1e9):
            best, best_s = scored[0]

        next_pop = [scored[i][0] for i in range(min(elite, len(scored)))]
        while len(next_pop) < pop_n:
            parent = rng.choice(next_pop)
            next_pop.append(mutate(parent))
        population = next_pop

    assert best is not None and best_s is not None

    shop_g = Counter()
    for day in best.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                grams = float(it.get("grams") or 0.0)
                if iid and grams > 0:
                    shop_g[iid] += grams
    shopping = [
        {"item_id": iid, "name": _display_name(iid, items_meta, locale), "grams": round(float(g), 1)}
        for iid, g in shop_g.most_common()
    ]

    return {
        "algorithm": "exp-004_meal_planner_v0",
        "as_of": str(date.today()),
        "inputs": {
            "kcal_day": float(kcal_day),
            "meals_per_day": int(meals_per_day),
            "variety": variety_level,
        },
        "plan": best,
        "shopping_list": shopping,
        "scores": {
            "kcal_loss": round(float(best_s.kcal_loss), 4),
            "micro_coverage": round(float(best_s.micro_coverage), 4),
            "variety_penalty": round(float(best_s.variety_penalty), 4),
            "friction_penalty": round(float(best_s.friction_penalty), 4),
            "total": round(float(best_s.total), 4),
        },
    }


def render_shopping_list_text(result: dict[str, Any], *, max_items: int = 18) -> str:
    if not isinstance(result, dict):
        return ""
    shopping = result.get("shopping_list")
    if not isinstance(shopping, list) or not shopping:
        return ""
    lines: list[str] = []
    lines.append("Lista de compras (aprox):")
    for row in shopping[: max(1, int(max_items))]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        grams = row.get("grams")
        if not name:
            continue
        try:
            g = float(grams)
            lines.append(f"- {name}: {round(g, 0):.0f} g")
        except Exception:
            lines.append(f"- {name}")
    if len(shopping) > max_items:
        lines.append(f"… +{len(shopping) - max_items} items")
    return "\n".join(lines).strip()


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    inputs = result.get("inputs") if isinstance(result.get("inputs"), dict) else {}
    scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    days = plan.get("days") if isinstance(plan.get("days"), list) else []

    out: list[str] = []
    out.append("Menú semanal (QAF):")
    out.append(f"kcal_day_target: {inputs.get('kcal_day')}")
    out.append(f"meals_per_day: {inputs.get('meals_per_day')}")
    out.append(f"micro_coverage: {scores.get('micro_coverage')}")
    out.append(f"variety_penalty: {scores.get('variety_penalty')}")
    if days:
        for i, d in enumerate(days[:7], start=1):
            out.append(f"Día {i}: {d.get('total_kcal')} kcal")
            meals = d.get('meals') if isinstance(d.get('meals'), list) else []
            for m in meals:
                slot = m.get('slot')
                items = m.get('items') if isinstance(m.get('items'), list) else []
                names = ", ".join([str(it.get('name') or '').strip() for it in items if str(it.get('name') or '').strip()])
                if names:
                    out.append(f"- {slot}: {names}")
    return "\n".join(out).strip()


def build_quick_actions_for_menu(*, variety_level: VarietyLevel, is_applied: bool = False) -> list[dict[str, Any]]:
    # UX wow: aplicar + lista + mutación local (máx 3 botones como el resto del chat)
    # Regla: si el menú ya está aplicado para la semana, NO mostrar el botón "Aplicar menú".
    actions: list[dict[str, Any]] = []
    if not bool(is_applied):
        actions.append({
            'label': 'Aplicar menú',
            'type': 'message',
            'text': 'Aplicar menú',
            'payload': {'meal_plan_apply': True},
        })
    actions.append({
        'label': 'Lista de compras',
        'type': 'message',
        'text': 'Lista de compras',
        'payload': {'meal_plan_view': 'shopping_list'},
    })

    actions.append({
        'label': 'Cambiar cena (mañana)',
        'type': 'message',
        'text': 'Cambiar cena (mañana)',
        'payload': {'meal_plan_mutate': {'day_index': 1, 'slot': 'cena', 'direction': 'high'}},
    })
    return actions[:3]


def mutate_plan_slot(
    *,
    result: dict[str, Any],
    day_index: int,
    slot: str,
    direction: Literal["simple", "normal", "high"],
    seed: int,
    locale: str,
    exclude_item_ids: list[str] | None,
) -> dict[str, Any]:
    """Mutación local: cambia solo 1 slot (ej. cena del día 2) sin regenerar toda la semana."""

    if not isinstance(result, dict):
        return result
    plan = result.get('plan') if isinstance(result.get('plan'), dict) else None
    if not plan:
        return result
    days = plan.get('days') if isinstance(plan.get('days'), list) else []
    if not days or not (0 <= int(day_index) < len(days)):
        return result
    slot = str(slot or '').strip().lower()
    if slot not in ('desayuno', 'almuerzo', 'cena', 'snack'):
        return result

    p = qaf_paths()
    items_meta = load_items_meta(p['items_meta'])
    nutrition_db = load_nutrition_db(p['nutrition_db'])
    micros_db = load_micros_db(p['micros_db'])

    exclude = {str(x).strip() for x in (exclude_item_ids or []) if str(x).strip()}
    pools = _classify_pools(nutrition_db, exclude)

    rng = random.Random(int(seed) & 0xFFFFFFFF)
    out = json.loads(json.dumps(result))

    day = out['plan']['days'][int(day_index)]
    meals = day.get('meals') if isinstance(day.get('meals'), list) else []
    target_meal = None
    for m in meals:
        if str(m.get('slot') or '').strip().lower() == slot:
            target_meal = m
            break
    if not target_meal:
        return out

    items = target_meal.get('items') if isinstance(target_meal.get('items'), list) else []
    if not items:
        return out

    # Elegimos qué item mutar: en high priorizamos los repetidos; en simple priorizamos vegetales.
    ids_all = []
    for d in out['plan']['days']:
        for m in d.get('meals') or []:
            for it in m.get('items') or []:
                iid = str(it.get('item_id') or '')
                if iid:
                    ids_all.append(iid)
    counts = Counter(ids_all)

    idx = 0
    if direction == 'high':
        idx = max(range(len(items)), key=lambda i: counts.get(str(items[i].get('item_id') or ''), 0))
    elif direction == 'simple':
        idx = min(range(len(items)), key=lambda i: float(items[i].get('kcal') or 0.0))
    else:
        idx = rng.randrange(0, len(items))

    iid_old = str(items[idx].get('item_id') or '')
    kcal_target = float(items[idx].get('kcal') or 0.0)

    pool_key = 'veg'
    if iid_old in pools.get('protein', []):
        pool_key = 'protein'
    elif iid_old in pools.get('carb', []):
        pool_key = 'carb'
    elif iid_old in pools.get('fat', []):
        pool_key = 'fat'
    elif iid_old in pools.get('fruit', []):
        pool_key = 'fruit'

    candidates = pools.get(pool_key) or pools['carb']
    # En high variedad, evitamos el mismo iid si hay alternativas.
    if direction == 'high':
        candidates = [c for c in candidates if c != iid_old] or candidates

    iid_new = rng.choice(candidates)
    grams = _grams_for_kcal(iid_new, kcal_target, nutrition_db)
    grams = round(grams / 5.0) * 5.0
    kcal_new = _kcal_for_grams(iid_new, grams, nutrition_db)

    items[idx] = {
        'item_id': iid_new,
        'name': _display_name(iid_new, items_meta, locale),
        'grams': float(grams),
        'kcal': round(float(kcal_new), 1),
    }
    target_meal['items'] = items
    target_meal['kcal'] = round(float(sum(float(it.get('kcal') or 0.0) for it in items)), 1)
    day['total_kcal'] = round(float(sum(float(m.get('kcal') or 0.0) for m in meals)), 1)

    # Recalcular shopping + scores (rápido)
    shop_g = Counter()
    for d in out['plan']['days']:
        for m in d.get('meals') or []:
            for it in m.get('items') or []:
                iid = str(it.get('item_id') or '')
                g = float(it.get('grams') or 0.0)
                if iid and g > 0:
                    shop_g[iid] += g
    out['shopping_list'] = [
        {'item_id': iid, 'name': _display_name(iid, items_meta, locale), 'grams': round(float(g), 1)}
        for iid, g in shop_g.most_common()
    ]

    kcal_day_target = float(((out.get('inputs') or {}).get('kcal_day')) or 2000.0)
    variety_level = str(((out.get('inputs') or {}).get('variety')) or 'normal')
    if variety_level not in ('simple', 'normal', 'high'):
        variety_level = 'normal'
    s = _score_plan(out['plan'], kcal_day_target=kcal_day_target, micros_db=micros_db, variety_level=variety_level)  # type: ignore[arg-type]
    out['scores'] = {
        'kcal_loss': round(float(s.kcal_loss), 4),
        'micro_coverage': round(float(s.micro_coverage), 4),
        'variety_penalty': round(float(s.variety_penalty), 4),
        'friction_penalty': round(float(s.friction_penalty), 4),
        'total': round(float(s.total), 4),
    }

    # Marcar meta
    out.setdefault('meta', {})
    try:
        out['meta']['mutated'] = {'day_index': int(day_index), 'slot': slot, 'direction': direction, 'as_of': str(date.today())}
    except Exception:
        pass
    return out
