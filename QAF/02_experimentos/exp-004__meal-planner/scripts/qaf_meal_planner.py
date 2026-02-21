from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


VarietyLevel = Literal["simple", "normal", "high"]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "backend").exists() and (p / "QAF").exists():
            return p
    return Path.cwd()


def _catalog_paths() -> dict[str, Path]:
    root = _repo_root()
    base = root / "backend" / "api" / "qaf_calories" / "data"
    return {
        "aliases": base / "aliases.csv",
        "items_meta": base / "items_meta.csv",
        "nutrition_db": base / "nutrition_db.csv",
        "micros_db": base / "micros_db.csv",
    }


def _load_csv_dicts(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def load_items_meta(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _load_csv_dicts(path):
        iid = (row.get("item_id") or "").strip()
        if not iid:
            continue
        out[iid] = row
    return out


def load_nutrition_db(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in _load_csv_dicts(path):
        iid = (row.get("item_id") or "").strip()
        if not iid:
            continue
        def f(key: str) -> float | None:
            raw = (row.get(key) or "").strip()
            if not raw:
                return None
            try:
                return float(raw)
            except Exception:
                return None
        out[iid] = {
            "kcal_per_100g": float(f("kcal_per_100g") or 0.0),
            "protein": float(f("protein_g_per_100g") or 0.0),
            "carbs": float(f("carbs_g_per_100g") or 0.0),
            "fat": float(f("fat_g_per_100g") or 0.0),
            "fiber": float(f("fiber_g_per_100g") or 0.0),
        }
    return out


def load_micros_db(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _load_csv_dicts(path):
        iid = (row.get("item_id") or "").strip()
        micro = (row.get("micro") or "").strip()
        level = (row.get("level") or "").strip().lower()
        if not iid or not micro:
            continue
        out[iid].append({"micro": micro, "level": level})
    return dict(out)


def display_name(iid: str, meta: dict[str, dict[str, str]] | None) -> str:
    row = (meta or {}).get(iid) or {}
    return (row.get("display_name_es") or row.get("display_name_en") or iid).strip() or iid


def _grams_for_kcal(iid: str, kcal: float, nutrition_db: dict[str, dict[str, float]]) -> float:
    kcal100 = float((nutrition_db.get(iid) or {}).get("kcal_per_100g") or 0.0)
    if kcal100 <= 0:
        return 0.0
    return max(0.0, (float(kcal) / kcal100) * 100.0)


def _kcal_for_grams(iid: str, grams: float, nutrition_db: dict[str, dict[str, float]]) -> float:
    kcal100 = float((nutrition_db.get(iid) or {}).get("kcal_per_100g") or 0.0)
    if kcal100 <= 0:
        return 0.0
    return (float(grams) / 100.0) * kcal100


def _classify_pools(nutrition_db: dict[str, dict[str, float]], exclude: set[str]) -> dict[str, list[str]]:
    pools = {"protein": [], "carb": [], "veg": [], "fat": [], "fruit": []}
    for iid, n in nutrition_db.items():
        if iid in exclude:
            continue
        kcal = float(n.get("kcal_per_100g") or 0.0)
        if kcal <= 0:
            continue
        p = float(n.get("protein") or 0.0)
        c = float(n.get("carbs") or 0.0)
        f = float(n.get("fat") or 0.0)
        fb = float(n.get("fiber") or 0.0)

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

    # fallback: si falta algo, degradar a cualquier item
    all_items = [iid for iid, n in nutrition_db.items() if iid not in exclude and float((n.get("kcal_per_100g") or 0.0)) > 0]
    for k in list(pools.keys()):
        if not pools[k]:
            pools[k] = all_items[:]
    return pools


@dataclass
class PlanScore:
    kcal_loss: float
    micro_coverage: float
    variety_penalty: float
    friction_penalty: float
    total: float


def _micro_coverage(plan: dict[str, Any], micros_db: dict[str, list[dict[str, str]]]) -> float:
    found: set[str] = set()
    for day in plan.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                for m in micros_db.get(iid) or []:
                    lvl = str(m.get("level") or "").lower()
                    if lvl in ("alto", "medio"):
                        found.add(str(m.get("micro") or "").strip())
    # Normalizamos a un techo simple para MVP
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
    # proxy: demasiados items distintos = más fricción
    ids: set[str] = set()
    for day in plan.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                if iid:
                    ids.add(iid)
    # 18 items únicos/semana es un techo MVP
    return min(1.0, max(0.0, (len(ids) - 18) / 18.0))


def score_plan(
    plan: dict[str, Any],
    *,
    kcal_day_target: float,
    micros_db: dict[str, list[dict[str, str]]],
    variety_level: VarietyLevel,
) -> PlanScore:
    # kcal loss
    kcal_err2 = 0.0
    for day in plan.get("days") or []:
        kcal = float(day.get("total_kcal") or 0.0)
        e = (kcal - float(kcal_day_target)) / max(1.0, float(kcal_day_target))
        kcal_err2 += e * e
    kcal_loss = kcal_err2 / max(1, len(plan.get("days") or []))

    micro = _micro_coverage(plan, micros_db)
    var_pen = _variety_penalty(plan)
    fr_pen = _friction_penalty(plan)

    # pesos por nivel de variedad
    if variety_level == "high":
        wv, wf = 0.9, 0.4
    elif variety_level == "simple":
        wv, wf = 0.3, 0.8
    else:
        wv, wf = 0.6, 0.6

    total = (1.2 * kcal_loss) + (0.9 * (1.0 - micro)) + (wv * var_pen) + (wf * fr_pen)
    return PlanScore(kcal_loss=kcal_loss, micro_coverage=micro, variety_penalty=var_pen, friction_penalty=fr_pen, total=total)


def _build_day(
    *,
    rng: random.Random,
    nutrition_db: dict[str, dict[str, float]],
    items_meta: dict[str, dict[str, str]],
    pools: dict[str, list[str]],
    kcal_day: float,
    meals_per_day: int,
) -> dict[str, Any]:
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

        # Composición simple por slot
        if slot == "snack":
            chosen = [rng.choice(pools["fruit"]), rng.choice(pools["fat"])]
            split = [0.65, 0.35]
        else:
            chosen = [rng.choice(pools["protein"]), rng.choice(pools["carb"]), rng.choice(pools["veg"])]
            split = [0.40, 0.40, 0.20]

        items = []
        kcal_meal = 0.0
        for iid, frac in zip(chosen, split):
            g = _grams_for_kcal(iid, target * frac, nutrition_db)
            g = round(g / 5.0) * 5.0
            k = _kcal_for_grams(iid, g, nutrition_db)
            kcal_meal += k
            items.append(
                {
                    "item_id": iid,
                    "name": display_name(iid, items_meta),
                    "grams": float(g),
                    "kcal": round(float(k), 1),
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
    exclude: list[str] | None,
    seed: int,
) -> dict[str, Any]:
    cat = _catalog_paths()
    items_meta = load_items_meta(cat["items_meta"])
    nutrition_db = load_nutrition_db(cat["nutrition_db"])
    micros_db = load_micros_db(cat["micros_db"])

    exclude_set = {str(x).strip() for x in (exclude or []) if str(x).strip()}
    pools = _classify_pools(nutrition_db, exclude_set)

    rng = random.Random(int(seed) & 0xFFFFFFFF)

    # Evolución corta
    pop_n = 24
    gens = 18
    elite = 6

    def new_candidate() -> dict[str, Any]:
        days = []
        for _ in range(7):
            d = _build_day(
                rng=rng,
                nutrition_db=nutrition_db,
                items_meta=items_meta,
                pools=pools,
                kcal_day=float(kcal_day),
                meals_per_day=int(meals_per_day),
            )
            days.append(d)
        plan = {"days": days}
        return plan

    def mutate(plan: dict[str, Any]) -> dict[str, Any]:
        out = json.loads(json.dumps(plan))
        if not out.get("days"):
            return out
        d = rng.choice(out["days"])
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
        # conservar kcal del item en la mutación
        kcal_target = float(items[idx].get("kcal") or 0.0)
        grams = _grams_for_kcal(iid_new, kcal_target, nutrition_db)
        grams = round(grams / 5.0) * 5.0
        kcal_new = _kcal_for_grams(iid_new, grams, nutrition_db)

        items[idx] = {
            "item_id": iid_new,
            "name": display_name(iid_new, items_meta),
            "grams": float(grams),
            "kcal": round(float(kcal_new), 1),
        }

        # Recalcular kcal meal/day
        kcal_meal = sum(float(it.get("kcal") or 0.0) for it in items)
        meal["kcal"] = round(float(kcal_meal), 1)
        d["total_kcal"] = round(float(sum(float(m.get("kcal") or 0.0) for m in meals)), 1)
        return out

    population = [new_candidate() for _ in range(pop_n)]

    best = None
    best_s = None
    for _ in range(gens):
        scored = [(p, score_plan(p, kcal_day_target=float(kcal_day), micros_db=micros_db, variety_level=variety_level)) for p in population]
        scored.sort(key=lambda x: x[1].total)
        if best is None or scored[0][1].total < (best_s.total if best_s else 1e9):
            best, best_s = scored[0]

        next_pop = [scored[i][0] for i in range(min(elite, len(scored)))]
        while len(next_pop) < pop_n:
            parent = rng.choice(next_pop)
            child = mutate(parent)
            next_pop.append(child)
        population = next_pop

    assert best is not None and best_s is not None

    # Shopping list
    shop_g = Counter()
    for day in best.get("days") or []:
        for meal in day.get("meals") or []:
            for it in meal.get("items") or []:
                iid = str(it.get("item_id") or "")
                g = float(it.get("grams") or 0.0)
                if iid and g > 0:
                    shop_g[iid] += g
    shopping = [
        {"item_id": iid, "name": display_name(iid, items_meta), "grams": round(float(g), 1)}
        for iid, g in shop_g.most_common()
    ]

    return {
        "plan": best,
        "scores": {
            "kcal_loss": round(float(best_s.kcal_loss), 4),
            "micro_coverage": round(float(best_s.micro_coverage), 4),
            "variety_penalty": round(float(best_s.variety_penalty), 4),
            "friction_penalty": round(float(best_s.friction_penalty), 4),
            "total": round(float(best_s.total), 4),
        },
        "shopping_list": shopping,
    }


def render_menu_text(result: dict[str, Any]) -> str:
    plan = (result or {}).get("plan") or {}
    days = plan.get("days") or []
    if not days:
        return ""
    out = []
    out.append("Menú semanal (resumen):")
    for i, d in enumerate(days[:7], start=1):
        kcal = d.get("total_kcal")
        out.append(f"Día {i} — {kcal} kcal")
        for meal in d.get("meals") or []:
            slot = str(meal.get("slot") or "")
            items = meal.get("items") or []
            names = ", ".join([str(it.get("name") or "") for it in items if str(it.get("name") or "").strip()])
            if names:
                out.append(f"- {slot}: {names}")
    return "\n".join(out).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    args = p.parse_args()
    raw = args.input_json
    payload = json.loads(Path(raw).read_text(encoding="utf-8")) if raw.endswith(".json") else json.loads(raw)

    profile = payload.get("profile") if isinstance(payload, dict) else {}
    constraints = payload.get("constraints") if isinstance(payload, dict) else {}

    kcal_day = float(profile.get("kcal_day") or 2000)
    meals_per_day = int(profile.get("meals_per_day") or 3)
    variety = str(profile.get("variety") or "normal").strip().lower()
    variety_level: VarietyLevel = variety if variety in ("simple", "normal", "high") else "normal"
    exclude = constraints.get("exclude") if isinstance(constraints, dict) else []

    seed = int(profile.get("seed") or 1337)
    res = generate_week_plan(
        kcal_day=kcal_day,
        meals_per_day=meals_per_day,
        variety_level=variety_level,
        exclude=exclude if isinstance(exclude, list) else [],
        seed=seed,
    )
    res["text"] = render_menu_text(res)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
