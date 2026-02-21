from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qaf_meal_planner import generate_week_plan


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()
    ds = Path(args.dataset)

    n = 0
    ok_days = 0
    for line in ds.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row: dict[str, Any] = json.loads(line)
        profile = row.get("profile") or {}
        constraints = row.get("constraints") or {}
        kcal_day = float(profile.get("kcal_day") or 2000)
        meals_per_day = int(profile.get("meals_per_day") or 3)
        variety = str(profile.get("variety") or "normal").strip().lower()
        exclude = constraints.get("exclude") if isinstance(constraints, dict) else []

        res = generate_week_plan(
            kcal_day=kcal_day,
            meals_per_day=meals_per_day,
            variety_level=variety if variety in ("simple", "normal", "high") else "normal",
            exclude=exclude if isinstance(exclude, list) else [],
            seed=42 + n,
        )
        n += 1
        days = ((res.get("plan") or {}).get("days") or [])
        if len(days) == 7:
            ok_days += 1

    print(f"examples={n}")
    print(f"days_ok={ok_days}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
