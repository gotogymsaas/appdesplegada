from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qaf_vision_calories import load_calorie_db, qaf_infer_from_vision


def _mae(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(abs(v) for v in values) / len(values)


def _pct(n: int, d: int) -> float:
    if not d:
        return 0.0
    return (float(n) / float(d)) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--calorie-db",
        default=str(Path(__file__).resolve().parents[1] / "data" / "calorie_db.csv"),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    calorie_db = load_calorie_db(Path(args.calorie_db))

    abs_errors = []
    range_hits = 0
    with_gt = 0
    confirm_count = 0
    missing_any = 0
    covered = 0
    processed = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            vision = row.get("vision") or {}
            user_id = row.get("user_id") if isinstance(row.get("user_id"), str) else None
            locale = row.get("locale") if isinstance(row.get("locale"), str) else None
            result = qaf_infer_from_vision(vision, calorie_db, user_id=user_id, locale=locale or "es-CO")

            processed += 1

            if float(result.get("total_calories") or 0.0) > 0:
                covered += 1

            if bool(result.get("needs_confirmation")):
                confirm_count += 1

            if result.get("missing_items"):
                missing_any += 1

            gt = row.get("ground_truth")
            if isinstance(gt, dict) and gt.get("total_calories") is not None:
                try:
                    gt_total = float(gt.get("total_calories"))
                    pred_total = float(result.get("total_calories") or 0.0)
                    abs_errors.append(pred_total - gt_total)

                    rng = result.get("total_calories_range") or {}
                    low = float(rng.get("low") or 0.0)
                    high = float(rng.get("high") or 0.0)
                    if low <= gt_total <= high:
                        range_hits += 1
                    with_gt += 1
                except Exception:
                    pass

    print(f"examples={processed}")
    print(f"coverage_total_calories={covered}/{processed} ({_pct(covered, processed):.1f}%)")
    print(f"needs_confirmation={confirm_count}/{processed} ({_pct(confirm_count, processed):.1f}%)")
    print(f"missing_items_any={missing_any}/{processed} ({_pct(missing_any, processed):.1f}%)")
    if abs_errors:
        mae = _mae(abs_errors)
        print(f"mae_total_calories={mae:.2f}")
        print(f"range_coverage={range_hits}/{with_gt} ({_pct(range_hits, with_gt):.1f}%)")
    else:
        print("mae_total_calories=NA (no ground_truth en dataset)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
