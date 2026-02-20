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
    processed = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            vision = row.get("vision") or {}
            result = qaf_infer_from_vision(vision, calorie_db)

            processed += 1
            gt = row.get("ground_truth")
            if isinstance(gt, dict) and gt.get("total_calories") is not None:
                try:
                    gt_total = float(gt.get("total_calories"))
                    pred_total = float(result.get("total_calories") or 0.0)
                    abs_errors.append(pred_total - gt_total)
                except Exception:
                    pass

    print(f"examples={processed}")
    if abs_errors:
        mae = _mae(abs_errors)
        print(f"mae_total_calories={mae:.2f}")
    else:
        print("mae_total_calories=NA (no ground_truth en dataset)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
