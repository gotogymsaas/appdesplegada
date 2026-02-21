from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qaf_goal_coherence import evaluate_meal


def _pct(n: int, d: int) -> float:
    if not d:
        return 0.0
    return (float(n) / float(d)) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    processed = 0
    accepted = 0
    partial = 0
    needs_confirmation = 0
    with_alerts = 0

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, Any] = json.loads(line)
            user_context = row.get("user_context") or {}
            meal = row.get("meal") or {}
            result = evaluate_meal(user_context, meal)

            processed += 1
            decision = str(result.get("decision") or "")
            if decision == "accepted":
                accepted += 1
            elif decision == "partial":
                partial += 1
            else:
                needs_confirmation += 1

            alerts = result.get("alerts") or []
            if isinstance(alerts, list) and alerts:
                with_alerts += 1

    print(f"examples={processed}")
    print(f"accepted={accepted}/{processed} ({_pct(accepted, processed):.1f}%)")
    print(f"partial={partial}/{processed} ({_pct(partial, processed):.1f}%)")
    print(f"needs_confirmation={needs_confirmation}/{processed} ({_pct(needs_confirmation, processed):.1f}%)")
    print(f"with_alerts={with_alerts}/{processed} ({_pct(with_alerts, processed):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
