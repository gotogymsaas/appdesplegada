from __future__ import annotations

import argparse
import json
from pathlib import Path

from qaf_posture_corrective import evaluate_posture


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")

    n = 0
    accepted = 0
    needs = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        row = json.loads(raw)
        payload = row.get("input") if isinstance(row.get("input"), dict) else row
        res = evaluate_posture(payload).payload
        n += 1
        if res.get("decision") == "accepted":
            accepted += 1
        else:
            needs += 1
    print(
        json.dumps(
            {
                "n": n,
                "accepted": accepted,
                "needs_confirmation": needs,
                "accepted_rate": (accepted / n) if n else 0.0,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
