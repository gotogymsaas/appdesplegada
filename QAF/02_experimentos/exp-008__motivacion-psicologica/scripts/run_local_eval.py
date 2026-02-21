from __future__ import annotations

import argparse
import json
from pathlib import Path

from qaf_motivation_psych import evaluate_motivation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")

    n = 0
    needs = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        row = json.loads(raw)
        payload = row.get("input") if isinstance(row.get("input"), dict) else row
        res = evaluate_motivation(payload).payload
        n += 1
        if res.get("decision") == "needs_confirmation":
            needs += 1

    print(json.dumps({"n": n, "needs_confirmation": needs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
