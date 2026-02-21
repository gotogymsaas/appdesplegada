from __future__ import annotations

import argparse
import json
from pathlib import Path

from qaf_progression_intelligent import evaluate_progression


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    path = Path(args.dataset)
    rows = 0
    needs = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        payload = obj.get('input') if isinstance(obj.get('input'), dict) else obj
        res = evaluate_progression(payload).payload
        rows += 1
        if res.get('decision') == 'needs_confirmation':
            needs += 1
    print(json.dumps({'n': rows, 'needs_confirmation': needs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
