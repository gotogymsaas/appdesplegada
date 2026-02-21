from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qaf_body_trend import evaluate_body_trend


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    args = p.parse_args()
    ds = Path(args.dataset)

    n = 0
    accepted = 0
    needs = 0
    for line in ds.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        row: dict[str, Any] = json.loads(line)
        prof = row.get('profile') or {}
        obs = row.get('observations') or {}
        r = evaluate_body_trend(prof, obs, horizon_weeks=6)
        n += 1
        if r.payload.get('decision') == 'accepted':
            accepted += 1
        else:
            needs += 1

    print(f"examples={n}")
    print(f"accepted={accepted}/{n}")
    print(f"needs_confirmation={needs}/{n}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
