from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


ScenarioKey = Literal["baseline", "follow_plan", "minus_200", "plus_200"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _band_width_kg_per_week(uncertainty_score: float) -> float:
    # banda operacional: 0.15kg..0.65kg/semana
    u = _clamp01(uncertainty_score)
    return 0.15 + (0.50 * u)


def _weekly_delta_kg(kcal_in_avg_day: float, tdee_kcal_day: float) -> float:
    return ((float(kcal_in_avg_day) - float(tdee_kcal_day)) * 7.0) / 7700.0


@dataclass(frozen=True)
class TrendResult:
    payload: dict[str, Any]


def evaluate_body_trend(
    profile: dict[str, Any],
    observations: dict[str, Any],
    *,
    horizon_weeks: int = 6,
) -> TrendResult:
    # Inputs
    try:
        tdee = float(profile.get("tdee_kcal_day"))
    except Exception:
        tdee = 0.0
    try:
        reco = profile.get("recommended_kcal_day")
        recommended = float(reco) if reco is not None else None
    except Exception:
        recommended = None

    try:
        w0 = float(observations.get("weight_current_week_avg_kg"))
    except Exception:
        w0 = 0.0
    try:
        wprev = observations.get("weight_previous_week_avg_kg")
        wprev = float(wprev) if wprev is not None else None
    except Exception:
        wprev = None

    kcal_in = observations.get("kcal_in_avg_day")
    try:
        kcal_in_avg = float(kcal_in) if kcal_in is not None else None
    except Exception:
        kcal_in_avg = None

    missing: list[str] = []
    if tdee <= 0:
        missing.append("tdee_kcal_day")
    if w0 <= 0:
        missing.append("weight_current_week_avg_kg")

    # Confidence heurística
    conf = 1.0
    if missing:
        conf *= 0.55
    if kcal_in_avg is None:
        conf *= 0.65
    if wprev is None:
        conf *= 0.85
    confidence = _clamp01(conf)
    uncertainty = _clamp01(1.0 - confidence)

    follow_up_questions: list[dict[str, Any]] = []
    decision = "accepted"
    decision_reason = "ok"
    if kcal_in_avg is None:
        decision = "needs_confirmation"
        decision_reason = "missing_kcal_in_avg"
        follow_up_questions.append(
            {
                "type": "confirm_kcal_in_avg_day",
                "prompt": "¿Cuál fue tu ingesta promedio diaria esta semana (kcal)?",
                "options": [],
            }
        )

    def sim(kcal_in_day: float) -> list[dict[str, Any]]:
        traj: list[dict[str, Any]] = []
        w = float(w0)
        bw = _band_width_kg_per_week(uncertainty)
        for wk in range(1, int(horizon_weeks) + 1):
            dw = _weekly_delta_kg(kcal_in_day, tdee)
            w = w + dw
            band = bw * math.sqrt(wk)
            traj.append(
                {
                    "week": wk,
                    "weight_kg": round(float(w), 3),
                    "weight_kg_min": round(float(w - band), 3),
                    "weight_kg_max": round(float(w + band), 3),
                    "delta_week_kg": round(float(dw), 4),
                }
            )
        return traj

    scenarios: dict[str, Any] = {}
    if tdee > 0 and w0 > 0:
        # baseline
        if kcal_in_avg is not None:
            scenarios["baseline"] = {
                "kcal_in_avg_day": float(kcal_in_avg),
                "trajectory": sim(float(kcal_in_avg)),
            }
            scenarios["minus_200"] = {
                "kcal_in_avg_day": float(kcal_in_avg) - 200.0,
                "trajectory": sim(float(kcal_in_avg) - 200.0),
            }
            scenarios["plus_200"] = {
                "kcal_in_avg_day": float(kcal_in_avg) + 200.0,
                "trajectory": sim(float(kcal_in_avg) + 200.0),
            }
        # follow plan
        if recommended is not None and recommended > 0:
            scenarios["follow_plan"] = {
                "kcal_in_avg_day": float(recommended),
                "trajectory": sim(float(recommended)),
            }

    explainability: list[str] = []
    if tdee > 0:
        explainability.append(f"TDEE estimado: {round(float(tdee), 0):.0f} kcal/día")
    if recommended is not None:
        explainability.append(f"Recomendación actual: {round(float(recommended), 0):.0f} kcal/día")
    explainability = explainability[:2]

    payload = {
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": {
            "score": round(float(confidence), 4),
            "uncertainty_score": round(float(uncertainty), 4),
            "missing": missing,
        },
        "inputs": {
            "horizon_weeks": int(horizon_weeks),
            "tdee_kcal_day": float(tdee) if tdee > 0 else None,
            "recommended_kcal_day": float(recommended) if recommended is not None else None,
            "kcal_in_avg_day": float(kcal_in_avg) if kcal_in_avg is not None else None,
            "weight_current_week_avg_kg": float(w0) if w0 > 0 else None,
            "weight_previous_week_avg_kg": float(wprev) if wprev is not None else None,
        },
        "scenarios": scenarios,
        "follow_up_questions": follow_up_questions,
        "explainability": explainability,
        "meta": {
            "algorithm": "exp-005_body_trend_v0",
            "as_of": str(date.today()),
        },
    }
    return TrendResult(payload=payload)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    conf = result.get('confidence') if isinstance(result.get('confidence'), dict) else {}
    scenarios = result.get('scenarios') if isinstance(result.get('scenarios'), dict) else {}
    lines: list[str] = []
    lines.append(f"decision: {result.get('decision')}")
    if conf.get('score') is not None:
        try:
            lines.append(f"confidence: {round(float(conf.get('score')), 3)}")
        except Exception:
            pass
    # Mostrar baseline o follow_plan
    main = scenarios.get('baseline') or scenarios.get('follow_plan')
    if isinstance(main, dict):
        traj = main.get('trajectory') if isinstance(main.get('trajectory'), list) else []
        if traj:
            last = traj[-1] if isinstance(traj[-1], dict) else None
            if last:
                lines.append(f"weight_in_{len(traj)}w: {last.get('weight_kg')} kg")
                lines.append(f"range_in_{len(traj)}w: {last.get('weight_kg_min')}–{last.get('weight_kg_max')} kg")
    explain = result.get('explainability')
    if isinstance(explain, list):
        for x in explain[:2]:
            if str(x).strip():
                lines.append(f"note: {str(x).strip()}")
    return "\n".join(lines).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True)
    args = p.parse_args()
    raw = args.input_json
    payload = json.loads(open(raw, 'r', encoding='utf-8').read()) if raw.endswith('.json') else json.loads(raw)
    profile = payload.get('profile') if isinstance(payload, dict) else {}
    obs = payload.get('observations') if isinstance(payload, dict) else {}
    r = evaluate_body_trend(profile, obs, horizon_weeks=int(payload.get('horizon_weeks') or 6))
    out = dict(r.payload)
    out['text'] = render_professional_summary(r.payload)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
