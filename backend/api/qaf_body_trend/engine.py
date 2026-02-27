from __future__ import annotations

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

    conf = 1.0
    if missing:
        conf *= 0.55
    if kcal_in_avg is None:
        conf *= 0.65
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
        if kcal_in_avg is not None:
            scenarios["baseline"] = {"kcal_in_avg_day": float(kcal_in_avg), "trajectory": sim(float(kcal_in_avg))}
            scenarios["minus_200"] = {"kcal_in_avg_day": float(kcal_in_avg) - 200.0, "trajectory": sim(float(kcal_in_avg) - 200.0)}
            scenarios["plus_200"] = {"kcal_in_avg_day": float(kcal_in_avg) + 200.0, "trajectory": sim(float(kcal_in_avg) + 200.0)}
        if recommended is not None and recommended > 0:
            scenarios["follow_plan"] = {"kcal_in_avg_day": float(recommended), "trajectory": sim(float(recommended))}

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
        },
        "scenarios": scenarios,
        "follow_up_questions": follow_up_questions,
        "explainability": explainability,
        "meta": {"algorithm": "exp-005_body_trend_v0", "as_of": str(date.today())},
    }
    return TrendResult(payload=payload)


def render_professional_summary(result: dict[str, Any], *, preferred_scenario: str | None = None) -> str:
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
    # Resumen WOW: 6 semanas para 2-3 escenarios (si existen)
    def _last(traj: Any):
        if not isinstance(traj, list) or not traj:
            return None
        return traj[-1] if isinstance(traj[-1], dict) else None

    def _fmt_last(label: str, sc: Any):
        if not isinstance(sc, dict):
            return
        last = _last(sc.get('trajectory'))
        if not last:
            return
        lines.append(
            f"{label}: {last.get('weight_kg')} kg (rango {last.get('weight_kg_min')}–{last.get('weight_kg_max')})"
        )

    if scenarios:
        pref = str(preferred_scenario or '').strip().lower() or None
        order: list[str] = ["baseline", "follow_plan", "minus_200", "plus_200"]
        if pref in order:
            order = [pref] + [x for x in order if x != pref]
            lines.append(f"escenario: {pref}")

        for k in order:
            _fmt_last(k, scenarios.get(k))
    explain = result.get('explainability')
    if isinstance(explain, list):
        for x in explain[:2]:
            if str(x).strip():
                lines.append(f"note: {str(x).strip()}")
    return "\n".join(lines).strip()


def build_quick_actions_for_trend(
    *,
    has_intake: bool,
    allow_simulations: bool = True,
    excluded_scenarios: list[str] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    excluded = {str(x or '').strip().lower() for x in (excluded_scenarios or [])}
    if not has_intake:
        # WOW: 1 tap para registrar ingesta promedio (sin UI extra)
        for kcal in (1800, 2000, 2200):
            actions.append({
                'label': f'Registrar {kcal} kcal/día',
                'type': 'message',
                'text': f'Registrar {kcal} kcal/día',
                'payload': {'body_trend_request': {'kcal_in_avg_day': float(kcal)}},
            })
    else:
        if allow_simulations:
            if 'follow_plan' not in excluded:
                actions.append({
                    'label': 'Simular recomendación',
                    'type': 'message',
                    'text': 'Simular recomendación',
                    'payload': {'body_trend_request': {'scenario': 'follow_plan'}},
                })
            if 'minus_200' not in excluded:
                actions.append({
                    'label': 'Simular -200 kcal',
                    'type': 'message',
                    'text': 'Simular -200 kcal',
                    'payload': {'body_trend_request': {'scenario': 'minus_200'}},
                })
            if 'plus_200' not in excluded:
                actions.append({
                    'label': 'Simular +200 kcal',
                    'type': 'message',
                    'text': 'Simular +200 kcal',
                    'payload': {'body_trend_request': {'scenario': 'plus_200'}},
                })

    actions.append({'label': 'Finalizar', 'type': 'services_menu', 'page': 'core'})
    return actions[:6]
