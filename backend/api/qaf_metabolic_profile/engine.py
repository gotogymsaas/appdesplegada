from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


Sex = Literal["male", "female"]
GoalType = Literal["deficit", "maintenance", "gain"]
ActivityLevel = Literal["low", "moderate", "high"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    dev = [abs(x - med) for x in values]
    return float(statistics.median(dev))


def robust_center(values: list[float]) -> tuple[float | None, dict[str, Any]]:
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    if len(vals) < 2:
        return (float(vals[0]) if vals else None), {"n": len(vals), "n_used": len(vals), "outliers_removed": 0}

    med = statistics.median(vals)
    mad = _mad(vals)
    if mad <= 1e-9:
        return float(med), {"n": len(vals), "n_used": len(vals), "outliers_removed": 0}

    thr = 3.5 * mad
    kept = [x for x in vals if abs(x - med) <= thr]
    if not kept:
        kept = vals
    center = float(statistics.median(kept))
    removed = max(0, len(vals) - len(kept))
    return center, {"n": len(vals), "n_used": len(kept), "outliers_removed": removed}


def mifflin_st_jeor(*, sex: Sex, weight_kg: float, height_cm: float, age: int) -> float:
    s = 5.0 if sex == "male" else -161.0
    return (10.0 * float(weight_kg)) + (6.25 * float(height_cm)) - (5.0 * float(age)) + s


def activity_factor(level: ActivityLevel) -> float:
    return {"low": 1.35, "moderate": 1.55, "high": 1.75}.get(level, 1.55)


def goal_delta_kcal(goal: GoalType) -> float:
    if goal == "deficit":
        return -400.0
    if goal == "gain":
        return 250.0
    return 0.0


def target_weekly_weight_change_pct(goal: GoalType) -> float:
    if goal == "deficit":
        return -0.5
    if goal == "gain":
        return 0.25
    return 0.0


@dataclass(frozen=True)
class MetabolicResult:
    payload: dict[str, Any]


def evaluate_weekly_metabolic_profile(
    profile: dict[str, Any],
    weights: dict[str, Any],
    *,
    last_alpha: float | None = None,
    max_weekly_kcal_adjust: float = 200.0,
) -> MetabolicResult:
    sex_raw = str(profile.get("sex") or "").strip().lower()
    sex: Sex | None = sex_raw if sex_raw in {"male", "female"} else None

    goal_raw = str(profile.get("goal_type") or "").strip().lower()
    goal: GoalType = goal_raw if goal_raw in {"deficit", "maintenance", "gain"} else "maintenance"

    act_raw = str(profile.get("activity_level") or "").strip().lower()
    activity: ActivityLevel = act_raw if act_raw in {"low", "moderate", "high"} else "moderate"

    try:
        age = int(profile.get("age"))
    except Exception:
        age = 0
    try:
        height_cm = float(profile.get("height_cm"))
    except Exception:
        height_cm = 0.0
    try:
        weight_kg = float(profile.get("weight_kg"))
    except Exception:
        weight_kg = 0.0

    try:
        override = profile.get("daily_target_kcal_override")
        override_kcal = float(override) if override is not None else None
    except Exception:
        override_kcal = None
    if override_kcal is not None and override_kcal <= 0:
        override_kcal = None

    missing: list[str] = []
    if sex is None:
        missing.append("sex")
    if age <= 0:
        missing.append("age")
    if height_cm <= 0:
        missing.append("height_cm")
    if weight_kg <= 0:
        missing.append("weight_kg")

    def _safe_list(key: str) -> list[float]:
        raw = weights.get(key)
        if not isinstance(raw, list):
            return []
        out: list[float] = []
        for v in raw:
            try:
                f = float(v)
            except Exception:
                continue
            if f > 0:
                out.append(f)
        return out

    cur_w = _safe_list("current_week")
    prev_w = _safe_list("previous_week")
    cur_center, cur_meta = robust_center(cur_w)
    prev_center, prev_meta = robust_center(prev_w)

    used_profile_weight = False
    if cur_center is None and weight_kg > 0:
        cur_center = float(weight_kg)
        used_profile_weight = True

    tmb = None
    tdee_base = None
    if not missing:
        tmb = float(mifflin_st_jeor(sex=sex, weight_kg=float(cur_center or weight_kg), height_cm=height_cm, age=age))
        tdee_base = float(tmb * activity_factor(activity))

    delta_kg_obs = None
    if cur_center is not None and prev_center is not None:
        delta_kg_obs = float(cur_center - prev_center)

    target_pct = target_weekly_weight_change_pct(goal)
    delta_kg_target = None
    if cur_center is not None:
        delta_kg_target = float(cur_center) * (float(target_pct) / 100.0)

    data_points = int(cur_meta.get("n_used") or 0) + int(prev_meta.get("n_used") or 0)
    outliers = int(cur_meta.get("outliers_removed") or 0) + int(prev_meta.get("outliers_removed") or 0)

    conf = 1.0
    if missing:
        conf *= 0.55
    if data_points < 4:
        conf *= 0.7
    if used_profile_weight:
        conf *= 0.75
    if outliers >= 2:
        conf *= 0.8
    confidence = _clamp01(conf)
    uncertainty = _clamp01(1.0 - confidence)

    alpha_prev = _clamp(float(last_alpha or 0.0), 0.0, 0.25)
    base_reco = None
    if override_kcal is not None:
        base_reco = float(override_kcal)
        reco_method = "override"
    elif tdee_base is not None:
        base_reco = float(tdee_base + goal_delta_kcal(goal))
        reco_method = "tdee_plus_goal_delta"
    else:
        reco_method = "missing_profile"

    adjustment_kcal = 0.0
    alpha_new = alpha_prev
    decision = "accepted"
    decision_reason = "stable"
    follow_up_questions: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []

    if missing or (cur_center is None) or (len(cur_w) < 1):
        decision = "needs_confirmation"
        decision_reason = "missing_profile_or_weights"
    elif delta_kg_obs is None or prev_center is None:
        decision = "needs_confirmation"
        decision_reason = "missing_previous_week"

    if decision != "accepted":
        if "sex" in missing:
            follow_up_questions.append(
                {
                    "type": "confirm_sex",
                    "prompt": "¿Cuál es tu sexo biológico?",
                    "options": [
                        {"label": "Masculino", "value": "male"},
                        {"label": "Femenino", "value": "female"},
                    ],
                }
            )
        follow_up_questions.append(
            {
                "type": "confirm_weekly_weight",
                "prompt": "¿Cuál fue tu peso promedio de esta semana (kg)?",
                "options": [],
            }
        )
        alerts.append(
            {
                "code": "needs_weekly_weight",
                "severity": "info",
                "message": "Para recalibrar tu perfil metabólico necesito tu peso promedio semanal.",
            }
        )
    else:
        if (delta_kg_target is not None) and (delta_kg_obs is not None):
            err_kg = float(delta_kg_obs - delta_kg_target)
            raw_adj = -float(err_kg) * 7700.0 / 7.0
            shrink = 1.0 - (0.6 * uncertainty)
            raw_adj *= max(0.25, shrink)
            adjustment_kcal = _clamp(raw_adj, -float(max_weekly_kcal_adjust), float(max_weekly_kcal_adjust))

            if tdee_base is not None and tdee_base > 0:
                alpha_step = _clamp(err_kg * 0.02, -0.03, 0.03)
                alpha_new = _clamp(alpha_prev + alpha_step, 0.0, 0.25)

        if uncertainty >= 0.55:
            alerts.append(
                {
                    "code": "low_confidence",
                    "severity": "info",
                    "message": "La estimación tiene baja confianza por pocos datos. El ajuste será conservador.",
                }
            )
        if outliers >= 2:
            alerts.append(
                {
                    "code": "weight_outliers",
                    "severity": "info",
                    "message": "Detecté valores atípicos en el peso. Usé un cálculo robusto para evitar cambios erráticos.",
                }
            )

    tdee_effective = None
    if tdee_base is not None:
        tdee_effective = float(tdee_base * (1.0 - float(alpha_new)))

    recommended_kcal_day = None
    if base_reco is not None:
        recommended_kcal_day = float(base_reco + float(adjustment_kcal))
        if sex == "female":
            recommended_kcal_day = max(1200.0, recommended_kcal_day)
        elif sex == "male":
            recommended_kcal_day = max(1400.0, recommended_kcal_day)

    explainability: list[str] = []
    if tmb is not None:
        explainability.append(f"TMB estimada: {round(float(tmb), 0):.0f} kcal/día")
    if tdee_effective is not None:
        explainability.append(f"Gasto estimado (TDEE): {round(float(tdee_effective), 0):.0f} kcal/día")
    explainability = explainability[:2]

    payload = {
        "profile": {
            "sex": sex_raw or None,
            "age": age or None,
            "height_cm": height_cm or None,
            "weight_kg_latest": round(float(cur_center), 3) if cur_center is not None else None,
            "goal_type": goal,
            "activity_level": activity,
        },
        "weights": {
            "current_week_center": round(float(cur_center), 3) if cur_center is not None else None,
            "previous_week_center": round(float(prev_center), 3) if prev_center is not None else None,
            "delta_kg_obs": round(float(delta_kg_obs), 4) if delta_kg_obs is not None else None,
            "delta_kg_target": round(float(delta_kg_target), 4) if delta_kg_target is not None else None,
        },
        "metabolic": {
            "tmb_kcal_day": round(float(tmb), 2) if tmb is not None else None,
            "tdee_base_kcal_day": round(float(tdee_base), 2) if tdee_base is not None else None,
            "adaptation_alpha": round(float(alpha_new), 4),
            "tdee_effective_kcal_day": round(float(tdee_effective), 2) if tdee_effective is not None else None,
        },
        "recommendation": {
            "kcal_day": round(float(recommended_kcal_day), 2) if recommended_kcal_day is not None else None,
            "weekly_adjustment_kcal_day": round(float(adjustment_kcal), 2),
            "method": reco_method,
            "max_weekly_kcal_adjust": float(max_weekly_kcal_adjust),
        },
        "confidence": {
            "score": round(float(confidence), 4),
            "uncertainty_score": round(float(uncertainty), 4),
            "missing": missing,
            "data_points": data_points,
            "outliers_removed": outliers,
            "used_profile_weight": bool(used_profile_weight),
        },
        "decision": decision,
        "decision_reason": decision_reason,
        "alerts": alerts,
        "follow_up_questions": follow_up_questions,
        "explainability": explainability,
        "meta": {
            "algorithm": "exp-003_metabolic_profile_v0",
            "as_of": str(date.today()),
        },
    }

    return MetabolicResult(payload=payload)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    reco = result.get("recommendation") if isinstance(result.get("recommendation"), dict) else {}
    metabolic = result.get("metabolic") if isinstance(result.get("metabolic"), dict) else {}
    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    decision = str(result.get("decision") or "")

    lines: list[str] = []
    lines.append(f"decision: {decision}")

    kcal = reco.get("kcal_day")
    if kcal is not None:
        try:
            lines.append(f"kcal_day: {round(float(kcal), 0):.0f}")
        except Exception:
            pass

    adj = reco.get("weekly_adjustment_kcal_day")
    if adj is not None:
        try:
            adj_f = float(adj)
            sign = "+" if adj_f >= 0 else ""
            lines.append(f"weekly_adjustment_kcal_day: {sign}{round(adj_f, 0):.0f}")
        except Exception:
            pass

    alpha = metabolic.get("adaptation_alpha")
    if alpha is not None:
        try:
            lines.append(f"adaptation_alpha: {round(float(alpha), 4)}")
        except Exception:
            pass

    score = conf.get("score")
    if score is not None:
        try:
            lines.append(f"confidence: {round(float(score), 3)}")
        except Exception:
            pass

    explain = result.get("explainability")
    if isinstance(explain, list) and explain:
        for x in explain[:2]:
            if str(x).strip():
                lines.append(f"note: {str(x).strip()}")

    alerts = result.get("alerts")
    if isinstance(alerts, list) and alerts:
        a = alerts[0] if isinstance(alerts[0], dict) else None
        if a and str(a.get("message") or "").strip():
            lines.append(f"alert: {str(a.get('message')).strip()}")

    return "\n".join(lines).strip()
