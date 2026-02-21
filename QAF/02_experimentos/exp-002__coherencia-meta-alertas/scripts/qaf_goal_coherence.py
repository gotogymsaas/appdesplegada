from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal


GoalType = Literal["deficit", "maintenance", "gain"]
MealSlot = Literal["breakfast", "lunch", "dinner", "snack", "unknown"]


@dataclass(frozen=True)
class GoalInference:
    goal_type: GoalType
    confidence: float
    source: str  # explicit|text|default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def infer_goal(user_context: dict[str, Any] | None) -> GoalInference:
    ctx = user_context or {}
    explicit = ctx.get("goal_type")
    if isinstance(explicit, str) and explicit in {"deficit", "maintenance", "gain"}:
        return GoalInference(goal_type=explicit, confidence=1.0, source="explicit")

    text = (ctx.get("goal_text") or "")
    if isinstance(text, str) and text.strip():
        t = text.lower()
        # déficit
        if re.search(r"\b(bajar|perder|definir|recortar|cut|déficit|deficit|quemar grasa|perder grasa)\b", t):
            return GoalInference(goal_type="deficit", confidence=0.75, source="text")
        # ganancia / masa
        if re.search(r"\b(subir|ganar|masa|volumen|bulk|bulking|aumentar)\b", t):
            return GoalInference(goal_type="gain", confidence=0.75, source="text")
        # mantenimiento
        if re.search(r"\b(mantener|mantenimiento|recomposición|recomp)\b", t):
            return GoalInference(goal_type="maintenance", confidence=0.7, source="text")

    # Default operativo: mantenimiento con baja confianza.
    return GoalInference(goal_type="maintenance", confidence=0.45, source="default")


def estimate_maintenance_kcal_day(*, weight_kg: float | None, activity_level: str | None = None) -> tuple[float | None, float, str]:
    """Estimador MVP sin sexo/BMR completo.

    - Base: 30 kcal/kg/día (aprox) si hay peso.
    - Ajuste por actividad: low/moderate/high.
    Retorna: (kcal, confidence, method)
    """

    if weight_kg is None or not (float(weight_kg) > 0):
        return None, 0.0, "missing_weight"

    base = 30.0 * float(weight_kg)
    al = (activity_level or "").strip().lower() or "moderate"
    factor = {
        "low": 0.92,
        "moderate": 1.0,
        "high": 1.10,
    }.get(al, 1.0)

    kcal = base * factor
    conf = 0.65 if al in {"low", "moderate", "high"} else 0.55
    return float(kcal), conf, "kcal_per_kg"


def daily_target_from_goal(maintenance_kcal: float, goal: GoalType) -> float:
    if goal == "deficit":
        return maintenance_kcal * 0.82
    if goal == "gain":
        return maintenance_kcal * 1.12
    return maintenance_kcal


def per_meal_weight(slot: MealSlot) -> float:
    return {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.30,
        "snack": 0.10,
        "unknown": 0.33,
    }.get(slot, 0.33)


def coherence_score(*, meal_kcal: float, target_kcal: float, sigma: float = 0.55) -> float:
    eps = 1e-6
    ratio = float(meal_kcal) / max(eps, float(target_kcal))
    score = math.exp(-abs(math.log(max(eps, ratio))) / float(sigma))
    return _clamp01(score)


def evaluate_meal(
    user_context: dict[str, Any] | None,
    meal: dict[str, Any] | None,
) -> dict[str, Any]:
    user_context = user_context or {}
    meal = meal or {}

    goal_inf = infer_goal(user_context)
    weight_kg = user_context.get("weight_kg")
    try:
        weight_kg_f = float(weight_kg) if weight_kg is not None else None
    except Exception:
        weight_kg_f = None

    maint_kcal, maint_conf, maint_method = estimate_maintenance_kcal_day(
        weight_kg=weight_kg_f,
        activity_level=(user_context.get("activity_level") if isinstance(user_context.get("activity_level"), str) else None),
    )

    missing: list[str] = []
    if maint_kcal is None:
        missing.append("weight_kg")

    meal_kcal_raw = meal.get("total_calories")
    try:
        meal_kcal = float(meal_kcal_raw) if meal_kcal_raw is not None else None
    except Exception:
        meal_kcal = None
    if meal_kcal is None or meal_kcal <= 0:
        missing.append("meal.total_calories")

    slot = meal.get("meal_slot")
    slot_norm: MealSlot = slot if isinstance(slot, str) and slot in {"breakfast", "lunch", "dinner", "snack", "unknown"} else "unknown"

    uncertainty_score = meal.get("uncertainty_score")
    try:
        u = _clamp01(float(uncertainty_score)) if uncertainty_score is not None else 0.35
    except Exception:
        u = 0.35

    needs_confirmation = meal.get("needs_confirmation")
    if not isinstance(needs_confirmation, bool):
        needs_confirmation = None

    # Targets
    daily_target = None
    per_meal_target = None
    if maint_kcal is not None:
        daily_target = float(daily_target_from_goal(float(maint_kcal), goal_inf.goal_type))
        per_meal_target = float(daily_target * per_meal_weight(slot_norm))

    alerts: list[dict[str, Any]] = []

    # Alertas por faltantes
    if goal_inf.confidence < 0.6:
        alerts.append(
            {
                "code": "missing_goal",
                "severity": "info",
                "message": "No tengo tu meta explícita (déficit/masa/mantenimiento). Si me la confirmas, puedo evaluar mejor esta comida.",
            }
        )
    if "weight_kg" in missing:
        alerts.append(
            {
                "code": "missing_weight",
                "severity": "info",
                "message": "Sin tu peso no puedo estimar un objetivo diario confiable. Si lo actualizas, la coherencia será más precisa.",
            }
        )
    if "meal.total_calories" in missing:
        alerts.append(
            {
                "code": "missing_calories",
                "severity": "warning",
                "message": "No tengo calorías totales de la comida. Primero necesito estimarlas para evaluar coherencia.",
            }
        )

    # Alertas por incertidumbre (porciones)
    if needs_confirmation is True or u >= 0.6:
        alerts.append(
            {
                "code": "high_uncertainty",
                "severity": "warning",
                "message": "La porción parece incierta. Confirmar la porción primero evita conclusiones erróneas.",
            }
        )

    # Evaluación principal
    classification = "unknown"
    score = 0.0
    ratio = None
    if meal_kcal is not None and per_meal_target is not None and per_meal_target > 0:
        ratio = float(meal_kcal) / float(per_meal_target)
        score = coherence_score(meal_kcal=meal_kcal, target_kcal=per_meal_target)

        if ratio <= 0.7:
            classification = "under"
        elif ratio <= 1.15:
            classification = "ok"
        elif ratio <= 1.35:
            classification = "over"
        else:
            classification = "high_over"

        # Alertas por exceso/defecto con meta
        if goal_inf.goal_type == "deficit" and classification in {"over", "high_over"}:
            alerts.append(
                {
                    "code": "over_target_meal",
                    "severity": "warning" if classification == "over" else "high",
                    "message": "Para déficit, esta comida parece alta vs tu objetivo por comida. Si fue intencional, lo compensamos en el resto del día.",
                }
            )
        if goal_inf.goal_type == "gain" and classification == "under":
            alerts.append(
                {
                    "code": "under_target_meal",
                    "severity": "info",
                    "message": "Para ganancia/masa, esta comida parece baja vs tu objetivo por comida. Podemos reforzar en la siguiente comida si lo deseas.",
                }
            )

    # Ajuste final por confianza de contexto
    context_conf = _clamp01(0.5 * goal_inf.confidence + 0.5 * maint_conf)
    score_adj = _clamp01(float(score) * (1.0 - 0.6 * u) * context_conf)

    # Decisión / colapso
    decision = "accepted"
    decision_reason = "stable"
    if "meal.total_calories" in missing:
        decision = "needs_confirmation"
        decision_reason = "missing_calories"
    elif needs_confirmation is True or u >= 0.6:
        decision = "needs_confirmation"
        decision_reason = "high_uncertainty"
    elif goal_inf.confidence < 0.6:
        decision = "partial"
        decision_reason = "missing_goal"

    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted" and goal_inf.confidence < 0.9:
        follow_up_questions.append(
            {
                "type": "confirm_goal",
                "prompt": "¿Cuál es tu meta principal ahora?",
                "options": [
                    {"label": "Déficit (bajar grasa)", "value": "deficit"},
                    {"label": "Mantenimiento", "value": "maintenance"},
                    {"label": "Masa (ganar músculo)", "value": "gain"},
                ],
            }
        )

    explainability = []
    if daily_target is not None:
        explainability.append(f"Objetivo diario estimado: {round(daily_target, 0):.0f} kcal ({goal_inf.goal_type})")
    if per_meal_target is not None and meal_kcal is not None:
        explainability.append(
            f"Comida ({slot_norm}): {round(meal_kcal, 0):.0f} kcal vs objetivo {round(per_meal_target, 0):.0f} kcal"
        )
    explainability = explainability[:2]

    return {
        "goal": {
            "goal_type": goal_inf.goal_type,
            "confidence": round(float(goal_inf.confidence), 4),
            "source": goal_inf.source,
        },
        "targets": {
            "maintenance_kcal_day": round(float(maint_kcal), 2) if maint_kcal is not None else None,
            "daily_target_kcal": round(float(daily_target), 2) if daily_target is not None else None,
            "per_meal_target_kcal": round(float(per_meal_target), 2) if per_meal_target is not None else None,
            "method": maint_method,
            "confidence": round(float(context_conf), 4),
        },
        "meal": {
            "total_calories": round(float(meal_kcal), 2) if meal_kcal is not None else None,
            "meal_slot": slot_norm,
            "uncertainty_score": round(float(u), 4),
            "needs_confirmation": needs_confirmation,
        },
        "evaluation": {
            "coherence_score": round(float(score_adj), 4),
            "classification": classification,
            "ratio": round(float(ratio), 4) if ratio is not None else None,
            "missing": missing,
        },
        "decision": decision,
        "decision_reason": decision_reason,
        "alerts": alerts,
        "follow_up_questions": follow_up_questions,
        "explainability": explainability,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, help="JSON string o ruta a .json")
    args = parser.parse_args()

    raw = args.input_json
    if raw and not raw.strip():
        raise SystemExit(2)

    try:
        if raw and raw.strip() and raw.strip().endswith(".json"):
            payload = json.loads(open(raw, "r", encoding="utf-8").read())
        else:
            payload = json.loads(raw)
    except Exception as ex:
        print(json.dumps({"ok": False, "error": f"invalid_json:{ex}"}, ensure_ascii=False))
        return 2

    user_context = payload.get("user_context") if isinstance(payload, dict) else None
    meal = payload.get("meal") if isinstance(payload, dict) else None

    result = evaluate_meal(user_context, meal)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
