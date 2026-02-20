from __future__ import annotations

from typing import Any


def decide(
    *,
    missing_items: list[str],
    low: float,
    high: float,
    estimate: float,
    range_driver: str | None,
    items_count: int,
    uncertainty_score: float,
    goal_kcal_meal: float | None,
) -> dict[str, Any]:
    range_width = max(0.0, float(high) - float(low))
    rel_width = (range_width / float(estimate)) if estimate else 1.0

    crosses_goal = False
    if goal_kcal_meal is not None:
        try:
            gk = float(goal_kcal_meal)
            crosses_goal = (float(low) <= gk <= float(high))
        except Exception:
            crosses_goal = False

    needs_confirmation = bool(missing_items) or (rel_width >= 0.35) or (range_width >= 180.0) or crosses_goal

    if missing_items:
        decision = "needs_confirmation"
        decision_reason = "missing_items_in_db"
    elif needs_confirmation and items_count > 1:
        decision = "partial"
        decision_reason = "confirm_range_driver"
    elif needs_confirmation:
        decision = "needs_confirmation"
        decision_reason = "high_uncertainty"
    else:
        decision = "accepted"
        decision_reason = "low_uncertainty"

    return {
        "needs_confirmation": bool(needs_confirmation),
        "decision": decision,
        "decision_reason": decision_reason,
    }
