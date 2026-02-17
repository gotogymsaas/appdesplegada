from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.utils import timezone

from api.models import User

from .models import FitnessSync


@dataclass(frozen=True)
class CoachState:
    morning_state: str
    energy_mode: str
    rationale: str
    last_sync_at: str | None
    metrics: dict[str, Any]


def _get_latest_metrics(user: User) -> tuple[dict[str, Any], timezone.datetime | None]:
    latest = FitnessSync.objects.filter(user=user).order_by("-created_at").first()
    if not latest:
        return {}, None
    metrics = latest.metrics or {}
    return metrics, latest.created_at


def compute_coach_state(user: User) -> CoachState:
    metrics, created_at = _get_latest_metrics(user)

    sleep_minutes = metrics.get("sleep_minutes")
    rhr = metrics.get("resting_heart_rate_bpm")

    try:
        sleep = float(sleep_minutes) if sleep_minutes is not None else None
    except Exception:
        sleep = None

    try:
        rhr_num = float(rhr) if rhr is not None else None
    except Exception:
        rhr_num = None

    # Heurística simple (MVP) para experiencia "Laura"
    if sleep is not None and sleep < 360:
        morning_state = "fatigada"
        energy_mode = "proteccion"
        rationale = "Dormiste menos de 6 horas. El cuerpo está más sensible hoy."
    elif sleep is not None and sleep < 420:
        morning_state = "neutral"
        energy_mode = "normal"
        rationale = "Dormiste un poco menos de lo ideal. Mejor intensidad moderada."
    else:
        morning_state = "recuperada"
        energy_mode = "normal"
        rationale = "Sueño suficiente para sostener una rutina estable."

    if rhr_num is not None and rhr_num >= 80:
        # Ajuste conservador por RHR alto
        if morning_state == "recuperada":
            morning_state = "neutral"
        energy_mode = "proteccion"
        rationale = (rationale + " FC en reposo elevada: conviene bajar intensidad.").strip()

    last_sync_at = created_at.isoformat() if created_at else None

    return CoachState(
        morning_state=morning_state,
        energy_mode=energy_mode,
        rationale=rationale,
        last_sync_at=last_sync_at,
        metrics=metrics,
    )


def coach_message_for_now(state: CoachState) -> str:
    if state.morning_state == "fatigada":
        return "Dormiste menos de lo habitual y tu cuerpo está más sensible hoy. Ajustemos intensidad."
    if state.morning_state == "neutral":
        return "Tu energía está en un punto medio hoy. Mantengamos intensidad moderada y cuidemos recuperación."
    return "Tu cuerpo parece estable hoy. Podemos avanzar con claridad y sin exceso."
