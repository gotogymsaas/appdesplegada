from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


Decision = Literal["accepted", "needs_confirmation"]
Action = Literal["progress", "deload", "variation", "minimum_viable", "swap_exercise"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _epley_1rm(load_kg: float, reps: int) -> float:
    r = max(1, int(reps))
    return float(load_kg) * (1.0 + (float(r) / 30.0))


def _tonnage(sets: int, reps: int, load_kg: float) -> float:
    return float(max(0, sets)) * float(max(0, reps)) * float(max(0.0, load_kg))


def parse_strength_line(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    t = raw.lower()
    m = re.search(r"^(?P<name>[^\d]+?)\s+(?P<sets>\d{1,2})\s*[x×]\s*(?P<reps>\d{1,3})(?:\s*[x×]\s*(?P<load>\d+(?:[\.,]\d+)?)(?:\s*kg)?)?\s*$", t)
    if not m:
        m2 = re.search(r"^(?P<sets>\d{1,2})\s*[x×]\s*(?P<reps>\d{1,3})(?:\s*[x×]\s*(?P<load>\d+(?:[\.,]\d+)?)(?:\s*kg)?)?\s+(?P<name>.+?)\s*$", t)
        if not m2:
            return None
        m = m2
    name = str(m.group('name') or '').strip()
    sets = int(m.group('sets'))
    reps = int(m.group('reps'))
    load_raw = m.group('load')
    load_kg = None
    if load_raw is not None:
        try:
            load_kg = float(str(load_raw).replace(',', '.'))
        except Exception:
            load_kg = None
    return {'type': 'strength', 'name': name, 'sets': sets, 'reps': reps, 'load_kg': load_kg}


def _readiness_score(signals: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    sleep_min = _safe_float(signals.get('sleep_minutes'))
    steps = _safe_float(signals.get('steps'))
    rhr = _safe_float(signals.get('resting_heart_rate_bpm'))
    rhr_base = _safe_float(signals.get('resting_hr_baseline_7d'))
    mood = str(signals.get('mood') or '').strip().lower() or None
    lifestyle_band = str(signals.get('lifestyle_band') or '').strip().lower() or None

    S = _clamp01((sleep_min or 0.0) / 480.0) if sleep_min is not None else None
    P = _clamp01((steps or 0.0) / 12000.0) if steps is not None else None

    H = None
    if rhr is not None and rhr_base is not None and rhr_base > 0:
        delta = (float(rhr) - float(rhr_base)) / float(rhr_base)
        H = _clamp01(1.0 - 2.0 * delta)
    elif rhr is not None:
        H = _clamp01(1.0 - ((float(rhr) - 55.0) / 30.0))

    mood_pen = 0.0
    if mood in ('fatiga', 'ansioso'):
        mood_pen = 0.20
    elif mood == 'frustrado':
        mood_pen = 0.15
    elif mood == 'euforico':
        mood_pen = -0.05
    if lifestyle_band in ('recovery', 'fatigue'):
        mood_pen = max(mood_pen, 0.18)

    weights = {'S': 0.45, 'P': 0.20, 'H': 0.25}
    present = {'S': S is not None, 'P': P is not None, 'H': H is not None}
    denom = sum(weights[k] for k, ok in present.items() if ok)
    if denom <= 1e-9:
        base = 0.55
    else:
        base = 0.0
        if S is not None:
            base += float(S) * (weights['S'] / denom)
        if P is not None:
            base += float(P) * (weights['P'] / denom)
        if H is not None:
            base += float(H) * (weights['H'] / denom)
    R = int(round(100.0 * _clamp01(base - 0.10 * mood_pen)))
    return R, {
        'sleep_score01': S,
        'steps_score01': P,
        'hr_score01': H,
        'mood_penalty': round(float(mood_pen), 4),
        'lifestyle_band': lifestyle_band,
    }


def _plateau_strength(history: list[dict[str, Any]]) -> tuple[bool, str]:
    rows = [h for h in history if isinstance(h, dict)]
    if len(rows) < 3:
        return False, 'insufficient_history'
    est = [_safe_float(r.get('est_1rm')) for r in rows[-3:]]
    rpes = [_safe_float(r.get('rpe')) for r in rows[-3:]]
    if any(v is None for v in est):
        ton = [_safe_float(r.get('tonnage')) for r in rows[-3:]]
        if any(v is None for v in ton):
            return False, 'missing_metrics'
        est = ton
    eps = 0.015
    p1 = (est[1] - est[0]) / max(1.0, float(est[0]))
    p2 = (est[2] - est[1]) / max(1.0, float(est[1]))
    rpe_up = False
    try:
        if rpes[2] is not None and rpes[1] is not None:
            rpe_up = float(rpes[2]) > float(rpes[1])
    except Exception:
        rpe_up = False
    if abs(p1) < eps and abs(p2) < eps and (rpe_up or (rpes[2] is not None and float(rpes[2]) >= 8.0)):
        return True, '3_sessions_no_gain_rpe_up'
    return False, 'no_plateau'


def _plateau_cardio(history: list[dict[str, Any]]) -> tuple[bool, str]:
    rows = [h for h in history if isinstance(h, dict)]
    if len(rows) < 3:
        return False, 'insufficient_history'
    mins = [_safe_float(r.get('minutes')) for r in rows[-3:]]
    rpe = [_safe_float(r.get('rpe')) for r in rows[-3:]]
    if any(v is None for v in mins) or any(v is None for v in rpe):
        return False, 'missing_metrics'
    stable = abs(float(mins[2]) - float(mins[1])) <= 2 and abs(float(mins[1]) - float(mins[0])) <= 2
    rpe_rise = float(rpe[2]) > float(rpe[1])
    if stable and rpe_rise and float(rpe[2]) >= 8:
        return True, 'minutes_stable_rpe_rising'
    return False, 'no_plateau'


def _decision(*, readiness: int, plateau: bool, rpe: float | None, completion_pct: float | None, pain: bool) -> tuple[Action, str]:
    if pain:
        return 'swap_exercise', 'pain_reported'
    if readiness < 45:
        return 'deload', 'low_readiness'
    if rpe is not None and float(rpe) >= 9.0:
        return 'deload', 'very_high_rpe'
    if completion_pct is not None and float(completion_pct) < 0.6:
        return 'minimum_viable', 'low_completion'
    if plateau:
        return 'variation', 'plateau_detected'
    if readiness >= 65 and (rpe is None or float(rpe) <= 7.0):
        return 'progress', 'ready_to_push'
    return 'minimum_viable', 'default_safe'


def _micro_goal(action: Action, *, modality: str, exercise_name: str | None = None) -> str:
    if action == 'progress':
        if modality == 'strength' and exercise_name:
            return f"Hoy: +1 rep en la primera serie de {exercise_name}."
        if modality == 'cardio':
            return 'Hoy: +2–5 min manteniendo respiración controlada.'
        return 'Hoy: mejora mínima medible.'
    if action == 'deload':
        return 'Hoy: técnica perfecta y terminar sin acumular fatiga.'
    if action == 'variation':
        return 'Hoy: cambiamos estímulo para romper estancamiento sin riesgo.'
    if action == 'swap_exercise':
        return 'Hoy: bajar impacto y elegir una variante segura.'
    return 'Hoy: cumplir el mínimo viable y proteger la constancia.'


@dataclass(frozen=True)
class ProgressionResult:
    payload: dict[str, Any]


def evaluate_progression(payload: dict[str, Any]) -> ProgressionResult:
    session = payload.get('session') if isinstance(payload.get('session'), dict) else {}
    strength = payload.get('strength') if isinstance(payload.get('strength'), dict) else None
    cardio = payload.get('cardio') if isinstance(payload.get('cardio'), dict) else None
    history = payload.get('history') if isinstance(payload.get('history'), dict) else {}
    signals = payload.get('signals') if isinstance(payload.get('signals'), dict) else {}

    missing: list[str] = []
    rpe = _safe_float(session.get('rpe_1_10'))
    completion = _safe_float(session.get('completion_pct'))
    pain = bool(session.get('pain'))
    if rpe is None:
        missing.append('rpe_1_10')
    if completion is None:
        missing.append('completion_pct')

    modality = 'unknown'
    if strength:
        modality = 'strength'
    elif cardio:
        modality = 'cardio'
    else:
        missing.append('modality')

    readiness, readiness_meta = _readiness_score(signals)

    plateau = False
    plateau_reason = ''
    exercise_name = None
    if modality == 'strength' and strength:
        exercise_name = str(strength.get('name') or '').strip() or None
        h = history.get(f"strength:{exercise_name}") if exercise_name else None
        plateau, plateau_reason = _plateau_strength(h if isinstance(h, list) else [])
    if modality == 'cardio' and cardio:
        plateau, plateau_reason = _plateau_cardio(history.get('cardio:default') if isinstance(history.get('cardio:default'), list) else [])

    action, action_reason = _decision(readiness=readiness, plateau=plateau, rpe=rpe, completion_pct=completion, pain=pain)
    micro = _micro_goal(action, modality=modality, exercise_name=exercise_name)

    confidence = _clamp01(0.75 - 0.12 * float(len(missing)))
    decision: Decision = 'accepted'
    decision_reason = 'ok'
    if missing:
        decision = 'needs_confirmation'
        decision_reason = 'missing_inputs'

    out = {
        'decision': decision,
        'decision_reason': decision_reason,
        'confidence': {'score': round(float(confidence), 4), 'missing': missing},
        'readiness': {'score': int(readiness), 'meta': readiness_meta},
        'plateau': {'detected': bool(plateau), 'reason': plateau_reason},
        'decision_engine': {'action': action, 'reason': action_reason},
        'micro_goal': micro,
        'meta': {'algorithm': 'exp-009_progression_intelligent_v0', 'as_of': str(date.today())},
    }
    return ProgressionResult(payload=out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ''
    r = result.get('readiness') if isinstance(result.get('readiness'), dict) else {}
    p = result.get('plateau') if isinstance(result.get('plateau'), dict) else {}
    d = result.get('decision_engine') if isinstance(result.get('decision_engine'), dict) else {}
    conf = result.get('confidence') if isinstance(result.get('confidence'), dict) else {}
    lines: list[str] = []
    if result.get('decision') == 'needs_confirmation':
        lines.append('Para ajustar tu evolución de entrenamiento con precisión, necesito 2–3 datos rápidos.')
    if r.get('score') is not None:
        lines.append(f"Readiness: {r.get('score')}/100")
    if p.get('detected') is True:
        lines.append('Plateau: detectado (no es falta de disciplina; es señal de fatiga/estímulo).')
    if d.get('action'):
        lines.append(f"Acción: {d.get('action')}")
    if result.get('micro_goal'):
        lines.append(f"Micro-objetivo: {result.get('micro_goal')}")
    if conf.get('missing'):
        lines.append('Faltan: ' + ', '.join([str(x) for x in conf.get('missing')]))
    return "\n".join(lines).strip()
