from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


Decision = Literal["accepted", "needs_confirmation"]
Mood = Literal["euforico", "neutral", "fatiga", "frustrado", "ansioso"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _norm_dict(d: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in d.values())
    if total <= 1e-9:
        n = len(d) if d else 1
        return {k: 1.0 / n for k in d.keys()} if d else {}
    return {k: max(0.0, float(v)) / total for k, v in d.items()}


def _ema(prev: dict[str, float], obs: dict[str, float], alpha: float) -> dict[str, float]:
    out = {}
    keys = set(prev.keys()) | set(obs.keys())
    for k in keys:
        p = float(prev.get(k, 0.0) or 0.0)
        o = float(obs.get(k, 0.0) or 0.0)
        out[k] = (1.0 - float(alpha)) * p + float(alpha) * o
    return out


def _keyword_score(text: str, patterns: list[str]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    for p in patterns:
        try:
            if re.search(p, t):
                score += 1.0
        except Exception:
            continue
    return float(1.0 - math.exp(-0.9 * score))


def infer_motivation_observation(message: str) -> tuple[dict[str, float], dict[str, int]]:
    msg = (message or "").strip()
    logro = _keyword_score(msg, [r"\b(meta|objetiv|record|récord|marca|progreso|nivel|subir|mejorar|%|kg|reps|pr)\b"])
    disciplina = _keyword_score(msg, [r"\b(constancia|h[aá]bito|rutina|cumplir|disciplina|no\s+negocio|ritual|racha)\b"])
    salud = _keyword_score(msg, [r"\b(salud|energ[ií]a|dorm|sue[nñ]o|dolor|recuper|estres|estr[eé]s|ansios|fatig|cansad)\b"])
    estetica = _keyword_score(msg, [r"\b(verse|verme|cuerpo|defin|abdomen|gl[uú]teo|silueta|ropa|est[eé]tica|postura)\b"])
    comunidad = _keyword_score(msg, [r"\b(equipo|amig|grupo|juntos|acompa[nñ]ad|comunidad|rank|reto\s+grupal)\b"])
    raw = {
        "logro": logro,
        "disciplina": disciplina,
        "salud": salud,
        "estetica": estetica,
        "comunidad": comunidad,
    }
    obs = _norm_dict(raw)
    evidence = {k: (1 if raw.get(k, 0.0) > 0.05 else 0) for k in raw.keys()}
    return obs, evidence


def infer_daily_mood(message: str, *, lifestyle_band: str | None = None) -> tuple[Mood, float, list[str]]:
    msg = (message or "").lower()
    signals: list[str] = []

    fat = bool(re.search(r"\b(cansad|agotad|fatig|sin\s+energ[ií]a|no\s+dorm|desvel)\b", msg))
    anx = bool(re.search(r"\b(ansios|ansiedad|estres|estr[eé]s|me\s+abruma|abrumad)\b", msg))
    fru = bool(re.search(r"\b(frustr|no\s+pude|me\s+dio\s+pereza|fall[eé]|me\s+rend[ií])\b", msg))
    euf = bool(re.search(r"\b(vamos|listo|motiv|con\s+todo|hoy\s+si|me\s+siento\s+bien)\b", msg))

    if fat:
        signals.append("text_fatigue")
    if anx:
        signals.append("text_anxiety")
    if fru:
        signals.append("text_frustration")
    if euf:
        signals.append("text_euphoria")

    if lifestyle_band in ("recovery", "fatigue"):
        signals.append(f"lifestyle_{lifestyle_band}")

    if anx:
        return "ansioso", 0.72, signals
    if fru:
        return "frustrado", 0.68, signals
    if fat or lifestyle_band in ("recovery", "fatigue"):
        return "fatiga", 0.7, signals
    if euf:
        return "euforico", 0.6, signals
    return "neutral", 0.45, signals


def _top_dimension(vector: dict[str, float]) -> str:
    if not vector:
        return "disciplina"
    return sorted(vector.items(), key=lambda kv: float(kv[1]), reverse=True)[0][0]


def _tone_for(profile_top: str, mood: Mood, *, pressure: str | None) -> str:
    if mood in ("ansioso", "fatiga"):
        return "calmado"
    if mood == "frustrado":
        return "reencuadre"
    if profile_top == "logro":
        return "directo" if pressure != "suave" else "calmado"
    if profile_top == "salud":
        return "calmado"
    if profile_top == "disciplina":
        return "estructurado" if pressure != "suave" else "calmado"
    if profile_top == "estetica":
        return "inspiracional"
    if profile_top == "comunidad":
        return "cercano"
    return "calmado"


def _challenge_for(profile_top: str, mood: Mood, *, intervention_level: int) -> dict[str, Any]:
    if intervention_level >= 3:
        return {"id": "mode_renacer_7d", "type": "modo_facil", "label": "Modo Renacer (7 días): hábitos mínimos", "minutes": 6}
    if mood in ("fatiga", "ansioso"):
        return {"id": "recovery_12", "type": "recuperacion", "label": "12 min: movilidad suave + respiración lenta", "minutes": 12}
    if mood == "frustrado" or intervention_level == 2:
        return {"id": "ritual_6", "type": "minimo", "label": "6 min: cumple el ritual (caminar + estirar)", "minutes": 6}
    if profile_top == "logro":
        return {"id": "perf_5pct", "type": "rendimiento", "label": "Reto: +5% en tu ejercicio principal (o +1 rep si no puedes subir peso)"}
    if profile_top == "salud":
        return {"id": "health_reset", "type": "salud", "label": "Reto: 10–12 min de movilidad + respiración para bajar carga", "minutes": 12}
    if profile_top == "disciplina":
        return {"id": "streak_ritual", "type": "consistencia", "label": "Reto: hoy solo 6–10 min. No es intensidad, es constancia.", "minutes": 8}
    if profile_top == "estetica":
        return {"id": "core_posture", "type": "enfoque", "label": "Reto: postura + core 10 min (se nota antes de bajar un kilo)", "minutes": 10}
    if profile_top == "comunidad":
        return {"id": "share_commit", "type": "social", "label": "Reto: comparte tu compromiso con alguien y cumple una sesión corta hoy"}
    return {"id": "default_min", "type": "minimo", "label": "Reto: 8–12 min de movimiento suave hoy", "minutes": 10}


def _reward_for(profile_top: str, *, streak: int) -> dict[str, Any]:
    if profile_top == "logro":
        return {"type": "badge", "label": "Hito desbloqueado", "note": "Enfoque en progreso medible"}
    if profile_top == "disciplina":
        return {"type": "streak", "label": f"Cadena de constancia: {int(streak)} días", "note": "La racha protege tu identidad"}
    if profile_top == "salud":
        return {"type": "wellbeing", "label": "Energía recuperada", "note": "Prioriza recuperación inteligente"}
    if profile_top == "estetica":
        return {"type": "visual", "label": "Cambio visible", "note": "Progreso sin obsesión"}
    if profile_top == "comunidad":
        return {"type": "social", "label": "Impacto compartido", "note": "Acompañamiento cuenta"}
    return {"type": "streak", "label": f"Progreso: {int(streak)} días", "note": "Sigue"}


@dataclass(frozen=True)
class MotivationResult:
    payload: dict[str, Any]


def evaluate_motivation(payload: dict[str, Any]) -> MotivationResult:
    message = str(payload.get("message") or "").strip()
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    preferences = payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {}
    gamification = payload.get("gamification") if isinstance(payload.get("gamification"), dict) else {}
    lifestyle = payload.get("lifestyle") if isinstance(payload.get("lifestyle"), dict) else {}

    prev_vec = memory.get("vector") if isinstance(memory.get("vector"), dict) else {}
    prev_vec = {str(k): float(v) for k, v in prev_vec.items() if _safe_float(v) is not None}
    if not prev_vec:
        prev_vec = {"logro": 0.2, "disciplina": 0.2, "salud": 0.2, "estetica": 0.2, "comunidad": 0.2}

    # Evitar contaminar el perfil con mensajes que son solo taps/botones.
    msg_low = (message or "").strip().lower()
    non_signal = msg_low in (
        "suave",
        "medio",
        "firme",
        "✅ lo hago",
        "lo hago",
        "modo fácil 7 días",
        "🟡 modo fácil 7 días",
    ) or (len(msg_low) <= 3)

    if non_signal:
        obs_vec, evidence = (_norm_dict({k: 0.0 for k in prev_vec.keys()}), {k: 0 for k in prev_vec.keys()})
        updated = _norm_dict(prev_vec)
    else:
        obs_vec, evidence = infer_motivation_observation(message)
        updated = _ema(_norm_dict(prev_vec), obs_vec, alpha=0.25)
        updated = _norm_dict(updated)

    lifestyle_band = None
    try:
        lifestyle_band = (lifestyle.get("dhss") or {}).get("band") if isinstance(lifestyle.get("dhss"), dict) else None
    except Exception:
        lifestyle_band = None

    mood, mood_conf, mood_signals = infer_daily_mood(message, lifestyle_band=str(lifestyle_band) if lifestyle_band else None)

    streak = int(_safe_float(gamification.get("streak")) or 0)
    days_inactive = int(_safe_float(memory.get("days_inactive")) or 0)

    intervention = 0
    if days_inactive >= 7:
        intervention = 3
    elif mood in ("frustrado", "ansioso"):
        intervention = 2
    elif mood == "fatiga":
        intervention = 1

    # Modo fácil (Renacer) activo: fuerza intervención 3 durante la ventana.
    try:
        ren_until = memory.get("renacer_until")
        if isinstance(ren_until, str) and ren_until:
            until_d = date.fromisoformat(ren_until[:10])
            if date.today() <= until_d:
                intervention = max(int(intervention), 3)
                mood_signals.append("mode_renacer_active")
    except Exception:
        pass

    profile_top = _top_dimension(updated)
    pressure = str(preferences.get("pressure") or "").strip().lower() or None
    tone = _tone_for(profile_top, mood, pressure=pressure)
    challenge = _challenge_for(profile_top, mood, intervention_level=int(intervention))
    reward = _reward_for(profile_top, streak=streak)

    text_signal_strength = max(obs_vec.values()) if obs_vec else 0.0
    conf = _clamp01(0.35 + 0.55 * float(text_signal_strength))
    if prev_vec:
        conf = _clamp01(conf + 0.05)

    decision: Decision = "accepted"
    decision_reason = "ok"
    follow_ups: list[dict[str, Any]] = []
    if not pressure:
        decision = "needs_confirmation"
        decision_reason = "missing_pressure_preference"
        follow_ups.append({
            "type": "pick_pressure",
            "prompt": "¿Cómo quieres que te empuje?",
            "options": [
                {"label": "Suave", "value": "suave"},
                {"label": "Medio", "value": "medio"},
                {"label": "Firme", "value": "firme"},
            ],
        })

    payload_out = {
        "decision": decision,
        "decision_reason": decision_reason,
        "profile": {
            "vector": {k: round(float(v), 4) for k, v in updated.items()},
            "top": profile_top,
            "confidence": round(float(conf), 4),
            "evidence": evidence,
        },
        "state": {
            "mood": mood,
            "confidence": round(float(mood_conf), 4),
            "signals": mood_signals,
        },
        "intervention": {
            "level": int(intervention),
            "days_inactive": int(days_inactive),
        },
        "tone": {"style": tone},
        "challenge": challenge,
        "reward": reward,
        "follow_up_questions": follow_ups,
        "meta": {"algorithm": "exp-008_motivacion_psicologica_v0", "as_of": str(date.today())},
    }
    return MotivationResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    prof = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    state = result.get("state") if isinstance(result.get("state"), dict) else {}
    chall = result.get("challenge") if isinstance(result.get("challenge"), dict) else {}
    reward = result.get("reward") if isinstance(result.get("reward"), dict) else {}

    lines: list[str] = []
    if result.get('decision') == 'needs_confirmation':
        lines.append("Antes de ajustar el tono, necesito una preferencia rápida.")
    if prof.get("top"):
        lines.append(f"perfil dominante: {prof.get('top')}")
    if state.get("mood"):
        lines.append(f"estado de hoy: {state.get('mood')}")
    if chall.get("label"):
        lines.append(f"reto: {chall.get('label')}")
    if reward.get("label"):
        lines.append(f"recompensa: {reward.get('label')}")
    lines.append("nota: si hay dolor fuerte o mareo, baja intensidad y prioriza seguridad.")
    return "\n".join(lines).strip()
