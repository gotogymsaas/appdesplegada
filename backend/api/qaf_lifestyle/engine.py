from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal


Decision = Literal["accepted", "needs_confirmation"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _normalize_steps(steps: float | None) -> tuple[float | None, float]:
    if steps is None:
        return None, 0.0
    s = float(max(0.0, steps))
    score = _clamp01(s / 12000.0)
    conf = 0.85 if s > 0 else 0.35
    return score, conf


def _normalize_sleep_minutes(minutes: float | None) -> tuple[float | None, float]:
    if minutes is None:
        return None, 0.0
    m = float(max(0.0, minutes))
    score = _clamp01((m / 60.0) / 8.0)
    conf = 0.9 if m >= 240 else 0.55
    return score, conf


def _normalize_calories(cal: float | None) -> tuple[float | None, float]:
    if cal is None:
        return None, 0.0
    c = float(max(0.0, cal))
    score = _clamp01((c - 500.0) / 2500.0)
    conf = 0.75 if c > 0 else 0.3
    return score, conf


def _normalize_rhr_inverse(rhr: float | None, *, baseline_7d: float | None) -> tuple[float | None, float, dict[str, Any]]:
    if rhr is None:
        return None, 0.0, {"method": "missing"}
    r = float(rhr)
    if r <= 0:
        return None, 0.0, {"method": "invalid"}

    if baseline_7d is None or baseline_7d <= 0:
        score = _clamp01(1.0 - ((r - 55.0) / 30.0))
        conf = 0.55
        return score, conf, {"method": "range_fallback"}

    delta_pct = (r - float(baseline_7d)) / float(baseline_7d)
    score = _clamp01(0.7 - (1.6 * delta_pct))
    conf = 0.8
    return score, conf, {"method": "baseline", "baseline_7d": round(float(baseline_7d), 2), "delta_pct": round(float(delta_pct), 4)}


def _normalize_self_report_1_5(value: Any) -> tuple[float | None, float]:
    v = _safe_int(value)
    if v is None:
        return None, 0.0
    v = max(1, min(5, v))
    score = (float(v) - 1.0) / 4.0
    return score, 0.6


def _last_n_days(metrics: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if not isinstance(metrics, list):
        return []

    def key(row):
        return str(row.get("date") or "")

    out = [m for m in metrics if isinstance(m, dict)]
    out.sort(key=key)
    return out[-int(n) :]


def _consecutive_below(series: list[dict[str, Any]], field: str, threshold: float, days: int) -> bool:
    if not series or days <= 1:
        return False
    count = 0
    for row in reversed(series):
        v = _safe_float(row.get(field))
        if v is None:
            break
        # En algunos proveedores 0 se usa como "sin dato".
        # Evitar contarlo como deterioro real.
        if field in ("sleep_minutes", "steps") and float(v) <= 0:
            break
        if float(v) < float(threshold):
            count += 1
        else:
            break
        if count >= days:
            return True
    return False


def _pick_microhabits(candidates: list[dict[str, Any]], *, memory: dict[str, Any]) -> list[dict[str, Any]]:
    last_ids = []
    try:
        last_ids = memory.get("last_ids") if isinstance(memory.get("last_ids"), list) else []
    except Exception:
        last_ids = []
    last_ids = [str(x) for x in last_ids if str(x).strip()]
    avoid = set(last_ids[:2])

    out: list[dict[str, Any]] = []
    for c in candidates:
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        if cid in avoid:
            continue
        out.append(c)
        if len(out) >= 3:
            break
    return out


def _catalog() -> list[dict[str, Any]]:
    return [
        {"id": "water_500", "label": "Beber 500 ml de agua antes del almuerzo", "type": "hydration", "effort": "low"},
        {"id": "walk_5_postmeal", "label": "Caminar 5 min después de una comida", "type": "movement", "effort": "low"},
        {"id": "break_60", "label": "Hacer 1 pausa de 2 min cada 60–90 min", "type": "sedentary", "effort": "low"},
        {"id": "breath_5", "label": "Respiración lenta 5 min (inhala 4, exhala 6)", "type": "stress", "effort": "low"},
        {"id": "mobility_3", "label": "Movilidad suave 3 min (cuello + cadera)", "type": "mobility", "effort": "low"},
        {"id": "sleep_earlier_30", "label": "Acostarte 30 min antes (solo hoy)", "type": "sleep", "effort": "medium"},
        {"id": "sunlight_5", "label": "Luz natural 5 min en la mañana", "type": "circadian", "effort": "low"},
        {"id": "walk_3x5", "label": "3 caminatas de 5 min (mañana/tarde/noche)", "type": "movement", "effort": "medium"},
    ]


@dataclass(frozen=True)
class LifestyleResult:
    payload: dict[str, Any]


def evaluate_lifestyle(payload: dict[str, Any]) -> LifestyleResult:
    daily_metrics = payload.get("daily_metrics") if isinstance(payload.get("daily_metrics"), list) else []
    self_report = payload.get("self_report") if isinstance(payload.get("self_report"), dict) else {}
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}

    series = _last_n_days(daily_metrics, 14)
    today = series[-1] if series else {}

    steps = _safe_float(today.get("steps"))
    sleep_minutes = _safe_float(today.get("sleep_minutes"))
    calories = _safe_float(today.get("calories"))
    rhr = _safe_float(today.get("resting_heart_rate_bpm"))

    # Sanitización de placeholders/valores implausibles para evitar lecturas genéricas.
    try:
        if steps is not None and float(steps) < 50:
            steps = None
    except Exception:
        pass
    try:
        if sleep_minutes is not None and float(sleep_minutes) < 30:
            sleep_minutes = None
    except Exception:
        pass
    try:
        if calories is not None and float(calories) < 50:
            calories = None
    except Exception:
        pass
    try:
        if rhr is not None and (float(rhr) < 35 or float(rhr) > 130):
            rhr = None
    except Exception:
        pass

    rhr_vals = []
    for row in series[-7:]:
        v = _safe_float(row.get("resting_heart_rate_bpm"))
        if v and v > 0:
            rhr_vals.append(float(v))
    rhr_baseline = _mean(rhr_vals[:-1]) if len(rhr_vals) >= 3 else None

    s_steps, c_steps = _normalize_steps(steps)
    s_sleep, c_sleep = _normalize_sleep_minutes(sleep_minutes)
    s_cal, c_cal = _normalize_calories(calories)
    s_stress_inv, c_rhr, rhr_meta = _normalize_rhr_inverse(rhr, baseline_7d=rhr_baseline)

    stress_sr, c_stress_sr = _normalize_self_report_1_5(self_report.get("stress_1_5"))
    sleepq_sr, c_sleepq_sr = _normalize_self_report_1_5(self_report.get("sleep_quality_1_5"))
    movement_sr, c_movement_sr = _normalize_self_report_1_5(self_report.get("movement_1_5"))

    stress_score = s_stress_inv
    stress_conf = c_rhr
    stress_method = "rhr" if s_stress_inv is not None else "self_report" if stress_sr is not None else "missing"
    if s_stress_inv is None and stress_sr is not None:
        stress_score = _clamp01(1.0 - float(stress_sr))
        stress_conf = c_stress_sr

    sleep_score = s_sleep
    sleep_conf = c_sleep
    if sleep_score is not None and sleepq_sr is not None:
        sleep_score = _clamp01((0.85 * float(sleep_score)) + (0.15 * float(sleepq_sr)))
        sleep_conf = _clamp01((0.85 * float(sleep_conf)) + (0.15 * float(c_sleepq_sr)))
    elif sleep_score is None and sleepq_sr is not None:
        # Si no hay wearable de sueño, aceptar auto-reporte como proxy (baja confianza)
        sleep_score = _clamp01(float(sleepq_sr))
        sleep_conf = 0.45

    # Movimiento: si faltan pasos, usar auto-reporte como proxy
    steps_score = s_steps
    steps_conf = c_steps
    if steps_score is None and movement_sr is not None:
        steps_score = _clamp01(float(movement_sr))
        steps_conf = 0.42

    activity_score = s_cal
    activity_conf = c_cal

    hydration_ml = _safe_float(self_report.get("hydration_ml"))
    hydration_score = None
    hydration_conf = 0.0
    if hydration_ml is not None and hydration_ml > 0:
        hydration_score = _clamp01(float(hydration_ml) / 2000.0)
        hydration_conf = 0.55

    missing: list[str] = []
    if sleep_score is None:
        missing.append("sleep")
    if steps_score is None:
        missing.append("steps")

    w_sleep = 0.34
    w_move = 0.24
    w_stress = 0.20
    w_hydr = 0.10
    w_activity = 0.12

    if hydration_score is None:
        total = w_sleep + w_move + w_stress + w_activity
        w_sleep, w_move, w_stress, w_activity = [w / total for w in (w_sleep, w_move, w_stress, w_activity)]
        w_hydr = 0.0

    parts: list[tuple[float, float]] = []
    if sleep_score is not None:
        parts.append((float(sleep_score), float(sleep_conf) * w_sleep))
    if steps_score is not None:
        parts.append((float(steps_score), float(steps_conf) * w_move))
    if stress_score is not None:
        parts.append((float(stress_score), float(stress_conf) * w_stress))
    if hydration_score is not None:
        parts.append((float(hydration_score), float(hydration_conf) * w_hydr))
    if activity_score is not None:
        parts.append((float(activity_score), float(activity_conf) * w_activity))

    def _weighted_value() -> float:
        value = 0.0
        denom = 0.0
        if sleep_score is not None:
            value += float(sleep_score) * w_sleep
            denom += w_sleep
        if steps_score is not None:
            value += float(steps_score) * w_move
            denom += w_move
        if stress_score is not None:
            value += float(stress_score) * w_stress
            denom += w_stress
        if hydration_score is not None:
            value += float(hydration_score) * w_hydr
            denom += w_hydr
        if activity_score is not None:
            value += float(activity_score) * w_activity
            denom += w_activity
        return (value / denom) if denom > 0 else 0.0

    dhss_01 = _clamp01(_weighted_value())
    conf = _clamp01(sum(w for _s, w in parts) / max(1e-6, (w_sleep + w_move + w_stress + w_hydr + w_activity)))

    decision: Decision = "accepted"
    decision_reason = "ok"
    follow_up_questions: list[dict[str, Any]] = []
    if "sleep" in missing:
        decision = "needs_confirmation"
        decision_reason = "missing_sleep"
        follow_up_questions.append({
            "type": "confirm_sleep_quality",
            "prompt": "¿Cómo dormiste hoy? (1–5)",
            "options": [{"label": str(i), "value": i} for i in range(1, 6)],
        })
    if "steps" in missing:
        decision = "needs_confirmation"
        decision_reason = "missing_steps"
        follow_up_questions.append({
            "type": "confirm_movement",
            "prompt": "Hoy, ¿cómo estuvo tu movimiento? (1–5)",
            "options": [{"label": str(i), "value": i} for i in range(1, 6)],
        })
    if stress_score is None:
        follow_up_questions.append({
            "type": "confirm_stress",
            "prompt": "¿Cómo estuvo tu estrés hoy? (1–5)",
            "options": [{"label": str(i), "value": i} for i in range(1, 6)],
        })

    patterns: list[dict[str, Any]] = []
    low_sleep_3 = _consecutive_below(series, "sleep_minutes", 360.0, 3)
    low_steps_5 = _consecutive_below(series, "steps", 5000.0, 5)
    if low_sleep_3:
        patterns.append({"key": "low_sleep_3d", "severity": "high", "message": "Tu cuerpo lleva 3 días con poco sueño. Hoy conviene priorizar recuperación."})
    if low_steps_5:
        patterns.append({"key": "low_steps_5d", "severity": "medium", "message": "Llevas varios días con poco movimiento. Hoy vamos a reactivar suave."})
    if rhr_baseline is not None and rhr is not None and rhr > 0:
        delta = (float(rhr) - float(rhr_baseline)) / float(rhr_baseline)
        if delta >= 0.08:
            patterns.append({"key": "rhr_up", "severity": "medium", "message": "Tu FC en reposo está por encima de tu promedio: podría ser acumulación de estrés/fatiga."})

    dhss = int(round(dhss_01 * 100.0))
    band = "high_capacity" if dhss >= 80 else "moderate" if dhss >= 60 else "fatigue" if dhss >= 40 else "recovery"

    catalog = _catalog()
    candidates: list[dict[str, Any]] = []
    if band in ("fatigue", "recovery") or low_sleep_3:
        candidates.extend([c for c in catalog if c["id"] in ("breath_5", "mobility_3", "sleep_earlier_30")])
    if low_steps_5 or (s_steps is not None and float(s_steps) < 0.35):
        candidates.extend([c for c in catalog if c["id"] in ("walk_3x5", "walk_5_postmeal", "break_60")])
    if hydration_score is not None and hydration_score < 0.55:
        candidates.extend([c for c in catalog if c["id"] == "water_500"])
    if not candidates:
        candidates.extend([c for c in catalog if c["id"] in ("sunlight_5", "walk_5_postmeal", "mobility_3")])

    microhabits = _pick_microhabits(candidates, memory=memory)

    uncertainty = _clamp01(1.0 - conf)
    payload_out = {
        "decision": decision,
        "decision_reason": decision_reason,
        "dhss": {"score": dhss, "band": band},
        "confidence": {"score": round(float(conf), 4), "uncertainty_score": round(float(uncertainty), 4), "missing": missing},
        "signals": {
            "sleep": {"value": sleep_minutes, "score01": (round(float(sleep_score), 4) if sleep_score is not None else None)},
            "steps": {"value": steps, "score01": (round(float(steps_score), 4) if steps_score is not None else None), "proxy": ("self_report" if s_steps is None and steps_score is not None else "wearable" if s_steps is not None else None)},
            "stress_inv": {"value": rhr, "score01": (round(float(stress_score), 4) if stress_score is not None else None), "method": stress_method, "meta": rhr_meta},
            "activity_prev": {"value": calories, "score01": (round(float(activity_score), 4) if activity_score is not None else None)},
            "hydration": {"value": hydration_ml, "score01": (round(float(hydration_score), 4) if hydration_score is not None else None)},
        },
        "patterns": patterns,
        "microhabits": microhabits,
        "follow_up_questions": follow_up_questions,
        "meta": {"algorithm": "exp-007_lifestyle_intelligence_v0", "as_of": str(date.today())},
    }
    return LifestyleResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    dhss = result.get("dhss") if isinstance(result.get("dhss"), dict) else {}
    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    sig = result.get("signals") if isinstance(result.get("signals"), dict) else {}
    patterns = result.get("patterns") if isinstance(result.get("patterns"), list) else []
    micro = result.get("microhabits") if isinstance(result.get("microhabits"), list) else []

    lines: list[str] = []

    lines.append("Estado de hoy — Tu Sistema")

    # DHSS
    band = str(dhss.get('band') or '').strip()
    score = dhss.get('score')
    if score is not None:
        try:
            score_i = int(score)
        except Exception:
            score_i = None
        if score_i is not None:
            label = (
                "Recuperación" if band == 'recovery' else
                "Fatiga" if band == 'fatigue' else
                "Capacidad moderada" if band == 'moderate' else
                "Alta capacidad" if band == 'high_capacity' else
                "Estado" 
            )
            lines.append(f"DHSS: {score_i}/100 — {label}")
            lines.append("")
            lines.append("DHSS = Daily Human System Score")
            lines.append("Es el indicador que usamos para decidir cuánto exigir hoy sin comprometer tu rendimiento a medio plazo.")

    # Confianza
    if conf.get('score') is not None:
        try:
            pct = round(float(conf.get('score')) * 100.0, 0)
            # No hacerlo protagonista: solo una nota corta.
            if pct >= 70:
                lines.append("")
                lines.append("Precisión: alta")
            elif pct >= 45:
                lines.append("")
                lines.append("Precisión: media")
            else:
                lines.append("")
                lines.append("Precisión: baja (faltan algunos datos para afinarlo)")
        except Exception:
            pass

    # Lectura rápida de señales (si hay valores)
    def _v(path: str):
        cur = sig
        for part in path.split('.'):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    sleep_min = _v('sleep.value')
    steps = _v('steps.value')
    rhr = _v('stress_inv.value')

    quick: list[str] = []
    try:
        if sleep_min is None:
            quick.append("Sueño: sin dato")
        else:
            # Si es muy bajo, suele ser 'no registrado' o incompleto.
            sm = float(sleep_min)
            if sm < 30:
                quick.append("Sueño: sin dato")
            else:
                quick.append(f"Sueño: {int(round(sm / 60.0))}h")
    except Exception:
        pass
    try:
        if steps is None:
            quick.append("Pasos: sin dato")
        else:
            st = float(steps)
            if st < 50:
                quick.append("Pasos: sin dato")
            else:
                quick.append(f"Pasos: {int(round(st))}")
    except Exception:
        pass
    try:
        if rhr is not None:
            quick.append(f"FC reposo: {int(round(float(rhr)))} bpm")
    except Exception:
        pass
    if quick:
        lines.append("\n📸 Tu foto del día")
        # Mostrar en líneas separadas para más claridad
        for q in quick[:3]:
            lines.append(f"• {q}")

    # Patrones (máx 2)
    if patterns:
        msgs = []
        for p in patterns[:2]:
            if isinstance(p, dict) and p.get('message'):
                msgs.append(str(p.get('message')))
        if msgs:
            lines.append("\n🔎 Lectura estratégica")
            for m in msgs:
                lines.append(m)
            lines.append("")
            lines.append("En alto rendimiento, saber cuándo bajar es parte del progreso.")

    # Recomendación de entrenamiento hoy (determinista por banda)
    lines.append("\n🎯 Enfoque óptimo para hoy")
    if band in ('recovery', 'fatigue'):
        try:
            if score is not None:
                lines.append("Hoy gana la consistencia, no la intensidad.")
                lines.append("")
                lines.append(f"Cuando el sistema marca {int(score)}/100, el objetivo es recuperar margen, no gastar el que queda.")
        except Exception:
            lines.append("Hoy gana la consistencia, no la intensidad.")
        lines.append("")
        lines.append("Elige una acción simple y ejecútala con calidad:")
        lines.append("")
        lines.append("• Caminata suave 20–30 min")
        lines.append("• Movilidad + estiramientos 8–12 min")
        lines.append("")
        lines.append("Pequeña carga. Máxima intención.")
    elif band == 'moderate':
        lines.append("Puedes entrenar, pero sin ir al límite.")
        lines.append("Calidad > ego.")
        lines.append("")
        lines.append("Elige una acción simple y ejecútala con calidad:")
        lines.append("")
        lines.append("• Fuerza moderada (RPE 6–7)")
        lines.append("• Cardio zona 2 25–40 min")
    else:
        lines.append("Buen día para progresar.")
        lines.append("Puedes empujar un poco más si tu técnica y energía se sienten estables.")
        lines.append("")
        lines.append("Elige una acción simple y ejecútala con calidad:")
        lines.append("")
        lines.append("• Fuerza con progresión")
        lines.append("• Intervalos cortos (si ya estás acostumbrado)")

    # Micro-hábitos
    if micro:
        names = [str(x.get('label')) for x in micro if isinstance(x, dict) and x.get('label')]
        if names:
            lines.append("\n✅ Micro-hábitos de precisión (elige 1–3)")
            for n in names[:3]:
                lines.append(f"• {n}")
            lines.append("")
            lines.append("Esto no es “hacer poco”.")
            lines.append("Es construir base para volver más fuerte mañana.")

    # Si la precisión salió baja, dar una forma rápida de mejorar sin fricción.
    try:
        if conf.get('score') is not None and float(conf.get('score') or 0.0) < 0.45:
            lines.append("\n🔧 Afinemos en 10 segundos")
            lines.append("Respóndeme así:")
            lines.append("")
            lines.append("• Sueño 1/5")
            lines.append("• Movimiento 1/5")
            lines.append("• Estrés 1/5")
            lines.append("")
            lines.append("Con eso ajusto el sistema.")
    except Exception:
        pass

    lines.append("\nResponde con una opción y lo dejamos listo:")
    lines.append("")
    lines.append("A) Caminata")
    lines.append("B) Movilidad")
    lines.append("C) Fuerza moderada")
    lines.append("D) Cardio zona 2")
    return "\n".join(lines).strip()
