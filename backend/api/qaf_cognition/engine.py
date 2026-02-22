from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _norm_0_100_to_01(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    return _clamp01(v / 100.0)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _lower_text(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _extract_confidence_score(obj: Any) -> Optional[float]:
    d = _safe_dict(obj)

    conf = d.get("confidence")
    if isinstance(conf, dict):
        if conf.get("score") is not None:
            return _clamp01(conf.get("score"))
        if conf.get("total") is not None:
            return _clamp01(conf.get("total"))

    if d.get("confidence_score") is not None:
        return _clamp01(d.get("confidence_score"))

    return None


def _extract_uncertainty_score(obj: Any) -> Optional[float]:
    d = _safe_dict(obj)
    unc = d.get("uncertainty")
    if isinstance(unc, dict) and unc.get("uncertainty_score") is not None:
        return _clamp01(unc.get("uncertainty_score"))

    if d.get("uncertainty_score") is not None:
        return _clamp01(d.get("uncertainty_score"))

    # algunos motores usan confidence.uncertainty_score
    conf = d.get("confidence")
    if isinstance(conf, dict) and conf.get("uncertainty_score") is not None:
        return _clamp01(conf.get("uncertainty_score"))

    return None


def _quality_from_confidence_uncertainty(conf: Optional[float], unc: Optional[float]) -> float:
    c = 0.6 if conf is None else _clamp01(conf)
    u = 0.5 if unc is None else _clamp01(unc)
    return _clamp01(0.65 * c + 0.35 * (1.0 - u))


def _pick_latest_week_entry(week_map: Any, week_id: str) -> Optional[Dict[str, Any]]:
    wm = _safe_dict(week_map)
    if week_id in wm and isinstance(wm.get(week_id), dict):
        return wm.get(week_id)

    # fallback: último por key ordenada
    keys = [k for k in wm.keys() if isinstance(k, str)]
    if not keys:
        return None
    latest_key = sorted(keys)[-1]
    row = wm.get(latest_key)
    return row if isinstance(row, dict) else None


@dataclass(frozen=True)
class CognitionInputs:
    locale: str
    week_id: str
    message: str
    observations: Dict[str, Any]

    # persistencia (estado del usuario)
    coach_state: Dict[str, Any]
    coach_weekly_state: Dict[str, Any]

    # perfil mínimo
    goal_type: Optional[str]
    activity_level: Optional[str]


def _extract_persisted_results(inp: CognitionInputs) -> Dict[str, Any]:
    cs = inp.coach_state
    ws = inp.coach_weekly_state

    lifestyle_last = (_safe_dict(cs.get("lifestyle_last")).get("result"))
    motivation_last = (_safe_dict(cs.get("motivation_last")).get("result"))
    progression_last = (_safe_dict(cs.get("progression_last")).get("result"))

    metabolic_last = _safe_dict(ws.get("metabolic_last"))
    body_trend_last = _safe_dict(ws.get("body_trend_last")).get("result")

    muscle_week = _pick_latest_week_entry(ws.get("muscle_measure"), inp.week_id)
    shape_week = _pick_latest_week_entry(ws.get("shape_presence"), inp.week_id)
    pp_week = _pick_latest_week_entry(ws.get("posture_proportion"), inp.week_id)

    return {
        "lifestyle_last": lifestyle_last if isinstance(lifestyle_last, dict) else None,
        "motivation_last": motivation_last if isinstance(motivation_last, dict) else None,
        "progression_last": progression_last if isinstance(progression_last, dict) else None,
        "metabolic_last": metabolic_last if isinstance(metabolic_last, dict) else None,
        "body_trend_last": body_trend_last if isinstance(body_trend_last, dict) else None,
        "muscle_measure_week": _safe_dict(muscle_week).get("result") if isinstance(muscle_week, dict) else None,
        "shape_presence_week": _safe_dict(shape_week).get("result") if isinstance(shape_week, dict) else None,
        "posture_proportion_week": _safe_dict(pp_week).get("result") if isinstance(pp_week, dict) else None,
    }


def _text_uncertainty_proxy(message: str) -> float:
    m = _lower_text(message)
    if not m:
        return 0.4

    cues = [
        "no sé",
        "no se",
        "confund",
        "perdid",
        "bloque",
        "ansio",
        "me cuesta",
        "no puedo",
        "no soy capaz",
        "me preocupa",
        "estres",
        "estrés",
        "agob",
        "no entiendo",
    ]
    hits = 0
    for c in cues:
        if c in m:
            hits += 1

    # proxy simple 0..1
    return _clamp01(0.25 + 0.18 * hits)


def _text_quantum_trigger(message: str) -> bool:
    m = _lower_text(message)
    if not m:
        return False
    triggers = ["qaf", "análisis cuántico", "analisis cuantico", "análisis qaf", "analisis qaf", "qaf lunes", "modelo qaf"]
    return any(t in m for t in triggers)


def _extract_dimension_scores(inp: CognitionInputs, persisted: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Devuelve (scores, qualities) por dimensión: nutrition/training/health/quantum.

    - score: 0..1 (qué tan 'activo' o 'relevante' está el dominio)
    - quality: 0..1 (calidad de señal/datos)
    """

    obs = inp.observations

    # Señales directas opcionales desde el caller (n8n/chat)
    nutrition_obs = _safe_dict(obs.get("nutrition"))
    training_obs = _safe_dict(obs.get("training"))
    health_obs = _safe_dict(obs.get("health"))
    quantum_obs = _safe_dict(obs.get("quantum"))

    def _extract_score_0_100(result: Any, path: List[str]) -> Optional[int]:
        cur = result
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        try:
            v = int(cur)
        except Exception:
            try:
                v = int(float(cur))
            except Exception:
                return None
        return max(0, min(100, v))

    # Lifestyle (Exp-007) -> energía/estrés base
    lifestyle = persisted.get("lifestyle_last")
    dhss_01 = None
    if isinstance(lifestyle, dict):
        dhss = lifestyle.get("dhss")
        if isinstance(dhss, dict):
            dhss_01 = _norm_0_100_to_01(dhss.get("score"))
        elif dhss is not None:
            dhss_01 = _norm_0_100_to_01(dhss)

    lifestyle_conf = _extract_confidence_score(lifestyle)
    lifestyle_unc = _extract_uncertainty_score(lifestyle)
    q_lifestyle = _quality_from_confidence_uncertainty(lifestyle_conf, lifestyle_unc)

    # Motivation (Exp-008)
    motivation = persisted.get("motivation_last")
    mot_conf = _extract_confidence_score(motivation)
    mot_unc = _extract_uncertainty_score(motivation)
    q_motivation = _quality_from_confidence_uncertainty(mot_conf, mot_unc)

    # Progression (Exp-009)
    progression = persisted.get("progression_last")
    prog_conf = _extract_confidence_score(progression)
    prog_unc = _extract_uncertainty_score(progression)
    q_progression = _quality_from_confidence_uncertainty(prog_conf, prog_unc)

    # Body posture related (Exp-012/013/010) -> health dimension
    shape = persisted.get("shape_presence_week")
    pp = persisted.get("posture_proportion_week")
    muscle = persisted.get("muscle_measure_week")

    q_shape = _quality_from_confidence_uncertainty(_extract_confidence_score(shape), _extract_uncertainty_score(shape))
    q_pp = _quality_from_confidence_uncertainty(_extract_confidence_score(pp), _extract_uncertainty_score(pp))
    q_muscle = _quality_from_confidence_uncertainty(_extract_confidence_score(muscle), _extract_uncertainty_score(muscle))

    # Metabolic last (Exp-003)
    metabolic = persisted.get("metabolic_last")
    q_metabolic = _quality_from_confidence_uncertainty(_extract_confidence_score(metabolic), _extract_uncertainty_score(metabolic))

    # Señales: por defecto neutras
    nutrition_score = 0.5
    training_score = 0.5
    health_score = 0.5

    # Ajustes por señales directas
    if nutrition_obs.get("score") is not None:
        nutrition_score = _clamp01(nutrition_obs.get("score"))
    if training_obs.get("score") is not None:
        training_score = _clamp01(training_obs.get("score"))
    if health_obs.get("score") is not None:
        health_score = _clamp01(health_obs.get("score"))

    # Ajustes por señales persistidas
    if dhss_01 is not None:
        # dhss alto -> más 'capacidad' general; se reparte
        nutrition_score = _clamp01(nutrition_score * 0.7 + dhss_01 * 0.3)
        training_score = _clamp01(training_score * 0.7 + dhss_01 * 0.3)
        health_score = _clamp01(health_score * 0.7 + dhss_01 * 0.3)

    # Exp-009 readiness -> training_score
    readiness_01 = None
    if isinstance(progression, dict):
        r0 = _extract_score_0_100(progression, ["readiness", "score"])
        readiness_01 = _norm_0_100_to_01(r0)
    if readiness_01 is not None:
        training_score = _clamp01(0.4 * training_score + 0.6 * readiness_01)

    # Exp-012/013/010 -> health_score (promedio de proxies disponibles)
    health_parts: List[float] = []
    if isinstance(shape, dict):
        s0 = _extract_score_0_100(shape, ["variables", "overall_presence"])
        s01 = _norm_0_100_to_01(s0)
        if s01 is not None:
            health_parts.append(float(s01))
    if isinstance(pp, dict):
        p0 = _extract_score_0_100(pp, ["variables", "alignment_silhouette_index"])
        p01 = _norm_0_100_to_01(p0)
        if p01 is not None:
            health_parts.append(float(p01))
    if isinstance(muscle, dict):
        # no hay overall; tomamos postura estática + balance como proxy
        m_posture = _extract_score_0_100(muscle, ["variables", "static_posture"])
        m_balance = _extract_score_0_100(muscle, ["variables", "upper_lower_balance"])
        parts = []
        for x in (m_posture, m_balance):
            x01 = _norm_0_100_to_01(x)
            if x01 is not None:
                parts.append(float(x01))
        if parts:
            health_parts.append(sum(parts) / float(len(parts)))
    if health_parts:
        health_score = _clamp01(0.35 * health_score + 0.65 * (sum(health_parts) / float(len(health_parts))))

    # Si hay resultados recientes específicos, subimos relevancia de esa dimensión
    if isinstance(progression, dict) and progression:
        training_score = _clamp01(training_score + 0.12)

    if isinstance(metabolic, dict) and metabolic:
        nutrition_score = _clamp01(nutrition_score + 0.10)

    if any(isinstance(x, dict) and x for x in (shape, pp, muscle)):
        health_score = _clamp01(health_score + 0.12)

    # Quantum se activa por necesidad de claridad o trigger explícito
    uncertainty_text = _text_uncertainty_proxy(inp.message)
    quantum_score = _clamp01(0.25 + 0.65 * uncertainty_text)
    if _text_quantum_trigger(inp.message):
        quantum_score = _clamp01(max(quantum_score, 0.9))
    if quantum_obs.get("score") is not None:
        quantum_score = _clamp01(quantum_obs.get("score"))

    # Calidades por dimensión
    q_nutrition = _clamp01(0.55 * q_metabolic + 0.45 * q_lifestyle)
    q_training = _clamp01(0.55 * q_progression + 0.45 * q_lifestyle)
    q_health = _clamp01(0.34 * q_shape + 0.33 * q_pp + 0.33 * q_muscle)
    q_quantum = _clamp01(0.6 * q_motivation + 0.4 * (1.0 - uncertainty_text))

    return (
        {
            "nutrition": nutrition_score,
            "training": training_score,
            "health": health_score,
            "quantum": quantum_score,
        },
        {
            "nutrition": q_nutrition,
            "training": q_training,
            "health": q_health,
            "quantum": q_quantum,
        },
    )


def _choose_mode(scores: Dict[str, float], message: str) -> str:
    if _text_quantum_trigger(message):
        return "quantum"

    # Si quantum es muy alto, también
    if scores.get("quantum", 0.0) >= 0.8:
        return "quantum"

    # elegir el máximo entre nutrition/training/health
    candidates = {k: scores.get(k, 0.0) for k in ("nutrition", "training", "health")}
    return max(candidates.items(), key=lambda kv: float(kv[1]))[0]


def evaluate_cognition(
    *,
    user_profile: Dict[str, Any],
    coach_state: Dict[str, Any],
    coach_weekly_state: Dict[str, Any],
    observations: Optional[Dict[str, Any]] = None,
    message: str = "",
    week_id: str = "",
    locale: str = "es-CO",
) -> Dict[str, Any]:
    """Motor determinista de cognición QAF (v0).

    - No usa LLM.
    - Opera con señales parciales y entrega un output estable para que n8n/LLM solo 'narre'.
    """

    inp = CognitionInputs(
        locale=(locale or "").strip() or "es-CO",
        week_id=(week_id or "").strip() or "unknown",
        message=str(message or "").strip(),
        observations=_safe_dict(observations),
        coach_state=_safe_dict(coach_state),
        coach_weekly_state=_safe_dict(coach_weekly_state),
        goal_type=(user_profile.get("goal_type") if isinstance(user_profile, dict) else None),
        activity_level=(user_profile.get("activity_level") if isinstance(user_profile, dict) else None),
    )

    persisted = _extract_persisted_results(inp)
    dim_scores, dim_quality = _extract_dimension_scores(inp, persisted)

    mode = _choose_mode(dim_scores, inp.message)

    # Estado base E/A/X/S (inspirado en el QAF Engine n8n, pero aquí determinista y con defaults)
    # E (energía): proxy por lifestyle (dhss) y motivación
    uncertainty_text = _text_uncertainty_proxy(inp.message)

    lifestyle = persisted.get("lifestyle_last")
    dhss_01 = None
    if isinstance(lifestyle, dict):
        dhss = lifestyle.get("dhss")
        if isinstance(dhss, dict):
            dhss_01 = _norm_0_100_to_01(dhss.get("score"))
        elif dhss is not None:
            dhss_01 = _norm_0_100_to_01(dhss)

    # Defaults neutros
    E = 0.6
    A = 0.6
    X = 0.5
    S = 0.4

    if dhss_01 is not None:
        E = _clamp01(0.45 + 0.55 * dhss_01)
        # S inverso al estado diario (si dhss alto, menor ruido físico)
        S = _clamp01(0.55 - 0.45 * dhss_01)

    # Ajustes por dimensión seleccionada
    A = _clamp01(A * 0.75 + dim_scores.get("nutrition", 0.5) * 0.25)
    X = _clamp01(X * 0.75 + dim_scores.get("training", 0.5) * 0.25)

    # Cognición: incertidumbre textual eleva S_eff y metacognición
    cognitive_noise = _clamp01(uncertainty_text)

    # Variables del Quantum Coach (0..1)
    Psi = 0.55
    if inp.goal_type in ("deficit", "maintenance", "gain"):
        Psi = 0.65

    Omega = _clamp01(0.34 * E + 0.33 * A + 0.33 * X)

    Iyo = _clamp01(0.25 + 0.60 * dim_scores.get("quantum", 0.25))

    # Omega_IA: coherencia de IA como proxy de 'capacidad del sistema' (calidad media de señales)
    Q_data = _clamp01(
        0.25 * dim_quality.get("nutrition", 0.6)
        + 0.25 * dim_quality.get("training", 0.6)
        + 0.25 * dim_quality.get("health", 0.6)
        + 0.25 * dim_quality.get("quantum", 0.6)
    )

    Omega_IA = _clamp01(0.55 + 0.35 * Q_data)

    # Entropía efectiva: mezcla de S (física) y ruido cognitivo
    S_eff = _clamp01(0.6 * S + 0.4 * cognitive_noise)

    # Alineación humano–IA: si hay goal_type y nutrición coherente, sube; si hay ruido alto, baja
    C_align = _clamp01(0.55 + 0.25 * (Psi - 0.5) + 0.20 * (1.0 - S_eff))

    # Gravedad emocional: proxy por incertidumbre (mientras no haya señal clínica)
    G = _clamp01(0.25 + 0.70 * cognitive_noise)

    # CAP (0..1): potencial de claridad/acción
    cap_raw = (Psi * Omega * Iyo * C_align * Q_data) / (1.0 + S_eff)
    CAP = _clamp01(cap_raw * 2.0)  # reescalar a rango útil

    # Índices compuestos simples (0..1)
    coherence = _clamp01(0.5 * Omega + 0.5 * (1.0 - S_eff))
    impact = _clamp01(0.55 * coherence + 0.45 * CAP)
    efficiency = _clamp01(0.6 * impact + 0.4 * (1.0 - S_eff))

    # Ruptura: umbral simple
    rupture_detected = bool(S_eff >= 0.85 and Omega <= 0.35)

    # Protocolo humano-in-the-loop: por defecto reversible
    human_validation_required = bool(rupture_detected)

    # Decisión principal
    decision_type = "proceed"
    follow_up_questions: List[Dict[str, Any]] = []

    if Q_data < 0.35:
        decision_type = "ask_clarifying"
        follow_up_questions.append(
            {
                "id": "missing_signal",
                "prompt": "Para darte una recomendación precisa, ¿qué necesitas hoy: nutrición, entrenamiento, salud corporal o claridad mental?",
                "options": ["Nutrición", "Entrenamiento", "Salud", "Claridad"],
            }
        )

    if rupture_detected:
        decision_type = "needs_confirmation"
        follow_up_questions.append(
            {
                "id": "stability_check",
                "prompt": "Antes de avanzar: ¿te sientes en un estado seguro y estable para tomar decisiones hoy? (sí/no)",
                "options": ["Sí", "No"],
            }
        )

    # Next actions (máximo 3) sin tocar UX: son sugerencias para que n8n/coach las exprese
    next_actions: List[Dict[str, Any]] = []
    if mode == "nutrition":
        next_actions = [
            {"id": "nutrition_1", "title": "Define tu meta de hoy en 1 frase", "timebox_minutes": 1},
            {"id": "nutrition_2", "title": "Registra una comida clave (con porción aproximada)", "timebox_minutes": 3},
            {"id": "nutrition_3", "title": "Elige un ajuste pequeño y sostenible (\u00b1100–200 kcal)", "timebox_minutes": 2},
        ]
    elif mode == "training":
        next_actions = [
            {"id": "training_1", "title": "Haz un micro-objetivo de sesión (1 ejercicio + RPE)", "timebox_minutes": 2},
            {"id": "training_2", "title": "Ejecuta 1 bloque corto (10–20 min) sin perfeccionismo", "timebox_minutes": 20},
            {"id": "training_3", "title": "Cierra con 1 nota: qué funcionó y qué ajustar", "timebox_minutes": 1},
        ]
    elif mode == "health":
        next_actions = [
            {"id": "health_1", "title": "Haz 1 chequeo corporal rápido (postura/respiración)", "timebox_minutes": 1},
            {"id": "health_2", "title": "Aplica 2 correcciones suaves (30–60s cada una)", "timebox_minutes": 3},
            {"id": "health_3", "title": "Planifica 1 ajuste semanal de entrenamiento", "timebox_minutes": 2},
        ]
    else:  # quantum
        next_actions = [
            {"id": "quantum_1", "title": "Define qué quieres decidir hoy (1 frase)", "timebox_minutes": 1},
            {"id": "quantum_2", "title": "Identifica la mayor fuente de ruido (1 cosa)", "timebox_minutes": 2},
            {"id": "quantum_3", "title": "Elige una acción mínima de 10 min (ventana de acción)", "timebox_minutes": 10},
        ]

    # Salida estable
    return {
        "success": True,
        "engine": {
            "name": "qaf-cognition",
            "version": "0.1",
        },
        "inputs": {
            "week_id": inp.week_id,
            "locale": inp.locale,
        },
        "dimensions": {
            "scores": dim_scores,
            "quality": dim_quality,
            "selected": mode,
        },
        "state": {
            "E": E,
            "A": A,
            "X": X,
            "S": S,
            "Psi": Psi,
            "Omega": Omega,
            "Iyo": Iyo,
            "Omega_IA": Omega_IA,
            "S_eff": S_eff,
            "C_align": C_align,
            "G": G,
            "Q_data": Q_data,
        },
        "indices": {
            "coherence": coherence,
            "impact": impact,
            "efficiency": efficiency,
            "CAP": CAP,
        },
        "flags": {
            "rupture_detected": rupture_detected,
            "human_validation_required": human_validation_required,
        },
        "decision": {
            "mode": mode,
            "type": decision_type,
            "follow_up_questions": follow_up_questions[:2],
            "next_3_actions": next_actions[:3],
        },
        "policy": {
            "llm_role": "narrate_only",
            "human_responsibility": True,
            "medical_disclaimer_required": True,
        },
        "sources": {
            "persisted": {
                "lifestyle": bool(persisted.get("lifestyle_last")),
                "motivation": bool(persisted.get("motivation_last")),
                "progression": bool(persisted.get("progression_last")),
                "metabolic": bool(persisted.get("metabolic_last")),
                "shape_presence": bool(persisted.get("shape_presence_week")),
                "posture_proportion": bool(persisted.get("posture_proportion_week")),
                "muscle_measure": bool(persisted.get("muscle_measure_week")),
            },
        },
    }
