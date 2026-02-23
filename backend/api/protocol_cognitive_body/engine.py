from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


PROTOCOL_VERSION = "0.1"


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _norm_0_100_to_01(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:
        return None
    v = max(0.0, min(100.0, v))
    return v / 100.0


def _pick_latest_week_entry(wm: Any, week_id: str) -> Optional[Dict[str, Any]]:
    """Devuelve la fila más reciente para week_id si existe; si no, la última disponible."""
    if not isinstance(wm, dict):
        return None
    if week_id and week_id in wm and isinstance(wm.get(week_id), dict):
        return wm.get(week_id)
    keys = [k for k in wm.keys() if isinstance(k, str) and k]
    if not keys:
        return None
    latest_key = sorted(keys)[-1]
    row = wm.get(latest_key)
    return row if isinstance(row, dict) else None


def _extract_path(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _fresh_enough(updated_at_iso: str, *, max_age_hours: int) -> bool:
    try:
        if not updated_at_iso:
            return False
        # acepta ISO con Z o sin TZ
        s = updated_at_iso.replace("Z", "")
        dt = datetime.fromisoformat(s)
        return (datetime.utcnow() - dt) <= timedelta(hours=max_age_hours)
    except Exception:
        return False


@dataclass(frozen=True)
class ProtocolOutput:
    payload: Dict[str, Any]


def evaluate_protocol(
    *,
    user_profile: Dict[str, Any],
    coach_state: Dict[str, Any],
    coach_weekly_state: Dict[str, Any],
    qaf_cognition: Optional[Dict[str, Any]] = None,
    observations: Optional[Dict[str, Any]] = None,
    week_id: str = "",
    locale: str = "es-CO",
) -> ProtocolOutput:
    """Protocolo Cognitivo Corporal (determinista).

    Consolida outputs (persistidos y/o observaciones de la interacción) en:
    - dimensiones estructurales (0..1)
    - índices (0..100)
    - estado evolutivo
    - restricciones y foco
    - ritmo temporal

    No usa LLM.
    """

    cs = _safe_dict(coach_state)
    ws = _safe_dict(coach_weekly_state)
    obs = _safe_dict(observations)

    # ---- Señales persistidas (fuentes) ----
    # Arquitectura corporal (Exp-013) semanal
    pp_week = _pick_latest_week_entry(ws.get("posture_proportion"), week_id)
    pp_res = _safe_dict(_safe_dict(pp_week).get("result"))

    # Presencia visual (Exp-012) semanal
    shape_week = _pick_latest_week_entry(ws.get("shape_presence"), week_id)
    shape_res = _safe_dict(_safe_dict(shape_week).get("result"))

    # Piel (Exp-011) último
    skin_last_blob = _safe_dict(cs.get("skin_last_result"))
    skin_res = _safe_dict(skin_last_blob.get("result"))

    # Lifestyle (Exp-007) último
    lifestyle_last_blob = _safe_dict(cs.get("lifestyle_last"))
    lifestyle_res = _safe_dict(lifestyle_last_blob.get("result"))

    # Progresión (Exp-009) último
    progression_last_blob = _safe_dict(cs.get("progression_last"))
    progression_res = _safe_dict(progression_last_blob.get("result"))

    # ---- Dimensiones estructurales (0..1) ----
    dimensions: Dict[str, Any] = {}

    # Alineación: ASI de Exp-013 si existe
    alignment_01 = _norm_0_100_to_01(_extract_path(pp_res, ["variables", "alignment_silhouette_index"]))
    eff_post_01 = _norm_0_100_to_01(_extract_path(pp_res, ["variables", "postural_efficiency_score"]))

    # Estabilidad de cadera (proxy)
    hip_stab_01 = None
    try:
        hip = _extract_path(pp_res, ["variables", "symmetry_monitor", "hip_stability"])
        hip_stab_01 = _norm_0_100_to_01(hip)
    except Exception:
        hip_stab_01 = None

    # Presencia visual: overall_presence de Exp-012
    presence_01 = _norm_0_100_to_01(_extract_path(shape_res, ["variables", "overall_presence"]))

    # Energía/recuperación: dhss de lifestyle (0..100)
    energy_01 = None
    try:
        dhss = _extract_path(lifestyle_res, ["dhss", "score"]) or lifestyle_res.get("dhss")
        energy_01 = _norm_0_100_to_01(dhss)
    except Exception:
        energy_01 = None

    # Readiness (training): progression.readiness.score
    readiness_01 = _norm_0_100_to_01(_extract_path(progression_res, ["readiness", "score"]))

    # Piel: score principal (si existe)
    skin_01 = _norm_0_100_to_01(skin_res.get("skin_health_score"))

    # Defaults neutros si faltan señales
    alignment_01 = alignment_01 if alignment_01 is not None else 0.6
    eff_post_01 = eff_post_01 if eff_post_01 is not None else 0.6
    hip_stab_01 = hip_stab_01 if hip_stab_01 is not None else 0.6
    presence_01 = presence_01 if presence_01 is not None else 0.55
    energy_01 = energy_01 if energy_01 is not None else 0.6
    readiness_01 = readiness_01 if readiness_01 is not None else 0.6
    skin_01 = skin_01 if skin_01 is not None else 0.6

    dimensions["alignment"] = {
        "score01": round(_clamp01(alignment_01), 4),
        "source": "exp-013",
    }
    dimensions["postural_efficiency"] = {
        "score01": round(_clamp01(eff_post_01), 4),
        "source": "exp-013",
    }
    dimensions["hip_stability"] = {
        "score01": round(_clamp01(hip_stab_01), 4),
        "source": "exp-013",
    }
    dimensions["visual_presence"] = {
        "score01": round(_clamp01(presence_01), 4),
        "source": "exp-012",
    }
    dimensions["energy"] = {
        "score01": round(_clamp01(energy_01), 4),
        "source": "exp-007",
    }
    dimensions["training_readiness"] = {
        "score01": round(_clamp01(readiness_01), 4),
        "source": "exp-009",
    }
    dimensions["skin_vitality"] = {
        "score01": round(_clamp01(skin_01), 4),
        "source": "exp-011",
    }

    # ---- Índices propietarios (0..100) ----
    # (Pesos simples y estables; se pueden refinar sin cambiar contratos)
    body_architecture_index_01 = _clamp01(0.45 * alignment_01 + 0.35 * eff_post_01 + 0.20 * hip_stab_01)
    presence_visual_index_01 = _clamp01(0.70 * presence_01 + 0.30 * alignment_01)
    biological_consistency_index_01 = _clamp01(0.50 * energy_01 + 0.30 * readiness_01 + 0.20 * skin_01)

    indices = {
        "body_architecture_index": int(round(body_architecture_index_01 * 100.0)),
        "visual_presence_index": int(round(presence_visual_index_01 * 100.0)),
        "biological_consistency_index": int(round(biological_consistency_index_01 * 100.0)),
    }

    # ---- Reglas y restricciones ----
    restrictions: List[Dict[str, Any]] = []

    if alignment_01 < 0.6:
        restrictions.append(
            {
                "id": "limit_intensity_alignment",
                "title": "Limitar intensidad por alineación",
                "reason": "Cuando el eje está inestable, subir carga suele amplificar compensaciones.",
                "rule": "Prioriza técnica/estabilidad; evita 1RM y volumen máximo por 7 días.",
                "severity": "high" if alignment_01 < 0.5 else "medium",
            }
        )

    if energy_01 < 0.5:
        restrictions.append(
            {
                "id": "prioritize_recovery",
                "title": "Priorizar recuperación",
                "reason": "Energía baja reduce tolerancia al estrés y empeora ejecución.",
                "rule": "Baja 10–20% volumen; agrega movilidad suave y sueño como prioridad.",
                "severity": "medium",
            }
        )

    # ---- Estado evolutivo ----
    # Estados: estable / en_ajuste / en_optimizacion / en_expansion / en_consolidacion
    state = "estable"
    if body_architecture_index_01 < 0.58 or biological_consistency_index_01 < 0.52:
        state = "en_ajuste"
    elif body_architecture_index_01 >= 0.78 and biological_consistency_index_01 >= 0.70:
        state = "en_optimizacion"
    if body_architecture_index_01 >= 0.85 and presence_visual_index_01 >= 0.80 and biological_consistency_index_01 >= 0.78:
        state = "en_expansion"

    # Consolidación: si el estado se mantiene y los índices no fluctúan fuerte
    # (mecánica mínima: contar repeticiones del mismo estado guardadas en coach_state)
    prev_proto = _safe_dict(cs.get("protocol_cognitive_body"))
    prev_state = str(prev_proto.get("evolution_state") or "").strip().lower()
    prev_streak = int(_safe_dict(prev_proto.get("meta")).get("state_streak") or 0)
    streak = prev_streak + 1 if prev_state == state else 1
    if streak >= 3 and state in ("en_optimizacion", "estable"):
        state = "en_consolidacion"

    # ---- Foco siguiente (prioridad) ----
    focus: Dict[str, Any] = {
        "primary": "alignment" if alignment_01 < min(presence_01, energy_01) else ("energy" if energy_01 < min(alignment_01, presence_01) else "visual_presence"),
        "week_goal": "",
        "next_3_actions": [],
    }

    if focus["primary"] == "alignment":
        focus["week_goal"] = "Limpiar eje (alineación) y estabilizar base antes de subir intensidad."
        focus["next_3_actions"] = [
            {"id": "axis_1", "title": "2 minutos: micro-ajustes de eje (cuello + escápula)", "timebox_minutes": 2},
            {"id": "axis_2", "title": "Sesión: técnica + tempo (sin cargas máximas)", "timebox_minutes": 25},
            {"id": "axis_3", "title": "Cierre: respiración nasal 60s + postura neutra", "timebox_minutes": 2},
        ]
    elif focus["primary"] == "energy":
        focus["week_goal"] = "Recuperación y consistencia: subir energía primero, luego performance."
        focus["next_3_actions"] = [
            {"id": "rec_1", "title": "Define hora fija de sueño hoy", "timebox_minutes": 1},
            {"id": "rec_2", "title": "Caminata suave 10–20 min", "timebox_minutes": 15},
            {"id": "rec_3", "title": "Movilidad torácica 2 min", "timebox_minutes": 2},
        ]
    else:
        focus["week_goal"] = "Presencia y dirección estética: consolidar línea y coherencia visual."
        focus["next_3_actions"] = [
            {"id": "pres_1", "title": "2 minutos: eje limpio antes de fotos/outfit", "timebox_minutes": 2},
            {"id": "pres_2", "title": "Elige 1 ajuste de sastrería (fit) basado en tu eje", "timebox_minutes": 3},
            {"id": "pres_3", "title": "Repite foto semanal con mismo encuadre", "timebox_minutes": 2},
        ]

    # Si el motor de cognición trae acciones, las usamos como guía secundaria (no reemplaza reglas del protocolo)
    cognition_mode = None
    cognition_actions: List[Dict[str, Any]] = []
    try:
        dec = _safe_dict(_safe_dict(qaf_cognition).get("decision"))
        cognition_mode = str(dec.get("mode") or "").strip().lower() or None
        cognition_actions = [a for a in _safe_list(dec.get("next_3_actions")) if isinstance(a, dict)][:3]
    except Exception:
        cognition_mode = None
        cognition_actions = []

    # ---- Ritmo temporal ----
    cadence = {
        "daily": ["energy", "clarity"],
        "weekly": ["alignment", "training_readiness", "visual_presence"],
        "monthly": ["biological_consistency_index"],
        "quarterly": ["consolidation_review"],
    }

    # ---- Resumen corto para narración (n8n/LLM) ----
    protocol_summary = {
        "evolution_state": state,
        "primary_focus": focus.get("primary"),
        "week_goal": focus.get("week_goal"),
        "restrictions": [r.get("title") for r in restrictions][:2],
        "indices": indices,
        "cognition_mode": cognition_mode,
    }

    out = {
        "success": True,
        "engine": {"name": "gtg-protocolo-cognitivo-corporal", "version": PROTOCOL_VERSION},
        "inputs": {
            "week_id": week_id,
            "locale": locale,
            "goal_type": user_profile.get("goal_type") if isinstance(user_profile, dict) else None,
            "activity_level": user_profile.get("activity_level") if isinstance(user_profile, dict) else None,
        },
        "dimensions": dimensions,
        "indices": indices,
        "evolution_state": state,
        "restrictions": restrictions,
        "focus": focus,
        "cadence": cadence,
        "summary": protocol_summary,
        "meta": {
            "updated_at": _now_iso(),
            "state_streak": int(streak),
        },
        "sources": {
            "persisted": {
                "posture_proportion": bool(pp_res),
                "shape_presence": bool(shape_res),
                "lifestyle": bool(lifestyle_res),
                "progression": bool(progression_res),
                "skin": bool(skin_res),
            }
        },
    }

    return ProtocolOutput(payload=out)
