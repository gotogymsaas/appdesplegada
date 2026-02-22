from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class MuscleMeasureResult:
    payload: dict[str, Any]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _kp_map(pose: dict[str, Any]) -> dict[str, dict[str, float]]:
    kps = pose.get("keypoints") if isinstance(pose.get("keypoints"), list) else []
    out: dict[str, dict[str, float]] = {}
    for kp in kps:
        if not isinstance(kp, dict):
            continue
        name = str(kp.get("name") or "").strip()
        x = _safe_float(kp.get("x"))
        y = _safe_float(kp.get("y"))
        s = _safe_float(kp.get("score"))
        if not name or x is None or y is None:
            continue
        out[name] = {"x": float(x), "y": float(y), "score": float(s or 0.0)}
    return out


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float((dx * dx + dy * dy) ** 0.5)


def _mid(kp: dict[str, dict[str, float]], a: str, b: str) -> tuple[tuple[float, float] | None, float]:
    ka = kp.get(a)
    kb = kp.get(b)
    if not ka or not kb:
        return None, 0.0
    p = ((float(ka["x"]) + float(kb["x"])) / 2.0, (float(ka["y"]) + float(kb["y"])) / 2.0)
    conf = min(float(ka.get("score") or 0.0), float(kb.get("score") or 0.0))
    return p, float(conf)


def _pose_quality_score(kp: dict[str, dict[str, float]], required: list[str]) -> tuple[float, list[str]]:
    missing: list[str] = []
    confs: list[float] = []
    present = 0
    for r in required:
        if r in kp:
            present += 1
            confs.append(float(kp[r].get("score") or 0.0))
        else:
            missing.append(r)
    ratio = present / max(1, len(required))
    avg_conf = (sum(confs) / max(1, len(confs))) if confs else 0.0
    score = _clamp01((0.65 * ratio) + (0.35 * avg_conf))
    return float(score), missing


def _score_from_delta(delta: float, *, good_at: float, bad_at: float) -> int:
    # delta: 0..+ (ideal 0). A good_at => score ~100. A bad_at => score ~0.
    if delta <= good_at:
        return 100
    if delta >= bad_at:
        return 0
    t = (delta - good_at) / max(1e-9, (bad_at - good_at))
    return int(round(100.0 * (1.0 - float(t))))


def evaluate_muscle_measure(payload: dict[str, Any]) -> MuscleMeasureResult:
    """Exp-010 (MVP): medición muscular *relativa* usando keypoints 2D.

    Entrada esperada:
      {"poses": {view_id: {keypoints: [...], image: {width,height}}}, "baseline": {...}?}

    - Acepta 1..4 vistas.
    - No promete cm reales (solo ratios y consistencia).
    """

    poses = payload.get("poses") if isinstance(payload.get("poses"), dict) else {}
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else None

    # Views soportadas (opcionales)
    allowed_views = ("front_relaxed", "side_right_relaxed", "back_relaxed", "front_flex")
    views_in = {k: v for k, v in poses.items() if isinstance(k, str) and k in allowed_views and isinstance(v, dict)}

    required_front = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose"]
    required_side = ["right_ear", "right_shoulder", "right_hip"]

    view_metrics: dict[str, dict[str, Any]] = {}
    quality_scores: list[float] = []

    def get_kp(view_id: str) -> dict[str, dict[str, float]]:
        pose = views_in.get(view_id) if isinstance(views_in.get(view_id), dict) else {}
        return _kp_map(pose)

    # Calculamos calidad por vista
    for view_id, pose in views_in.items():
        kp = _kp_map(pose)
        req = required_side if view_id == "side_right_relaxed" else required_front
        q, missing = _pose_quality_score(kp, req)
        view_metrics[view_id] = {"pose_quality": round(float(q), 4), "missing": missing}
        quality_scores.append(float(q))

    n_views = len(views_in)

    # Condición mínima: al menos 1 vista con calidad aceptable
    ok_views = [k for k, m in view_metrics.items() if float(m.get("pose_quality") or 0.0) >= 0.55]

    decision = "accepted" if ok_views else "needs_confirmation"
    decision_reason = "ok" if ok_views else "missing_or_low_quality_pose"

    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted":
        follow_up_questions.append(
            {
                "type": "retake_photos",
                "prompt": "No pude detectar bien tu cuerpo. Intenta con mejor luz y el cuerpo completo (pies a cabeza).",
                "options": [],
            }
        )

    # Escala interna: usamos torso (hombros→cadera) cuando está disponible para normalizar deltas.
    def _scale_from_front(kp: dict[str, dict[str, float]]) -> float | None:
        mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
        mid_hip, _ = _mid(kp, "left_hip", "right_hip")
        if not mid_sh or not mid_hip:
            return None
        s = _dist(mid_sh, mid_hip)
        return float(s) if s > 1e-6 else None

    # Métricas por vista (solo cuando estén disponibles)
    symmetry_score = None
    v_taper_score = None
    upper_lower_score = None
    posture_score = None
    definition_score = None

    insights: list[str] = []
    actions: list[str] = []

    # 1) Front (relaxed o flex) para hombros/cadera
    front_view = None
    for cand in ("front_relaxed", "front_flex"):
        if cand in views_in:
            front_view = cand
            break

    if front_view:
        kp = get_kp(front_view)
        scale = _scale_from_front(kp) or 1.0

        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        nose = kp.get("nose")

        if ls and rs and lh and rh:
            shoulder_w = abs(float(ls["x"]) - float(rs["x"]))
            hip_w = abs(float(lh["x"]) - float(rh["x"]))
            shoulder_level = abs(float(ls["y"]) - float(rs["y"])) / scale
            hip_level = abs(float(lh["y"]) - float(rh["y"])) / scale

            # Simetría bilateral (proxy): hombros + pelvis
            delta_sym = float(0.6 * shoulder_level + 0.4 * hip_level)
            symmetry_score = _score_from_delta(delta_sym, good_at=0.03, bad_at=0.12)

            # V-taper (proxy): hombros vs cadera
            ratio = (shoulder_w / hip_w) if hip_w > 1e-6 else None
            if ratio is not None:
                # 1.05 bajo, 1.25 alto (heurística)
                v_taper_score = int(round(_clamp((ratio - 1.05) / (1.25 - 1.05), 0.0, 1.0) * 100.0))

            # Balance sup/inf (proxy): ancho torso vs largo pierna
            lk = kp.get("left_knee")
            rk = kp.get("right_knee")
            la = kp.get("left_ankle")
            ra = kp.get("right_ankle")
            leg_len = None
            if lh and rh and lk and rk and la and ra:
                lhip = (float(lh["x"]), float(lh["y"]))
                rhip = (float(rh["x"]), float(rh["y"]))
                lk_p = (float(lk["x"]), float(lk["y"]))
                rk_p = (float(rk["x"]), float(rk["y"]))
                la_p = (float(la["x"]), float(la["y"]))
                ra_p = (float(ra["x"]), float(ra["y"]))
                l_leg = _dist(lhip, lk_p) + _dist(lk_p, la_p)
                r_leg = _dist(rhip, rk_p) + _dist(rk_p, ra_p)
                leg_len = float((l_leg + r_leg) / 2.0)

            if leg_len and leg_len > 1e-6 and ratio is not None:
                upper_lower = (ratio / (leg_len / max(1e-6, scale)))
                # normalizamos heurístico a 0..100
                upper_lower_score = int(round(_clamp((upper_lower - 2.0) / (3.6 - 2.0), 0.0, 1.0) * 100.0))

            # Postura estática (proxy): hombros/pelvis + cabeza centrada
            head_offset = None
            if nose:
                mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
                if mid_sh:
                    head_offset = abs(float(nose["x"]) - float(mid_sh[0])) / max(1e-6, shoulder_w)
            delta_post = float(0.5 * shoulder_level + 0.4 * hip_level + (0.1 * float(head_offset or 0.0)))
            posture_score = _score_from_delta(delta_post, good_at=0.03, bad_at=0.14)

            # Insights tipo coach
            if symmetry_score is not None and symmetry_score < 70:
                insights.append("Veo asimetría leve en hombros/pelvis. Puede ser postura o distribución de carga; lo trabajamos con unilateral y movilidad.")
                actions.append("Esta semana prioriza 2 ejercicios unilaterales (zancadas, press con mancuerna) con control.")

            if v_taper_score is not None:
                insights.append(f"Tu silueta (proxy V-taper) está en {v_taper_score}/100. Si quieres mejorarlo: espalda alta + deltoides + control de cintura.")
                actions.append("Añade 6–10 series semanales de espalda alta (remos/pull) y deltoides lateral.")

    # 2) Side view: forward head / rounded shoulders (muy básico)
    if "side_right_relaxed" in views_in:
        kp = get_kp("side_right_relaxed")
        ear = kp.get("right_ear")
        sh = kp.get("right_shoulder")
        hip = kp.get("right_hip")
        if ear and sh and hip:
            s = _dist((float(sh["x"]), float(sh["y"])), (float(hip["x"]), float(hip["y"])))
            if s > 1e-6:
                forward_head = abs(float(ear["x"]) - float(sh["x"])) / s
                rounded = abs(float(sh["x"]) - float(hip["x"])) / s
                # Ajustamos posture_score si estaba vacío
                if posture_score is None:
                    posture_score = _score_from_delta(0.6 * forward_head + 0.4 * rounded, good_at=0.12, bad_at=0.35)
                else:
                    posture_score = int(round(_clamp(float(posture_score) / 100.0 * 0.75 + _score_from_delta(0.6 * forward_head + 0.4 * rounded, good_at=0.12, bad_at=0.35) / 100.0 * 0.25, 0.0, 1.0) * 100.0))

                if forward_head >= 0.22:
                    insights.append("En perfil se sugiere cabeza adelantada. Un par de ajustes diarios mejoran mucho cómo te ves y cómo te sientes.")
                    actions.append("Haz 2×10 chin-tucks + estiramiento de pectoral 2 min/día.")

    # 3) Definición visual: MVP no fiable sin control de luz. Devolvemos score conservador basado en calidad.
    if quality_scores:
        qavg = sum(quality_scores) / max(1, len(quality_scores))
        definition_score = int(round(_clamp(qavg, 0.0, 1.0) * 100.0))

    # 4) Consistencia de medición (variable #7)
    base_cons = 0.0
    base_cons += 0.45 * _clamp01(sum(quality_scores) / max(1, len(quality_scores)) if quality_scores else 0.0)
    base_cons += 0.35 * _clamp01(n_views / 4.0)
    if baseline:
        base_cons += 0.20
    measurement_consistency = int(round(_clamp01(base_cons) * 100.0))

    # Confidence global
    conf = 1.0
    if decision != "accepted":
        conf *= 0.45
    conf *= _clamp01(sum(quality_scores) / max(1, len(quality_scores)) if quality_scores else 0.0)
    conf *= _clamp01(0.6 + 0.4 * (n_views / 4.0))
    confidence = _clamp01(conf)
    uncertainty = _clamp01(1.0 - confidence)

    # Defaults
    def _nz(v):
        return int(v) if v is not None else 0

    payload_out = {
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": {
            "score": round(float(confidence), 4),
            "uncertainty_score": round(float(uncertainty), 4),
            "n_views": n_views,
            "views": view_metrics,
        },
        "variables": {
            "symmetry": _nz(symmetry_score),
            "volume_by_group": {
                "shoulders": _nz(v_taper_score),
                "back": 0,
                "glutes": 0,
                "thigh": 0,
                "arms": 0,
            },
            "v_taper": _nz(v_taper_score),
            "upper_lower_balance": _nz(upper_lower_score),
            "static_posture": _nz(posture_score),
            "definition": _nz(definition_score),
            "measurement_consistency": int(measurement_consistency),
        },
        "insights": (insights[:3] if insights else ["Medición lista. Si quieres más precisión, agrega 2–4 vistas en la misma luz y encuadre."])
        ,
        "recommended_actions": (actions[:3] if actions else ["Repite la medición con la misma luz/encuadre para comparar semana a semana."]),
        "follow_up_questions": follow_up_questions,
        "meta": {"algorithm": "exp-010_muscle_measure_v0", "as_of": str(date.today())},
    }

    return MuscleMeasureResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    lines: list[str] = []
    lines.append(f"decision: {result.get('decision')}")

    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    if conf.get("score") is not None:
        try:
            lines.append(f"confidence: {round(float(conf.get('score')), 3)}")
        except Exception:
            pass

    vars_ = result.get("variables") if isinstance(result.get("variables"), dict) else {}
    try:
        lines.append(f"simetría: {int(vars_.get('symmetry') or 0)}/100")
        lines.append(f"v-taper: {int(vars_.get('v_taper') or 0)}/100")
        lines.append(f"postura (proxy): {int(vars_.get('static_posture') or 0)}/100")
        lines.append(f"consistencia medición: {int(vars_.get('measurement_consistency') or 0)}/100")
    except Exception:
        pass

    insights = result.get("insights")
    if isinstance(insights, list) and insights:
        for x in insights[:3]:
            if str(x).strip():
                lines.append(f"insight: {str(x).strip()}")

    return "\n".join(lines).strip()
