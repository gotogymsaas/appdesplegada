from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
import math


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


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    """Ángulo ABC en grados."""
    try:
        bax = float(a[0]) - float(b[0])
        bay = float(a[1]) - float(b[1])
        bcx = float(c[0]) - float(b[0])
        bcy = float(c[1]) - float(b[1])
        ba = math.hypot(bax, bay)
        bc = math.hypot(bcx, bcy)
        if ba <= 1e-9 or bc <= 1e-9:
            return None
        dot = bax * bcx + bay * bcy
        cosv = max(-1.0, min(1.0, float(dot) / float(ba * bc)))
        return float(math.degrees(math.acos(cosv)))
    except Exception:
        return None


def _get_xy(kp: dict[str, dict[str, float]], name: str) -> tuple[float, float] | None:
    k = kp.get(name)
    if not k:
        return None
    return (float(k["x"]), float(k["y"]))


def _group_score_from_ratio(r: float | None, *, lo: float, hi: float) -> int:
    if r is None:
        return 0
    if hi <= lo:
        return 0
    return int(round(_clamp((float(r) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0) * 100.0))


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
    focus = str(payload.get("focus") or "").strip().lower() or None
    height_cm = _safe_float(payload.get("height_cm"))

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

    def _body_height_px(kp: dict[str, dict[str, float]]) -> float | None:
        # Preferido: nose -> mid_ankle para aproximar altura visible
        nose = _get_xy(kp, "nose")
        la = _get_xy(kp, "left_ankle")
        ra = _get_xy(kp, "right_ankle")
        if nose and la and ra:
            mid_a = ((la[0] + ra[0]) / 2.0, (la[1] + ra[1]) / 2.0)
            h = _dist(nose, mid_a)
            return float(h) if h > 1e-6 else None
        # Fallback: torso * ratio antropométrico (muy aproximado)
        torso = _scale_from_front(kp)
        if torso and torso > 1e-6:
            # torso (hombro->cadera) suele ser ~0.27..0.32 de la estatura.
            return float(torso / 0.29)
        return None

    def _cm_per_px(kp: dict[str, dict[str, float]]) -> float | None:
        if height_cm is None or height_cm <= 0:
            return None
        bh = _body_height_px(kp)
        if not bh or bh <= 1e-6:
            return None
        return float(height_cm) / float(bh)

    # Métricas por vista (solo cuando estén disponibles)
    symmetry_score = None
    v_taper_score = None
    upper_lower_score = None
    posture_score = None
    definition_score = None

    # Proxies por grupo (0..100). Son *relativos* y dependen de encuadre/pose/luz.
    arms_score = None
    back_score = None
    glutes_score = None
    thigh_score = None

    insights: list[str] = []
    actions: list[str] = []

    # 1) Front (relaxed o flex) para hombros/cadera
    front_view = None
    for cand in ("front_relaxed", "front_flex"):
        if cand in views_in:
            front_view = cand
            break

    # Mediciones lineales (px y cm estimados cuando hay estatura)
    linear_px: dict[str, float] = {}
    linear_cm: dict[str, float] = {}
    cm_confidence = 0

    if front_view:
        kp = get_kp(front_view)
        scale = _scale_from_front(kp) or 1.0
        cpp = _cm_per_px(kp)
        if cpp is not None:
            # confianza de cm depende de consistencia y keypoints (muy aproximado)
            base_q = (sum(quality_scores) / max(1, len(quality_scores))) if quality_scores else 0.0
            cm_confidence = int(round(_clamp01(float(base_q) * 0.7 + _clamp01(n_views / 4.0) * 0.3) * 100.0))

        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        nose = kp.get("nose")

        if ls and rs and lh and rh:
            shoulder_w = abs(float(ls["x"]) - float(rs["x"]))
            hip_w = abs(float(lh["x"]) - float(rh["x"]))

            linear_px["shoulder_width"] = float(shoulder_w)
            linear_px["hip_width"] = float(hip_w)
            if cpp is not None:
                linear_cm["shoulder_width"] = round(float(shoulder_w) * float(cpp), 1)
                linear_cm["hip_width"] = round(float(hip_w) * float(cpp), 1)
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

            if leg_len and leg_len > 1e-6:
                linear_px["leg_length"] = float(leg_len)
                if cpp is not None:
                    linear_cm["leg_length"] = round(float(leg_len) * float(cpp), 1)

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

            # Grupo: glúteos (proxy) usando cadera vs hombros (no es proyección real, solo relación visual).
            try:
                if shoulder_w > 1e-6 and hip_w > 1e-6:
                    hip_over_sh = float(hip_w / shoulder_w)
                    glutes_score = _group_score_from_ratio(hip_over_sh, lo=0.72, hi=0.98)
            except Exception:
                pass

            # Grupo: piernas (proxy) usando longitud pierna vs torso (más estabilidad de medición que "volumen").
            try:
                if leg_len and leg_len > 1e-6:
                    leg_over_torso = float(leg_len / max(1e-6, scale))
                    thigh_score = _group_score_from_ratio(leg_over_torso, lo=1.55, hi=2.15)
            except Exception:
                pass

            # Grupo: espalda (proxy) mezclando V-taper + postura.
            try:
                if v_taper_score is not None and posture_score is not None:
                    back_score = int(round(_clamp((0.65 * (v_taper_score / 100.0) + 0.35 * (posture_score / 100.0)), 0.0, 1.0) * 100.0))
                elif v_taper_score is not None:
                    back_score = int(v_taper_score)
            except Exception:
                pass

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

                linear_px["forward_head_ratio"] = float(forward_head)
                # Ajustamos posture_score si estaba vacío
                if posture_score is None:
                    posture_score = _score_from_delta(0.6 * forward_head + 0.4 * rounded, good_at=0.12, bad_at=0.35)
                else:
                    posture_score = int(round(_clamp(float(posture_score) / 100.0 * 0.75 + _score_from_delta(0.6 * forward_head + 0.4 * rounded, good_at=0.12, bad_at=0.35) / 100.0 * 0.25, 0.0, 1.0) * 100.0))

                if forward_head >= 0.22:
                    insights.append("En perfil se sugiere cabeza adelantada. Un par de ajustes diarios mejoran mucho cómo te ves y cómo te sientes.")
                    actions.append("Haz 2×10 chin-tucks + estiramiento de pectoral 2 min/día.")

    # 2.b) Brazos / bíceps (proxy) usando el ángulo de codo en "front_flex" si existe.
    if "front_flex" in views_in:
        kp = get_kp("front_flex")
        # Longitud brazo (hombro->muñeca) como referencia para seguimiento
        cpp = _cm_per_px(kp)
        # Un codo más flexionado (ángulo menor) sugiere mejor ejecución de la pose.
        lsh = _get_xy(kp, "left_shoulder")
        lel = _get_xy(kp, "left_elbow")
        lwr = _get_xy(kp, "left_wrist")
        rsh = _get_xy(kp, "right_shoulder")
        rel = _get_xy(kp, "right_elbow")
        rwr = _get_xy(kp, "right_wrist")
        angles = []
        arm_lens = []
        for a, b, c in ((lsh, lel, lwr), (rsh, rel, rwr)):
            if a and b and c:
                ang = _angle(a, b, c)
                if ang is not None:
                    angles.append(float(ang))
                arm_lens.append(_dist(a, c))
        if angles:
            ang_avg = sum(angles) / max(1, len(angles))
            # Heurística: 60° = buena flex, 150° = casi recto.
            arms_score = int(round(_clamp((150.0 - ang_avg) / (150.0 - 60.0), 0.0, 1.0) * 100.0))
        if arm_lens:
            arm_len = float(sum(arm_lens) / max(1, len(arm_lens)))
            linear_px["arm_length"] = float(arm_len)
            if cpp is not None:
                linear_cm["arm_length"] = round(float(arm_len) * float(cpp), 1)

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

    # Progreso vs baseline (semana pasada) si existe.
    vs_last_week = None
    try:
        if isinstance(baseline, dict):
            bvars = baseline.get("variables") if isinstance(baseline.get("variables"), dict) else {}
            blinear_cm = baseline.get("measurements") if isinstance(baseline.get("measurements"), dict) else {}
            blinear_cm = blinear_cm.get("cm_est") if isinstance(blinear_cm.get("cm_est"), dict) else {}
            deltas = {}
            for key, cur in (
                ("symmetry", symmetry_score),
                ("v_taper", v_taper_score),
                ("upper_lower_balance", upper_lower_score),
                ("static_posture", posture_score),
                ("definition", definition_score),
                ("measurement_consistency", measurement_consistency),
            ):
                try:
                    prev = int(bvars.get(key) or 0)
                    cur_i = int(cur) if cur is not None else 0
                    deltas[key] = {"prev": prev, "now": cur_i, "delta": int(cur_i - prev)}
                except Exception:
                    continue
            if deltas:
                vs_last_week = {"available": True, "deltas": deltas}

            # Deltas para cm estimados (si existen)
            cm_deltas = {}
            for k, cur_v in linear_cm.items():
                try:
                    prev_v = float(blinear_cm.get(k)) if blinear_cm.get(k) is not None else None
                except Exception:
                    prev_v = None
                if prev_v is None:
                    continue
                try:
                    cur_f = float(cur_v)
                    delta = float(cur_f - float(prev_v))
                    pct = (delta / float(prev_v) * 100.0) if abs(float(prev_v)) > 1e-6 else None
                    cm_deltas[k] = {
                        "prev": round(float(prev_v), 1),
                        "now": round(float(cur_f), 1),
                        "delta": round(float(delta), 1),
                        "pct": round(float(pct), 2) if pct is not None else None,
                    }
                except Exception:
                    continue
            if cm_deltas:
                if vs_last_week is None:
                    vs_last_week = {"available": True, "deltas": {}}
                vs_last_week["cm_deltas"] = cm_deltas
    except Exception:
        vs_last_week = None

    payload_out = {
        "decision": decision,
        "decision_reason": decision_reason,
        "focus": focus,
        "progress": {"vs_last_week": vs_last_week},
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
                "back": _nz(back_score),
                "glutes": _nz(glutes_score),
                "thigh": _nz(thigh_score),
                "arms": _nz(arms_score),
            },
            "v_taper": _nz(v_taper_score),
            "upper_lower_balance": _nz(upper_lower_score),
            "static_posture": _nz(posture_score),
            "definition": _nz(definition_score),
            "measurement_consistency": int(measurement_consistency),
        },
        "measurements": {
            "px": {k: round(float(v), 4) for k, v in linear_px.items()},
            "cm_est": {k: float(v) for k, v in linear_cm.items()},
            "cm_confidence": int(cm_confidence),
            "note": "cm_est son estimaciones basadas en estatura + encuadre. Úsalas como tendencia, no como cinta métrica.",
        },
        "muscles_recognized": [
            "hombros (proxy V-taper)",
            "espalda (proxy por V-taper + postura)",
            "brazos/bíceps (proxy en frente flex suave)",
            "glúteos/cadera (proxy por relación cadera-hombros)",
            "pierna (proxy por relación pierna-torso)",
        ],
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

    user_display_name = str(result.get("user_display_name") or "").strip()
    hello = f"Hola {user_display_name}," if user_display_name else "Hola,"

    decision = str(result.get("decision") or "").strip().lower()
    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    vars_ = result.get("variables") if isinstance(result.get("variables"), dict) else {}
    vol = vars_.get("volume_by_group") if isinstance(vars_.get("volume_by_group"), dict) else {}
    focus = str(result.get("focus") or "").strip().lower() or None
    meas = result.get("measurements") if isinstance(result.get("measurements"), dict) else {}
    cm_est = meas.get("cm_est") if isinstance(meas.get("cm_est"), dict) else {}
    cm_conf = None
    try:
        cm_conf = int(meas.get("cm_confidence")) if meas.get("cm_confidence") is not None else None
    except Exception:
        cm_conf = None

    confidence_score = None
    try:
        confidence_score = float(conf.get("score")) if conf.get("score") is not None else None
    except Exception:
        confidence_score = None

    n_views = 0
    try:
        n_views = int(conf.get("n_views") or 0)
    except Exception:
        n_views = 0

    consistency = int(vars_.get("measurement_consistency") or 0)
    posture = int(vars_.get("static_posture") or 0)
    symmetry = int(vars_.get("symmetry") or 0)
    v_taper = int(vars_.get("v_taper") or 0)

    lines: list[str] = [hello]

    if decision != "accepted":
        lines.append(
            "No pude detectar bien tu cuerpo en esta foto (o la calidad fue baja)."
        )
        lines.append("Para que la medición sea útil semana a semana:")
        lines.append("- Cuerpo completo (pies a cabeza)")
        lines.append("- Buena luz + fondo limpio")
        lines.append("- Cámara a 2–3m, altura del pecho")
        lines.append("- Misma pose y encuadre cada semana")
        return "\n".join(lines).strip()

    lines.append("Perfecto. Ya tengo tu **Medición del progreso muscular** (comparación relativa; no promete cm exactos).")
    if confidence_score is not None:
        lines.append(f"Calidad de medición: {int(round(confidence_score * 100.0))}/100 (con {n_views} vista(s)).")

    lines.append("")
    lines.append("¿Qué estoy midiendo realmente?")
    lines.append("- Ratios y alineación (hombros/cadera, simetría, postura) a partir de keypoints 2D")
    lines.append("- Proxies por grupo (brazos/glúteos/espalda/pierna) que dependen de luz/pose/encuadre")
    lines.append("- Consistencia de medición para que la comparación semanal sea justa")

    lines.append("")
    lines.append("Resumen de hoy (0–100):")
    lines.append(f"- Consistencia de medición: {consistency}/100")
    lines.append(f"- Postura estática (proxy): {posture}/100")
    lines.append(f"- Simetría (hombros/pelvis): {symmetry}/100")
    lines.append(f"- Silueta V‑taper (proxy): {v_taper}/100")

    # Proxies por grupo
    def _g(name: str) -> int:
        try:
            return int(vol.get(name) or 0)
        except Exception:
            return 0

    lines.append("")
    lines.append("Proxies por grupo (0–100):")
    lines.append(f"- Brazos/bíceps (pose flex): {_g('arms')}/100")
    lines.append(f"- Espalda: {_g('back')}/100")
    lines.append(f"- Glúteos/cadera: {_g('glutes')}/100")
    lines.append(f"- Pierna: {_g('thigh')}/100")

    # cm estimados (si existen)
    try:
        if cm_est:
            lines.append("")
            head = "Medidas en cm (estimadas)" + (f" — confiabilidad {cm_conf}/100" if cm_conf is not None else "") + ":"
            lines.append(head)

            def _cm_line(key: str, label: str) -> str | None:
                v = cm_est.get(key)
                if v is None:
                    return None
                try:
                    return f"- {label}: {float(v):.1f} cm"
                except Exception:
                    return None

            for ln in (
                _cm_line("shoulder_width", "Ancho de hombros"),
                _cm_line("hip_width", "Ancho de cadera"),
                _cm_line("arm_length", "Largo de brazo (hombro→muñeca)"),
                _cm_line("leg_length", "Largo de pierna (cadera→tobillo)"),
            ):
                if ln:
                    lines.append(ln)
            lines.append("Nota: úsalo como tendencia semanal, no como cinta métrica.")
    except Exception:
        pass

    # Comparación vs semana pasada
    try:
        prog = result.get("progress") if isinstance(result.get("progress"), dict) else {}
        vs = prog.get("vs_last_week") if isinstance(prog.get("vs_last_week"), dict) else None
        if vs and isinstance(vs.get("deltas"), dict):
            deltas = vs.get("deltas")
            def _fmt_delta(k: str, label: str) -> str | None:
                d = deltas.get(k) if isinstance(deltas.get(k), dict) else None
                if not d:
                    return None
                try:
                    delta = int(d.get("delta") or 0)
                    sign = "+" if delta >= 0 else ""
                    return f"- {label}: {sign}{delta} pts"
                except Exception:
                    return None

            parts = [
                _fmt_delta("symmetry", "Simetría"),
                _fmt_delta("v_taper", "V‑taper"),
                _fmt_delta("static_posture", "Postura"),
                _fmt_delta("measurement_consistency", "Consistencia"),
            ]
            parts = [p for p in parts if p]
            if parts:
                lines.append("")
                lines.append("Cambios vs tu semana pasada:")
                lines.extend(parts[:4])

            # cm deltas (si existieran)
            try:
                cm_d = vs.get("cm_deltas") if isinstance(vs.get("cm_deltas"), dict) else {}
                if cm_d:
                    lines.append("\nCambios en cm (estimados):")
                    for k, row in list(cm_d.items())[:4]:
                        if not isinstance(row, dict):
                            continue
                        label_map = {
                            "shoulder_width": "Hombros",
                            "hip_width": "Cadera",
                            "arm_length": "Brazo",
                            "leg_length": "Pierna",
                        }
                        lab = label_map.get(str(k), str(k))
                        try:
                            delta = float(row.get("delta") or 0.0)
                            sign = "+" if delta >= 0 else ""
                            pct = row.get("pct")
                            if pct is not None:
                                lines.append(f"- {lab}: {sign}{delta:.1f} cm ({sign}{float(pct):.2f}%)")
                            else:
                                lines.append(f"- {lab}: {sign}{delta:.1f} cm")
                        except Exception:
                            continue
            except Exception:
                pass
    except Exception:
        pass

    # Qué hacer esta semana (marketing + accionable)
    lines.append("")
    lines.append("Qué deberías hacer esta semana (para que el progreso se note y se mida mejor):")
    lines.append("1) **Fotos de referencia (recomendado):** frente relajado + perfil derecho. Extra (mejor): espalda + frente flex suave.")
    lines.append("2) **Mismo protocolo:** misma luz, misma distancia, misma altura de cámara, misma ropa/pose.")
    lines.append("3) Si es selfie en espejo: usa temporizador y aléjate (2–3m). Evita taparte con el celular.")
    lines.append("4) Si te toman la foto: cámara a la altura del pecho, centrado, sin gran angular.")
    lines.append("5) Si vas a **centralizar el músculo** (enfoque): perfecto, pero no recortes los puntos clave (hombros/cadera/codos/rodillas) o baja la confiabilidad.")
    lines.append("6) En entrenamiento: progresión simple (más reps o más carga) en 2–3 ejercicios clave.")

    # Ajuste por foco
    if focus in ("biceps", "bíceps", "bicep"):
        lines.append("")
        lines.append("Enfoque: Bíceps (brazo más lleno y marcado)")
        lines.append("- 2 días/semana: 6–10 series totales de curl (mancuerna/barra/polea)")
        lines.append("- 1 ejercicio de tirón (remo/pull‑down) para soporte de espalda y brazo")
        lines.append("- Rango 8–15 reps, dejando 1–2 reps en reserva")
    elif focus in ("glutes", "gluteos", "glúteos"):
        lines.append("")
        lines.append("Enfoque: Glúteos (más forma y potencia)")
        lines.append("- 2 días/semana: hip thrust o puente pesado + RDL (bisagra) + zancada")
        lines.append("- 8–12 series efectivas/semana (sube 1–2 series si recuperas bien)")
        lines.append("- Pausa de 1s arriba en thrust para sentir activación")

    # Insights del motor
    insights = result.get("insights")
    if isinstance(insights, list) and insights:
        lines.append("")
        lines.append("Notas rápidas:")
        for x in insights[:3]:
            if str(x).strip():
                lines.append(f"- {str(x).strip()}")

    lines.append("")
    lines.append("Si sientes dolor fuerte o mareo, baja intensidad o detente. Progreso sí; lesión no.")
    return "\n".join(lines).strip()
