from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class PostureProportionResult:
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
    # delta: 0..+ (ideal 0). A good_at => ~100. A bad_at => ~0.
    if delta <= good_at:
        return 100
    if delta >= bad_at:
        return 0
    t = (delta - good_at) / max(1e-9, (bad_at - good_at))
    return int(round(100.0 * (1.0 - float(t))))


def _score_from_ratio(value: float, *, lo: float, hi: float) -> int:
    if value <= lo:
        return 0
    if value >= hi:
        return 100
    t = (value - lo) / max(1e-9, (hi - lo))
    return int(round(_clamp01(t) * 100.0))


def evaluate_posture_proportion(payload: dict[str, Any]) -> PostureProportionResult:
    """Exp-013: Posture & Proportion Intelligence™ (MVP).

    Input esperado:
      {"poses": {view_id: {keypoints:[...], image:{width,height}}}, "baseline"?: {...}}

    Views:
      - front_relaxed (obligatoria)
      - side_right_relaxed (obligatoria)
      - back_relaxed (opcional)

    Salida:
      - Posture Score (A-score) 0..100
      - Proportion Score (P-score) 0..100 (proxy por keypoints)
      - Alignment & Silhouette Index (unificado)
      - 2 correcciones inmediatas + 1 ajuste semanal

    Guardrails:
      - No cm reales
      - No diagnóstico médico
      - No juicios estéticos
    """

    poses = payload.get("poses") if isinstance(payload.get("poses"), dict) else {}
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else None

    allowed_views = ("front_relaxed", "side_right_relaxed", "back_relaxed")
    views_in = {k: v for k, v in poses.items() if isinstance(k, str) and k in allowed_views and isinstance(v, dict)}

    required_front = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "nose",
    ]
    required_side = ["right_ear", "right_shoulder", "right_hip"]
    required_back = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]

    view_metrics: dict[str, dict[str, Any]] = {}
    quality_scores: list[float] = []

    def get_kp(view_id: str) -> dict[str, dict[str, float]]:
        pose = views_in.get(view_id) if isinstance(views_in.get(view_id), dict) else {}
        return _kp_map(pose)

    for view_id, pose in views_in.items():
        kp = _kp_map(pose)
        if view_id == "side_right_relaxed":
            req = required_side
        elif view_id == "back_relaxed":
            req = required_back
        else:
            req = required_front
        q, missing = _pose_quality_score(kp, req)
        view_metrics[view_id] = {"pose_quality": round(float(q), 4), "missing": missing}
        quality_scores.append(float(q))

    n_views = len(views_in)

    # Reglas mínimas: front + side con calidad aceptable
    q_front = float(view_metrics.get("front_relaxed", {}).get("pose_quality") or 0.0)
    q_side = float(view_metrics.get("side_right_relaxed", {}).get("pose_quality") or 0.0)

    decision = "accepted" if (q_front >= 0.55 and q_side >= 0.55) else "needs_confirmation"
    decision_reason = "ok" if decision == "accepted" else "missing_or_low_quality_pose"

    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted":
        follow_up_questions.append(
            {
                "type": "retake_photos",
                "prompt": "Necesito 2 fotos con cuerpo completo: frontal y perfil (buena luz, cámara a la altura del pecho, 2–3m).",
                "options": [],
            }
        )

    # Escala interna: torso (hombros→cadera) desde front
    def _scale_from_front(kp: dict[str, dict[str, float]]) -> float | None:
        mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
        mid_hip, _ = _mid(kp, "left_hip", "right_hip")
        if not mid_sh or not mid_hip:
            return None
        s = _dist(mid_sh, mid_hip)
        return float(s) if s > 1e-6 else None

    # --- A-score (postura) ---
    shoulder_level_score = None
    pelvis_level_score = None
    base_axis_score = None
    forward_head_score = None
    rounded_shoulders_score = None

    # --- P-score (proporción proxy) ---
    v_taper_score = None
    torso_leg_score = None
    stance_grounding_score = None

    insights: list[str] = []
    actions_now: list[dict[str, Any]] = []
    weekly_adjustment: dict[str, Any] | None = None

    # FRONT metrics
    if "front_relaxed" in views_in:
        kp = get_kp("front_relaxed")
        scale = _scale_from_front(kp) or 1.0

        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        la = kp.get("left_ankle")
        ra = kp.get("right_ankle")

        if ls and rs:
            shoulder_level = abs(float(ls["y"]) - float(rs["y"])) / scale
            shoulder_level_score = _score_from_delta(float(shoulder_level), good_at=0.03, bad_at=0.12)

        if lh and rh:
            pelvis_level = abs(float(lh["y"]) - float(rh["y"])) / scale
            pelvis_level_score = _score_from_delta(float(pelvis_level), good_at=0.03, bad_at=0.12)

        # Base axis (rodilla→tobillo) proxy: diferencias de x entre rodilla y tobillo por lado
        lk = kp.get("left_knee")
        rk = kp.get("right_knee")
        if lk and rk and la and ra:
            l = abs(float(lk["x"]) - float(la["x"]))
            r = abs(float(rk["x"]) - float(ra["x"]))
            base_axis = float(0.5 * (l + r))
            base_axis_score = _score_from_delta(base_axis, good_at=0.03, bad_at=0.12)

        # Proportion proxies
        if ls and rs and lh and rh:
            shoulder_w = abs(float(ls["x"]) - float(rs["x"]))
            hip_w = abs(float(lh["x"]) - float(rh["x"]))
            if hip_w > 1e-6:
                v_taper = shoulder_w / hip_w
                v_taper_score = _score_from_ratio(float(v_taper), lo=1.00, hi=1.25)

        if lh and rh and lk and rk and la and ra:
            mid_hip, _ = _mid(kp, "left_hip", "right_hip")
            mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
            if mid_hip and mid_sh:
                torso = _dist(mid_sh, mid_hip)
                lhip = (float(lh["x"]), float(lh["y"]))
                rhip = (float(rh["x"]), float(rh["y"]))
                lk_p = (float(lk["x"]), float(lk["y"]))
                rk_p = (float(rk["x"]), float(rk["y"]))
                la_p = (float(la["x"]), float(la["y"]))
                ra_p = (float(ra["x"]), float(ra["y"]))
                l_leg = _dist(lhip, lk_p) + _dist(lk_p, la_p)
                r_leg = _dist(rhip, rk_p) + _dist(rk_p, ra_p)
                leg = float((l_leg + r_leg) / 2.0)
                if leg > 1e-6:
                    tl = float(torso / leg)
                    torso_leg_score = _score_from_ratio(float(tl), lo=0.45, hi=0.75)

        if la and ra and ls and rs:
            shoulder_w = abs(float(ls["x"]) - float(rs["x"]))
            if shoulder_w > 1e-6:
                ankle_w = abs(float(la["x"]) - float(ra["x"]))
                stance = ankle_w / shoulder_w
                # ideal ~0.45..0.85
                if stance <= 0.45:
                    stance_grounding_score = _score_from_ratio(float(stance), lo=0.15, hi=0.45)
                elif stance >= 0.85:
                    stance_grounding_score = _score_from_ratio(float(1.15 - stance), lo=0.30, hi=0.85)
                else:
                    stance_grounding_score = int(round(_clamp01((stance - 0.45) / (0.85 - 0.45)) * 100.0))

    # SIDE metrics
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
                forward_head_score = _score_from_delta(float(forward_head), good_at=0.12, bad_at=0.35)
                rounded_shoulders_score = _score_from_delta(float(rounded), good_at=0.10, bad_at=0.30)

    # BACK metrics (opcional): simetría hombros/pelvis como corroboración
    back_symmetry_score = None
    if "back_relaxed" in views_in:
        kp = get_kp("back_relaxed")
        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        if ls and rs and lh and rh:
            # sin escala fiable atrás; usamos torso de front si existe
            scale = None
            if "front_relaxed" in views_in:
                scale = _scale_from_front(get_kp("front_relaxed"))
            scale = float(scale or 1.0)
            shoulder_level = abs(float(ls["y"]) - float(rs["y"])) / scale
            pelvis_level = abs(float(lh["y"]) - float(rh["y"])) / scale
            back_symmetry_score = _score_from_delta(float(0.6 * shoulder_level + 0.4 * pelvis_level), good_at=0.03, bad_at=0.12)

    # Agregación de scores
    def _nz(v: int | None) -> int:
        return int(v) if v is not None else 0

    a_parts = [x for x in [shoulder_level_score, pelvis_level_score, base_axis_score, forward_head_score, rounded_shoulders_score] if x is not None]
    posture_score = int(round(sum(a_parts) / max(1, len(a_parts)))) if a_parts else 0

    p_parts = [x for x in [v_taper_score, torso_leg_score, stance_grounding_score] if x is not None]
    proportion_score = int(round(sum(p_parts) / max(1, len(p_parts)))) if p_parts else 0

    # Unificado
    # 55% postura, 45% proporción (proxy)
    asi = int(round(0.55 * float(posture_score) + 0.45 * float(proportion_score)))

    # Baseline delta opcional
    baseline_delta = None
    try:
        if isinstance(baseline, dict):
            b_vars = baseline.get("variables") if isinstance(baseline.get("variables"), dict) else {}
            b_asi = int(b_vars.get("alignment_silhouette_index") or 0)
            baseline_delta = int(asi) - int(b_asi)
    except Exception:
        baseline_delta = None

    # Confidence global
    conf = 1.0
    if decision != "accepted":
        conf *= 0.45
    conf *= _clamp01(sum(quality_scores) / max(1, len(quality_scores)) if quality_scores else 0.0)
    # Penaliza si falta back (opcional) muy poco
    conf *= _clamp01(0.85 + 0.15 * (1.0 if "back_relaxed" in views_in else 0.0))
    confidence = _clamp01(conf)
    uncertainty = _clamp01(1.0 - confidence)

    # --- Patrones coach (reglas simples) ---
    pattern_keys: list[str] = []

    def add_pattern(key: str):
        if key and key not in pattern_keys:
            pattern_keys.append(key)

    if forward_head_score is not None and forward_head_score < 70:
        add_pattern("forward_head")
    if rounded_shoulders_score is not None and rounded_shoulders_score < 70:
        add_pattern("rounded_shoulders")
    if pelvis_level_score is not None and pelvis_level_score < 70:
        add_pattern("pelvis_imbalance")
    if base_axis_score is not None and base_axis_score < 70:
        add_pattern("base_axis")
    if v_taper_score is not None and v_taper_score < 55:
        add_pattern("low_v_taper_proxy")

    # --- Correcciones inmediatas (WOW) ---
    # Mantener siempre 2, pero ajustadas al patrón.
    def corr(ex_id: str, title: str, cue: str, duration_sec: int) -> dict[str, Any]:
        return {
            "exercise_id": ex_id,
            "title": title,
            "cue": cue,
            "duration_sec": int(duration_sec),
        }

    # Defaults
    c1 = corr("chin_tucks", "Chin tucks (retracción de mentón)", "Lleva la barbilla hacia atrás como haciendo ‘doble mentón’ sin inclinar la cabeza.", 45)
    c2 = corr("doorway_pec_stretch", "Estiramiento de pectoral", "Pecho abierto, costillas abajo; respira lento por la nariz.", 45)

    if "pelvis_imbalance" in pattern_keys or "base_axis" in pattern_keys:
        c2 = corr("dead_bug", "Dead bug (core profundo)", "Costillas abajo, exhala largo; controla sin arquear la zona lumbar.", 60)

    if "rounded_shoulders" in pattern_keys and "forward_head" not in pattern_keys:
        c1 = corr("scap_retractions", "Retracción escapular", "Hombros abajo y atrás (sin arquear), aprieta entre escápulas 2s.", 45)

    actions_now = [c1, c2]

    # Weekly adjustment (1)
    weekly_focus: list[str] = []
    if "rounded_shoulders" in pattern_keys or "low_v_taper_proxy" in pattern_keys:
        weekly_focus.extend(["espalda alta", "deltoides lateral"])
    if "pelvis_imbalance" in pattern_keys or "base_axis" in pattern_keys:
        weekly_focus.extend(["glúteo medio", "core anti-rotación"])
    if not weekly_focus:
        weekly_focus = ["movilidad torácica", "core"]

    weekly_adjustment = {
        "title": "Ajuste semanal",
        "focus": weekly_focus[:3],
        "note": "Sube el énfasis en estos grupos esta semana y baja 10–20% volumen en el grupo dominante si estás muy cargado.",
    }

    # Insights: 1 frase wow + soporte
    if decision == "accepted":
        headline_parts = []
        if "forward_head" in pattern_keys and "rounded_shoulders" in pattern_keys:
            headline_parts.append("Tu cuello está adelantado y los hombros tienden a rotar hacia adentro")
        elif "forward_head" in pattern_keys:
            headline_parts.append("Tu cuello tiende a adelantarse")
        elif "rounded_shoulders" in pattern_keys:
            headline_parts.append("Tus hombros tienden a redondearse")
        if "pelvis_imbalance" in pattern_keys:
            headline_parts.append("y la pelvis se ve algo desbalanceada")

        if headline_parts:
            insights.append("; ".join(headline_parts) + ": esto puede bajar tu presencia y eficiencia al entrenar.")
        else:
            insights.append("Tu alineación se ve bastante estable. Vamos a reforzarla con dos micro-correcciones para consolidar presencia y eficiencia.")

        insights.append("Proporción aquí es *eficiencia corporal* (proxy por ratios), no estética.")

        if back_symmetry_score is not None and back_symmetry_score < 70:
            insights.append("En la vista de espalda se sugiere asimetría leve. Unilateral + control suele corregirlo con el tiempo.")

    # Resultado
    payload_out: dict[str, Any] = {
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": {
            "score": round(float(confidence), 4),
            "uncertainty_score": round(float(uncertainty), 4),
            "n_views": n_views,
            "views": view_metrics,
        },
        "variables": {
            "posture_score": int(posture_score),
            "proportion_score": int(proportion_score),
            "alignment_silhouette_index": int(asi),
            "posture_components": {
                "shoulder_level": _nz(shoulder_level_score),
                "pelvis_level": _nz(pelvis_level_score),
                "base_axis": _nz(base_axis_score),
                "forward_head": _nz(forward_head_score),
                "rounded_shoulders": _nz(rounded_shoulders_score),
                "back_symmetry": _nz(back_symmetry_score),
            },
            "proportion_components": {
                "v_taper_proxy": _nz(v_taper_score),
                "torso_leg": _nz(torso_leg_score),
                "stance_grounding": _nz(stance_grounding_score),
            },
        },
        "patterns": pattern_keys,
        "immediate_corrections": actions_now,
        "weekly_adjustment": weekly_adjustment,
        "insights": insights[:3] if insights else ["Análisis listo. Si quieres más precisión, repite las fotos con la misma luz y encuadre."],
        "recommended_actions": ["Haz las 2 correcciones ahora y repite una foto rápida para ver el cambio."],
        "follow_up_questions": follow_up_questions,
        "meta": {"algorithm": "exp-013_posture_proportion_v0", "as_of": str(date.today())},
    }

    if baseline_delta is not None:
        payload_out["baseline_delta"] = {"alignment_silhouette_index": int(baseline_delta)}

    return PostureProportionResult(payload=payload_out)


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
        lines.append(f"posture_score: {int(vars_.get('posture_score') or 0)}/100")
        lines.append(f"proportion_score: {int(vars_.get('proportion_score') or 0)}/100")
        lines.append(f"alignment_silhouette_index: {int(vars_.get('alignment_silhouette_index') or 0)}/100")
    except Exception:
        pass

    delta = result.get("baseline_delta") if isinstance(result.get("baseline_delta"), dict) else {}
    if delta.get("alignment_silhouette_index") is not None:
        try:
            d = int(delta.get("alignment_silhouette_index") or 0)
            sign = "+" if d >= 0 else ""
            lines.append(f"cambio vs baseline (ASI): {sign}{d}")
        except Exception:
            pass

    insights = result.get("insights")
    if isinstance(insights, list) and insights:
        for x in insights[:2]:
            if str(x).strip():
                lines.append(f"insight: {str(x).strip()}")

    weekly = result.get("weekly_adjustment") if isinstance(result.get("weekly_adjustment"), dict) else {}
    if weekly.get("focus"):
        try:
            focus = weekly.get("focus")
            if isinstance(focus, list):
                focus_str = ", ".join([str(x) for x in focus if str(x).strip()])
                if focus_str:
                    lines.append(f"ajuste semanal: {focus_str}")
        except Exception:
            pass

    return "\n".join(lines).strip()
