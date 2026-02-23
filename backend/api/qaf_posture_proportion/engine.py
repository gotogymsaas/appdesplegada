from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
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


def _tilt_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Inclinación (grados) vs horizontal, en rango ~[-90..90].

    Nota: en imagen, Y crece hacia abajo. Usamos |dx| para evitar saltos a ~180°.
    """
    dx = abs(float(b[0]) - float(a[0]))
    dy = float(b[1]) - float(a[1])
    return float(math.degrees(math.atan2(dy, max(1e-9, dx))))


def _pct01_to_100(x01: float) -> int:
    return int(round(_clamp01(float(x01)) * 100.0))


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

    # Calidad por vista
    q_front = float(view_metrics.get("front_relaxed", {}).get("pose_quality") or 0.0)
    q_side = float(view_metrics.get("side_right_relaxed", {}).get("pose_quality") or 0.0)

    # Aceptación: front + side con calidad aceptable.
    # Si llega solo 1 foto con calidad, calculamos parcial pero marcamos needs_confirmation.
    has_partial = (q_front >= 0.55) or (q_side >= 0.55)
    decision = "accepted" if (q_front >= 0.55 and q_side >= 0.55) else "needs_confirmation"
    decision_reason = "ok" if decision == "accepted" else ("partial_views" if has_partial else "missing_or_low_quality_pose")

    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted":
        follow_up_questions.append(
            {
                "type": "retake_photos",
                "prompt": "Puedo darte un estimado parcial con 1 foto, pero para mayor precisión necesito 2: frontal + perfil (cuerpo completo, buena luz, cámara a la altura del pecho, 2–3m).",
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

    # --- Proxies Exp-013+ (robustez) ---
    shoulder_tilt_deg = None
    pelvis_tilt_deg = None
    shoulder_tilt_note = None
    pelvis_tilt_note = None
    axis_asymmetry_pct = None
    load_distribution_pct = None
    hip_stability_score = None
    postural_efficiency_score = None
    metricfit_alignment_note = None

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

            # PoseLine Coach (proxy 2D): inclinación hombros (grados)
            try:
                raw = float(_tilt_deg((float(ls["x"]), float(ls["y"])), (float(rs["x"]), float(rs["y"]))))
                shoulder_tilt_deg = float(raw)
                # Nota humana: y+ es abajo → si rs.y > ls.y entonces hombro derecho está más bajo.
                if raw > 0.35:
                    shoulder_tilt_note = "derecho más bajo"
                elif raw < -0.35:
                    shoulder_tilt_note = "derecho más alto"
                else:
                    shoulder_tilt_note = "nivelado"
            except Exception:
                shoulder_tilt_deg = None
                shoulder_tilt_note = None

        if lh and rh:
            pelvis_level = abs(float(lh["y"]) - float(rh["y"])) / scale
            pelvis_level_score = _score_from_delta(float(pelvis_level), good_at=0.03, bad_at=0.12)

            # PoseLine Coach (proxy 2D): inclinación cadera (grados)
            try:
                raw = float(_tilt_deg((float(lh["x"]), float(lh["y"])), (float(rh["x"]), float(rh["y"]))))
                pelvis_tilt_deg = float(raw)
                if raw > 0.35:
                    pelvis_tilt_note = "derecha más baja"
                elif raw < -0.35:
                    pelvis_tilt_note = "derecha más alta"
                else:
                    pelvis_tilt_note = "nivelada"
            except Exception:
                pelvis_tilt_deg = None
                pelvis_tilt_note = None

        # Base axis (rodilla→tobillo) proxy: diferencias de x entre rodilla y tobillo por lado
        lk = kp.get("left_knee")
        rk = kp.get("right_knee")
        if lk and rk and la and ra:
            l = abs(float(lk["x"]) - float(la["x"]))
            r = abs(float(rk["x"]) - float(ra["x"]))
            base_axis = float(0.5 * (l + r))
            base_axis_score = _score_from_delta(base_axis, good_at=0.03, bad_at=0.12)

        # Symmetry Monitor (proxy 2D): asimetría de eje normalizada
        # - No es un % clínico; es un proxy (0..100) basado en deltas normalizadas por escala.
        try:
            parts = []
            if ls and rs:
                parts.append(abs(float(ls["y"]) - float(rs["y"])) / scale)
            if lh and rh:
                parts.append(abs(float(lh["y"]) - float(rh["y"])) / scale)
            if lk and rk and la and ra:
                parts.append(0.5 * (abs(float(lk["x"]) - float(la["x"])) + abs(float(rk["x"]) - float(ra["x"]))))
            # Mapear a 0..1 con rangos similares a nuestros umbrales.
            if parts:
                # 0.03 ~ limpio; 0.12 ~ fuerte
                mean_delta = float(sum(parts) / max(1, len(parts)))
                axis_asymmetry_pct = int(round(_clamp01((mean_delta - 0.01) / (0.14 - 0.01)) * 100.0))
        except Exception:
            axis_asymmetry_pct = None

        # Load Distribution Visual (proxy 2D): desplazamiento del centro de pelvis vs centro de tobillos
        try:
            mid_hip, _ = _mid(kp, "left_hip", "right_hip")
            mid_ank, _ = _mid(kp, "left_ankle", "right_ankle")
            if mid_hip and mid_ank:
                x_off = abs(float(mid_hip[0]) - float(mid_ank[0])) / max(1e-6, scale)
                # 0.02..0.12 aprox
                load_distribution_pct = int(round(_clamp01((x_off - 0.01) / (0.14 - 0.01)) * 100.0))
        except Exception:
            load_distribution_pct = None

        # Hip Stability Index (proxy 2D): mezcla de nivel de pelvis + centro de masa (muy simple)
        try:
            deltas = []
            if pelvis_level_score is not None:
                # Convertimos score a "inestabilidad" 0..1
                deltas.append((100.0 - float(pelvis_level_score)) / 100.0)
            if load_distribution_pct is not None:
                deltas.append(float(load_distribution_pct) / 100.0)
            if deltas:
                instab = float(sum(deltas) / max(1, len(deltas)))
                hip_stability_score = int(round(100.0 * (1.0 - _clamp01(instab))))
        except Exception:
            hip_stability_score = None

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

                # MetricFit Alignment Proxy (heurística): cuello adelantado / hombro adelantado suele “romper” cuello/solapa.
                try:
                    if forward_head_score is not None and forward_head_score < 75:
                        metricfit_alignment_note = "En camisas/blazers, el cuello puede verse ‘tirante’ por el eje (cabeza adelantada), no por talla."
                    elif rounded_shoulders_score is not None and rounded_shoulders_score < 75:
                        metricfit_alignment_note = "Si notas tirantez en pecho/solapa, primero limpia hombros (apertura) antes de culpar la talla."
                except Exception:
                    metricfit_alignment_note = None

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

    # Postural Efficiency Score (nuevo): premia postura + estabilidad (cadera) y penaliza asimetría
    try:
        hip_s = float(hip_stability_score if hip_stability_score is not None else posture_score)
        asym_pen = float(axis_asymmetry_pct if axis_asymmetry_pct is not None else 0.0)
        # 0..100
        postural_efficiency_score = int(round(_clamp01((0.55 * (posture_score / 100.0)) + (0.25 * (hip_s / 100.0)) + (0.20 * (1.0 - (asym_pen / 100.0)))) * 100.0))
    except Exception:
        postural_efficiency_score = None

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
    # Aviso de fiabilidad si falta una vista clave
    if decision != "accepted" and has_partial:
        insights.insert(0, "Resultado parcial: con 1 foto puedo estimar algunas señales, pero no es 100% fiable sin la segunda vista.")

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
            "postural_efficiency_score": int(postural_efficiency_score or 0),
            "symmetry_monitor": {
                "axis_asymmetry_pct": int(axis_asymmetry_pct) if axis_asymmetry_pct is not None else None,
                "load_distribution_pct": int(load_distribution_pct) if load_distribution_pct is not None else None,
                "hip_stability": int(hip_stability_score) if hip_stability_score is not None else None,
            },
            "pose_line": {
                "shoulder_tilt_deg": round(float(shoulder_tilt_deg), 2) if shoulder_tilt_deg is not None else None,
                "pelvis_tilt_deg": round(float(pelvis_tilt_deg), 2) if pelvis_tilt_deg is not None else None,
                "shoulder_note": shoulder_tilt_note,
                "pelvis_note": pelvis_tilt_note,
            },
            "metricfit_alignment_proxy": {
                "note": metricfit_alignment_note,
            },
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
        "meta": {"algorithm": "exp-013_posture_proportion_v1", "as_of": str(date.today())},
    }

    if baseline_delta is not None:
        payload_out["baseline_delta"] = {"alignment_silhouette_index": int(baseline_delta)}

    return PostureProportionResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    decision = str(result.get("decision") or "").strip().lower()
    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    vars_ = result.get("variables") if isinstance(result.get("variables"), dict) else {}
    patterns = result.get("patterns") if isinstance(result.get("patterns"), list) else []

    confidence_pct = None
    try:
        if conf.get("score") is not None:
            confidence_pct = int(round(float(conf.get("score")) * 100.0))
    except Exception:
        confidence_pct = None

    lines: list[str] = []
    lines.append("**Arquitectura Corporal (QAF)**")
    lines.append("(Lectura premium por foto: proxies por keypoints 2D; **no son medidas en cm** y no es diagnóstico.)")

    if decision != "accepted":
        lines.append("\n**⚠️ Necesito 2 fotos para ser preciso**")
        lines.append("- Frente relajado (cuerpo completo)")
        lines.append("- Perfil derecho (cuerpo completo)")
        lines.append("- Buena luz, sin contraluz; cámara a 2–3m y altura del pecho")
        return "\n".join(lines).strip()

    if confidence_pct is not None:
        lines.append(f"\n**✅ Listo** · Confianza de captura: {confidence_pct}%")
    else:
        lines.append("\n**✅ Listo**")

    # Resumen ejecutivo (lujo): 1 idea + 1 consecuencia + 1 objetivo
    insights = result.get("insights")
    headline = ""
    if isinstance(insights, list) and insights and str(insights[0]).strip():
        headline = str(insights[0]).strip()
    elif isinstance(patterns, list) and patterns:
        headline = "Detecté señales que bajan la limpieza del eje y la presencia en cámara."
    else:
        headline = "Tu base se ve estable. Vamos a consolidarla con micro‑ajustes finos."

    lines.append("\n**🧭 Lectura ejecutiva**")
    lines.append(f"- {headline}")
    lines.append("- Objetivo: una línea más limpia (eje) + caída más eficiente (sin tensión innecesaria).")

    # Índices (con contexto)
    try:
        lines.append("\n**📌 Índices (0–100)**")
        # Mostrar en formato porcentaje (score 0..100 => % de estabilidad/limpieza, no “% corporal”).
        eff = int(vars_.get('postural_efficiency_score') or 0)
        post = int(vars_.get('posture_score') or 0)
        prop = int(vars_.get('proportion_score') or 0)
        asi = int(vars_.get('alignment_silhouette_index') or 0)
        lines.append(f"- Eficiencia postural: {eff}%")
        lines.append(f"- Postura (alineación): {post}%")
        lines.append(f"- Proporción (proxy): {prop}%")
        lines.append(f"- Índice unificado (ASI): {asi}%")
    except Exception:
        pass

    lines.append("\n**🪡 Cómo leer esto (sin tecnicismos)**")
    lines.append("- *Eficiencia postural*: qué tan fácil se sostiene tu eje sin compensaciones.")
    lines.append("- *ASI*: mezcla postura+proxy para seguimiento semanal (tu ‘número único’).")

    # PoseLine + Simetría
    try:
        pose_line = vars_.get("pose_line") if isinstance(vars_.get("pose_line"), dict) else {}
        sym = vars_.get("symmetry_monitor") if isinstance(vars_.get("symmetry_monitor"), dict) else {}

        has_any = any(v is not None for v in [pose_line.get("shoulder_tilt_deg"), pose_line.get("pelvis_tilt_deg"), sym.get("axis_asymmetry_pct"), sym.get("load_distribution_pct")])
        if has_any:
            lines.append("\n**🧭 Señales (proxies por foto)**")
            if pose_line.get("shoulder_tilt_deg") is not None:
                note = str(pose_line.get('shoulder_note') or '').strip()
                tail = f" ({note})" if note else ""
                lines.append(f"- PoseLine (hombros): {pose_line.get('shoulder_tilt_deg')}°{tail}")
            if pose_line.get("pelvis_tilt_deg") is not None:
                note = str(pose_line.get('pelvis_note') or '').strip()
                tail = f" ({note})" if note else ""
                lines.append(f"- PoseLine (cadera): {pose_line.get('pelvis_tilt_deg')}°{tail}")
            if sym.get("axis_asymmetry_pct") is not None:
                ax = int(sym.get('axis_asymmetry_pct'))
                # Interpretación simple (marketing, no clínica)
                ax_tag = "baja" if ax <= 12 else ("media" if ax <= 25 else "alta")
                lines.append(f"- Asimetría de eje (proxy): {ax}% ({ax_tag})")
            if sym.get("load_distribution_pct") is not None:
                ld = int(sym.get('load_distribution_pct'))
                ld_tag = "estable" if ld <= 18 else ("mejorable" if ld <= 35 else "irregular")
                lines.append(f"- Distribución de carga (proxy): {ld}% ({ld_tag})")
            if sym.get("hip_stability") is not None:
                lines.append(f"- Estabilidad de cadera: {int(sym.get('hip_stability'))}/100")
    except Exception:
        pass

    # Prioridad 80/20
    lines.append("\n**🎯 Prioridad 80/20 (esta semana)**")
    if isinstance(patterns, list) and ('forward_head' in patterns or 'rounded_shoulders' in patterns):
        lines.append("- Eje cervical + apertura torácica: limpia presencia y mejora la caída de cualquier prenda.")
    elif isinstance(patterns, list) and ('pelvis_imbalance' in patterns or 'base_axis' in patterns):
        lines.append("- Base y pelvis: estabilidad primero, luego simetría (eso sube tu eficiencia global).")
    else:
        lines.append("- Consolidación: mantener eje limpio y estabilidad sin sobrecorregir.")

    # Micro-ajustes (Balance Correction Plan)
    lines.append("\n**🎯 Micro‑ajustes (hoy, 2 minutos)**")
    imm = result.get("immediate_corrections")
    if isinstance(imm, list) and imm:
        for ex in imm[:2]:
            if not isinstance(ex, dict):
                continue
            title = str(ex.get("title") or "").strip()
            cue = str(ex.get("cue") or "").strip()
            sec = int(ex.get("duration_sec") or 0)
            if title and cue and sec:
                lines.append(f"- {title}: {sec}s · {cue}")

    weekly = result.get("weekly_adjustment") if isinstance(result.get("weekly_adjustment"), dict) else {}
    if weekly:
        focus = weekly.get("focus") if isinstance(weekly.get("focus"), list) else []
        focus = [str(x).strip() for x in focus if str(x).strip()]
        if focus:
            lines.append("\n**📅 Ajuste semanal (3 sesiones)**")
            lines.append(f"- Enfoque: {', '.join(focus[:3])}")
            note = str(weekly.get('note') or '').strip()
            if note:
                lines.append(f"- Nota: {note}")

    # Fit proxy (MetricFit)
    try:
        mf = vars_.get("metricfit_alignment_proxy") if isinstance(vars_.get("metricfit_alignment_proxy"), dict) else {}
        note = str(mf.get("note") or "").strip()
        if note:
            lines.append("\n**🧵 Fit (proxy tipo atelier)**")
            lines.append(f"- {note}")
    except Exception:
        pass

    # Tendencia vs baseline
    delta = result.get("baseline_delta") if isinstance(result.get("baseline_delta"), dict) else {}
    if delta.get("alignment_silhouette_index") is not None:
        try:
            d = int(delta.get("alignment_silhouette_index") or 0)
            sign = "+" if d >= 0 else ""
            lines.append(f"\n**📈 Cambio vs última medición**: {sign}{d} puntos (ASI)")
        except Exception:
            pass

    # Sugerencias extra (más valor): deterministas por patrones/scores.
    lines.append("\n**✅ Sugerencias personalizadas (más impacto, sin complicarte)**")

    try:
        eff = int(vars_.get('postural_efficiency_score') or 0)
        post = int(vars_.get('posture_score') or 0)
        sym = vars_.get("symmetry_monitor") if isinstance(vars_.get("symmetry_monitor"), dict) else {}
        load_pct = sym.get('load_distribution_pct')
        axis_pct = sym.get('axis_asymmetry_pct')
    except Exception:
        eff = 0
        post = 0
        load_pct = None
        axis_pct = None

    suggestions: list[str] = []
    if isinstance(patterns, list) and 'forward_head' in patterns:
        suggestions.append("Hoy, prioriza 3–5 pausas de 20s: barbilla atrás + cuello largo (sin levantar mentón).")
        suggestions.append("En computadora: sube pantalla a la altura de ojos; el eje mejora más por entorno que por fuerza.")
    if isinstance(patterns, list) and 'rounded_shoulders' in patterns:
        suggestions.append("En calentamiento: 2 series de apertura torácica + retracción escapular suave (sin arquear lumbar).")
        suggestions.append("En press/pecho: baja 10–15% la carga 1 semana y gana control de escápula; sube eficiencia sin dolor.")
    if isinstance(patterns, list) and ('pelvis_imbalance' in patterns or 'base_axis' in patterns):
        suggestions.append("Antes de entrenar: 60s de respiración nasal + costillas abajo; eso estabiliza pelvis en segundos.")
        suggestions.append("En sentadillas: piensa ‘trípode del pie’ (dedo gordo + dedo pequeño + talón) para repartir carga.")
    if load_pct is not None:
        try:
            if int(load_pct) >= 35:
                suggestions.append("En estático (foto): reparte peso 50/50 y desbloquea rodillas; el eje se limpia de inmediato.")
        except Exception:
            pass
    if axis_pct is not None:
        try:
            if int(axis_pct) >= 25:
                suggestions.append("Esta semana, agrega 1 ejercicio unilateral (zancada o remo unilateral) con tempo lento: corrige asimetría con elegancia.")
        except Exception:
            pass

    if post and post < 70:
        suggestions.append("Para fotos comparables: mismo encuadre, misma distancia (2–3m), misma luz. Esa consistencia vale más que una ‘mejor pose’. ")
    if eff and eff >= 85:
        suggestions.append("Tu base está fuerte: el upgrade ahora es ‘pulido fino’ (menos corrección, más consistencia).")

    # Guardrail: siempre entregar suficientes bullets
    base = [
        "En caminata: imagina una cuerda que te ‘crece’ desde la coronilla; te da presencia sin rigidez.",
        "En el gym: termina cada sesión con 1 minuto de ‘eje limpio’ (de pie, respiración nasal, hombros bajos).",
        "Si una prenda ‘tira’, prueba primero 30s de eje (mentón atrás + hombros abajo) antes de ajustar talla.",
    ]
    for x in base:
        if x not in suggestions:
            suggestions.append(x)

    for s in suggestions[:9]:
        lines.append(f"- {s}")

    # Cierre (luxury): cómo medir + qué esperar
    lines.append("\n**📍 Seguimiento (lujo = consistencia)**")
    lines.append("- Repite las fotos con la misma luz/encuadre 1 vez por semana.")
    lines.append("- Señal de éxito: menos tensión en cuello/hombros y eje más ‘limpio’ en cámara.")

    return "\n".join(lines).strip()
