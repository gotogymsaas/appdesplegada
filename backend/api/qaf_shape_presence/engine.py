from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ShapePresenceResult:
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


def evaluate_shape_presence(payload: dict[str, Any]) -> ShapePresenceResult:
    """Exp-012: Shape & Presence Intelligence™ (MVP).

    Objetivo: dar feedback de *presencia* y *proporción* usando keypoints 2D (cliente) sin prometer medidas reales.

    Entrada esperada:
      {"poses": {view_id: {keypoints:[...], image:{width,height}}}, "baseline"?: {...}}

    Views soportadas (opcionales):
      - front_relaxed (recomendado)
      - side_right_relaxed (opcional)

    Salida: payload QAF con variables 0..100 + acciones sugeridas.
    """

    poses = payload.get("poses") if isinstance(payload.get("poses"), dict) else {}
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else None

    allowed_views = ("front_relaxed", "side_right_relaxed")
    views_in = {k: v for k, v in poses.items() if isinstance(k, str) and k in allowed_views and isinstance(v, dict)}

    required_front = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
        "nose",
    ]
    required_side = ["right_ear", "right_shoulder", "right_hip"]

    view_metrics: dict[str, dict[str, Any]] = {}
    quality_scores: list[float] = []

    def get_kp(view_id: str) -> dict[str, dict[str, float]]:
        pose = views_in.get(view_id) if isinstance(views_in.get(view_id), dict) else {}
        return _kp_map(pose)

    for view_id, pose in views_in.items():
        kp = _kp_map(pose)
        req = required_side if view_id == "side_right_relaxed" else required_front
        q, missing = _pose_quality_score(kp, req)
        view_metrics[view_id] = {"pose_quality": round(float(q), 4), "missing": missing}
        quality_scores.append(float(q))

    n_views = len(views_in)
    ok_views = [k for k, m in view_metrics.items() if float(m.get("pose_quality") or 0.0) >= 0.55]

    decision = "accepted" if ok_views else "needs_confirmation"
    decision_reason = "ok" if ok_views else "missing_or_low_quality_pose"

    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted":
        follow_up_questions.append(
            {
                "type": "retake_photos",
                "prompt": "No pude detectar bien tu cuerpo. Intenta con luz uniforme y cuerpo completo (pies a cabeza).",
                "options": [],
            }
        )

    # Normalización interna por torso
    def _scale_from_front(kp: dict[str, dict[str, float]]) -> float | None:
        mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
        mid_hip, _ = _mid(kp, "left_hip", "right_hip")
        if not mid_sh or not mid_hip:
            return None
        s = _dist(mid_sh, mid_hip)
        return float(s) if s > 1e-6 else None

    alignment_symmetry = None
    silhouette_v_taper = None
    torso_leg_balance = None
    stance_grounding = None
    profile_stack = None

    insights: list[str] = []
    actions: list[str] = []

    # FRONT: proporciones + alineación + base
    if "front_relaxed" in views_in:
        kp = get_kp("front_relaxed")
        scale = _scale_from_front(kp) or 1.0

        ls = kp.get("left_shoulder")
        rs = kp.get("right_shoulder")
        lh = kp.get("left_hip")
        rh = kp.get("right_hip")
        nose = kp.get("nose")
        la = kp.get("left_ankle")
        ra = kp.get("right_ankle")

        if ls and rs and lh and rh:
            shoulder_w = abs(float(ls["x"]) - float(rs["x"]))
            hip_w = abs(float(lh["x"]) - float(rh["x"]))
            shoulder_level = abs(float(ls["y"]) - float(rs["y"])) / scale
            hip_level = abs(float(lh["y"]) - float(rh["y"])) / scale

            # Alineación/simetría (proxy)
            head_offset = 0.0
            if nose and shoulder_w > 1e-6:
                mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
                if mid_sh:
                    head_offset = abs(float(nose["x"]) - float(mid_sh[0])) / shoulder_w
            delta_align = float(0.5 * shoulder_level + 0.35 * hip_level + 0.15 * head_offset)
            alignment_symmetry = _score_from_delta(delta_align, good_at=0.03, bad_at=0.14)

            # Silueta (proxy V-taper): hombros vs cadera
            ratio = (shoulder_w / hip_w) if hip_w > 1e-6 else None
            if ratio is not None:
                # heurística: 1.00 bajo, 1.25 alto
                silhouette_v_taper = _score_from_ratio(float(ratio), lo=1.00, hi=1.25)

            # Balance torso/pierna (proxy)
            lk = kp.get("left_knee")
            rk = kp.get("right_knee")
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
                torso = float(scale)
                tl = torso / leg_len
                # ideal aproximado 0.50..0.75 (muy heurístico)
                torso_leg_balance = _score_from_ratio(float(tl), lo=0.45, hi=0.75)

            # Base/grounding: apertura de pies vs hombros
            if la and ra and shoulder_w > 1e-6:
                ankle_w = abs(float(la["x"]) - float(ra["x"]))
                stance = ankle_w / shoulder_w
                # ideal ~0.45..0.85 (demasiado cerrado o muy abierto resta)
                if stance <= 0.45:
                    stance_grounding = _score_from_ratio(float(stance), lo=0.15, hi=0.45)
                elif stance >= 0.85:
                    stance_grounding = _score_from_ratio(float(1.15 - stance), lo=0.30, hi=0.85)
                else:
                    stance_grounding = int(round(_clamp01((stance - 0.45) / (0.85 - 0.45)) * 100.0))

            # Insights neutrales
            if alignment_symmetry is not None and alignment_symmetry < 70:
                insights.append("Veo una ligera desalineación (hombros/pelvis/cabeza). Suele mejorar rápido con movilidad + control escapular y core.")
                actions.append("Durante 7 días: 2×10 chin-tucks + 2×12 retracciones escapulares (lento).")

            if silhouette_v_taper is not None and silhouette_v_taper < 55:
                insights.append("Si tu objetivo es una silueta más atlética (proxy V-taper), el camino suele ser espalda alta + deltoides lateral + control de cintura.")
                actions.append("Añade 6–10 series/semana de remos/pull + laterales, y 2 sesiones de core anti-rotación.")

            if stance_grounding is not None and stance_grounding < 55:
                insights.append("Tu base (apertura de pies) podría ser más estable. Una base estable suele aumentar la ‘presencia’ en postura y en levantamientos.")
                actions.append("Practica 3×30s de ‘stance’ estable (pies firmes, costillas abajo, respiración nasal) antes de entrenar.")

    # SIDE: stacking muy básico
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
                delta = float(0.6 * forward_head + 0.4 * rounded)
                profile_stack = _score_from_delta(delta, good_at=0.12, bad_at=0.35)

                if profile_stack < 70:
                    insights.append("En perfil se sugiere cabeza adelantada o hombros algo redondeados. Ajustes pequeños pueden cambiar mucho la presencia.")
                    actions.append("2 min/día: estiramiento pectoral + 2×10 chin-tucks.")

    # Confidence global
    conf = 1.0
    if decision != "accepted":
        conf *= 0.45
    conf *= _clamp01(sum(quality_scores) / max(1, len(quality_scores)) if quality_scores else 0.0)
    conf *= _clamp01(0.7 + 0.3 * (n_views / 2.0))
    confidence = _clamp01(conf)
    uncertainty = _clamp01(1.0 - confidence)

    def _nz(v: int | None) -> int:
        return int(v) if v is not None else 0

    # Overall: promedio de variables disponibles
    parts = [x for x in [alignment_symmetry, silhouette_v_taper, torso_leg_balance, stance_grounding, profile_stack] if x is not None]
    overall = int(round(sum(parts) / max(1, len(parts)))) if parts else 0

    baseline_delta = None
    try:
        if isinstance(baseline, dict):
            b_vars = baseline.get("variables") if isinstance(baseline.get("variables"), dict) else {}
            b_overall = int(b_vars.get("overall_presence") or 0)
            baseline_delta = int(overall) - int(b_overall)
    except Exception:
        baseline_delta = None

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
            "overall_presence": int(overall),
            "alignment_symmetry": _nz(alignment_symmetry),
            "silhouette_v_taper": _nz(silhouette_v_taper),
            "torso_leg_balance": _nz(torso_leg_balance),
            "stance_grounding": _nz(stance_grounding),
            "profile_stack": _nz(profile_stack),
        },
        "insights": (
            insights[:3]
            if insights
            else [
                "Análisis listo (proxy por keypoints). Si quieres más precisión, agrega una foto frontal con cuerpo completo y luz uniforme.",
            ]
        ),
        "recommended_actions": (
            actions[:3]
            if actions
            else [
                "Repite la medición en condiciones parecidas para comparar semana a semana.",
            ]
        ),
        "follow_up_questions": follow_up_questions,
        "meta": {"algorithm": "exp-012_shape_presence_v0", "as_of": str(date.today())},
    }

    if baseline_delta is not None:
        payload_out["baseline_delta"] = {"overall_presence": int(baseline_delta)}

    return ShapePresenceResult(payload=payload_out)


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
        lines.append(f"presencia (overall): {int(vars_.get('overall_presence') or 0)}/100")
        lines.append(f"alineación/simetría: {int(vars_.get('alignment_symmetry') or 0)}/100")
        lines.append(f"silueta V-taper (proxy): {int(vars_.get('silhouette_v_taper') or 0)}/100")
        lines.append(f"base/grounding: {int(vars_.get('stance_grounding') or 0)}/100")
    except Exception:
        pass

    delta = result.get("baseline_delta") if isinstance(result.get("baseline_delta"), dict) else {}
    if delta.get("overall_presence") is not None:
        try:
            d = int(delta.get("overall_presence") or 0)
            sign = "+" if d >= 0 else ""
            lines.append(f"cambio vs baseline (overall): {sign}{d}")
        except Exception:
            pass

    insights = result.get("insights")
    if isinstance(insights, list) and insights:
        for x in insights[:3]:
            if str(x).strip():
                lines.append(f"insight: {str(x).strip()}")

    return "\n".join(lines).strip()
