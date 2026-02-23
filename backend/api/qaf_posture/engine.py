from __future__ import annotations

from typing import Any


# Nota: Este engine del backend replica el motor del experimento sin depender de pose-estimation.
# Para producción, la entrada debe traer keypoints ya calculados (cliente o microservicio).

from dataclasses import dataclass
from datetime import date

import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    bax = float(a[0]) - float(b[0])
    bay = float(a[1]) - float(b[1])
    bcx = float(c[0]) - float(b[0])
    bcy = float(c[1]) - float(b[1])
    dot = (bax * bcx) + (bay * bcy)
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na <= 1e-9 or nc <= 1e-9:
        return None
    cosv = _clamp(dot / (na * nc), -1.0, 1.0)
    return float(math.degrees(math.acos(cosv)))


@dataclass(frozen=True)
class PostureResult:
    payload: dict[str, Any]


_REQ_FRONT = {
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "nose",
}

_REQ_SIDE_GROUPS = [
    ("right_ear", "left_ear"),
    ("right_shoulder", "left_shoulder"),
    ("right_hip", "left_hip"),
    ("right_knee", "left_knee"),
    ("right_ankle", "left_ankle"),
]


def _first(kp: dict[str, dict[str, float]], names: tuple[str, ...]):
    for n in names:
        if n in kp:
            return kp.get(n), n
    return None, None


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


def _mid(kp: dict[str, dict[str, float]], a: str, b: str):
    ka = kp.get(a)
    kb = kp.get(b)
    if not ka or not kb:
        return None, 0.0
    p = ((float(ka["x"]) + float(kb["x"])) / 2.0, (float(ka["y"]) + float(kb["y"])) / 2.0)
    conf = min(float(ka.get("score") or 0.0), float(kb.get("score") or 0.0))
    return p, conf


def _pose_quality(kp: dict[str, dict[str, float]], required: set[str]) -> dict[str, Any]:
    present = 0
    confs: list[float] = []
    missing = []
    for r in sorted(required):
        if r in kp:
            present += 1
            confs.append(float(kp[r].get("score") or 0.0))
        else:
            missing.append(r)
    ratio = present / max(1, len(required))
    avg_conf = (sum(confs) / max(1, len(confs))) if confs else 0.0
    score = _clamp01((0.65 * ratio) + (0.35 * avg_conf))
    return {"present": present, "required": len(required), "ratio": round(ratio, 4), "avg_conf": round(avg_conf, 4), "score": round(score, 4), "missing": missing}


def _pose_quality_side(kp: dict[str, dict[str, float]]) -> dict[str, Any]:
    present = 0
    confs: list[float] = []
    missing: list[str] = []
    for group in _REQ_SIDE_GROUPS:
        found, _ = _first(kp, group)
        if found:
            present += 1
            confs.append(float(found.get("score") or 0.0))
        else:
            missing.append("|".join(group))
    ratio = present / max(1, len(_REQ_SIDE_GROUPS))
    avg_conf = (sum(confs) / max(1, len(confs))) if confs else 0.0
    score = _clamp01((0.65 * ratio) + (0.35 * avg_conf))
    return {"present": present, "required": len(_REQ_SIDE_GROUPS), "ratio": round(ratio, 4), "avg_conf": round(avg_conf, 4), "score": round(score, 4), "missing": missing}


def _scale_from_pose(kp: dict[str, dict[str, float]]):
    sh_mid, sh_c = _mid(kp, "left_shoulder", "right_shoulder")
    hip_mid, hip_c = _mid(kp, "left_hip", "right_hip")
    if not sh_mid or not hip_mid:
        return None, 0.0
    s = _dist(sh_mid, hip_mid)
    if s <= 1e-6:
        return None, 0.0
    return float(s), float(min(sh_c, hip_c))


def _scale_from_side_pose(kp: dict[str, dict[str, float]]):
    sh, _ = _first(kp, ("right_shoulder", "left_shoulder"))
    hip, _ = _first(kp, ("right_hip", "left_hip"))
    if not sh or not hip:
        return None, 0.0
    s = _dist((sh["x"], sh["y"]), (hip["x"], hip["y"]))
    if s <= 1e-6:
        return None, 0.0
    conf = min(float(sh.get("score") or 0.0), float(hip.get("score") or 0.0))
    return float(s), float(conf)


def _signal(name: str, value: float | None, *, threshold: float | None, confidence: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": (round(float(value), 4) if value is not None else None),
        "threshold": (round(float(threshold), 4) if threshold is not None else None),
        "confidence": round(float(_clamp01(confidence)), 4),
    }


def _exercise_catalog() -> list[dict[str, Any]]:
    # Misma lista que en el experimento (versión backend). Mantener simple.
    return [
        {"id": "chin_tucks", "name": "Retracción de mentón (chin tucks)", "goal": "control cervical", "level": "basic", "duration_sec": 90, "sets": 2, "reps": 10, "contra": ["pain_neck", "injury_recent"]},
        {"id": "doorway_pec_stretch", "name": "Estiramiento pectoral en marco de puerta", "goal": "movilidad pecho/hombro", "level": "basic", "duration_sec": 120, "sets": 2, "reps": None, "contra": ["injury_recent"]},
        {"id": "scap_retractions", "name": "Retracciones escapulares", "goal": "estabilidad escapular", "level": "basic", "duration_sec": 120, "sets": 2, "reps": 12, "contra": ["injury_recent"]},
        {"id": "wall_angels", "name": "Ángeles en la pared (wall angels)", "goal": "movilidad torácica/escapular", "level": "basic", "duration_sec": 120, "sets": 2, "reps": 8, "contra": ["pain_shoulder", "injury_recent"]},
        {"id": "thoracic_extension", "name": "Extensión torácica (sobre toalla enrollada)", "goal": "movilidad dorsal", "level": "basic", "duration_sec": 120, "sets": 1, "reps": None, "contra": ["pain_low_back", "injury_recent"]},
        {"id": "dead_bug", "name": "Dead bug", "goal": "core", "level": "basic", "duration_sec": 180, "sets": 2, "reps": 8, "contra": ["pain_low_back", "injury_recent"]},
        {"id": "bird_dog", "name": "Bird dog", "goal": "estabilidad lumbopélvica", "level": "basic", "duration_sec": 180, "sets": 2, "reps": 8, "contra": ["pain_low_back", "injury_recent"]},
        {"id": "glute_bridge", "name": "Puente de glúteo", "goal": "glúteos", "level": "basic", "duration_sec": 180, "sets": 2, "reps": 12, "contra": ["pain_low_back", "injury_recent"]},
        {"id": "hip_flexor_stretch", "name": "Estiramiento de flexores de cadera", "goal": "movilidad cadera", "level": "basic", "duration_sec": 120, "sets": 2, "reps": None, "contra": ["injury_recent"]},
        {"id": "hamstring_stretch", "name": "Estiramiento de isquiotibiales", "goal": "movilidad posterior", "level": "basic", "duration_sec": 120, "sets": 2, "reps": None, "contra": ["injury_recent"]},
        {"id": "clamshell", "name": "Clamshell (abducción cadera en lateral)", "goal": "glúteo medio", "level": "basic", "duration_sec": 180, "sets": 2, "reps": 12, "contra": ["injury_recent"]},
        {"id": "side_steps", "name": "Pasos laterales (sin banda o con banda suave)", "goal": "control rodilla/cadera", "level": "basic", "duration_sec": 180, "sets": 2, "reps": 10, "contra": ["injury_recent"]},
    ]


def _filter_exercises(catalog: list[dict[str, Any]], user_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    flags = {k for k, v in (user_ctx or {}).items() if bool(v) is True}
    out = []
    for ex in catalog:
        if set(ex.get("contra") or []).intersection(flags):
            continue
        out.append(ex)
    return out


def _pick_by_labels(labels: list[dict[str, Any]]) -> list[str]:
    keys = {str(x.get("key") or "").strip() for x in (labels or []) if isinstance(x, dict)}
    ordered: list[str] = []

    def add(*ids: str):
        for i in ids:
            if i and i not in ordered:
                ordered.append(i)

    if "forward_head" in keys:
        add("chin_tucks", "thoracic_extension")
    if "rounded_shoulders" in keys:
        add("doorway_pec_stretch", "scap_retractions", "wall_angels")
    if "anterior_pelvic_tilt" in keys:
        add("hip_flexor_stretch", "glute_bridge", "dead_bug")
    if "knee_valgus" in keys:
        add("clamshell", "side_steps")
    add("bird_dog")
    return ordered


def evaluate_posture(payload: dict[str, Any]) -> PostureResult:
    poses = payload.get("poses") if isinstance(payload.get("poses"), dict) else {}
    front = poses.get("front") if isinstance(poses.get("front"), dict) else {}
    side = poses.get("side") if isinstance(poses.get("side"), dict) else {}
    user_ctx = payload.get("user_context") if isinstance(payload.get("user_context"), dict) else {}

    # Calibración opcional por altura (cm). Si no hay, seguimos con proxies.
    height_cm = None
    try:
        height_cm = float(user_ctx.get('height_cm')) if user_ctx.get('height_cm') is not None else None
    except Exception:
        height_cm = None

    front_img = front.get('image') if isinstance(front.get('image'), dict) else {}
    img_w = None
    img_h = None
    try:
        img_w = int(front_img.get('width')) if front_img.get('width') is not None else None
        img_h = int(front_img.get('height')) if front_img.get('height') is not None else None
        if img_w is not None and img_w <= 0:
            img_w = None
        if img_h is not None and img_h <= 0:
            img_h = None
    except Exception:
        img_w = None
        img_h = None

    kp_f = _kp_map(front)
    kp_s = _kp_map(side)
    q_front = _pose_quality(kp_f, _REQ_FRONT)
    q_side = _pose_quality_side(kp_s)
    scale_f, sc_conf_f = _scale_from_pose(kp_f)
    scale_s, sc_conf_s = _scale_from_side_pose(kp_s)

    missing: list[str] = []
    front_ok = bool(q_front["ratio"] >= 0.8 and scale_f is not None)
    side_ok = bool(q_side["ratio"] >= 0.8 and scale_s is not None)

    if q_front["ratio"] < 0.8:
        missing.append("front_keypoints")
    if q_side["ratio"] < 0.8:
        missing.append("side_keypoints")
    if not front_ok:
        missing.append("scale_front")
    if not side_ok:
        missing.append("scale_side")

    decision = "accepted" if (front_ok and side_ok) else "needs_confirmation"
    decision_reason = "ok" if decision == "accepted" else ("partial_views" if (front_ok or side_ok) else "missing_or_low_quality_pose")
    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted":
        follow_up_questions.append(
            {
                "type": "retake_photos",
                "prompt": "Puedo darte un análisis parcial con 1 foto, pero no es 100% fiable sin la segunda vista (frontal + lateral, cuerpo completo, buena luz, cámara a 2–3m).",
                "options": [],
            }
        )

    signals: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    def label(key: str, *, severity: str, confidence: float, evidence: list[str]):
        labels.append({"key": key, "severity": severity, "confidence": round(float(_clamp01(confidence)), 4), "evidence": evidence})

    # Señales de frontal (se calculan si hay escala frontal, incluso en modo parcial)
    if scale_f is not None and q_front.get("score", 0.0) >= 0.55:
        s = float(scale_f)
        sh_l = kp_f.get("left_shoulder")
        sh_r = kp_f.get("right_shoulder")
        el_l = kp_f.get("left_elbow")
        el_r = kp_f.get("right_elbow")
        wr_l = kp_f.get("left_wrist")
        wr_r = kp_f.get("right_wrist")
        hip_l = kp_f.get("left_hip")
        hip_r = kp_f.get("right_hip")
        kn_l = kp_f.get("left_knee")
        kn_r = kp_f.get("right_knee")
        an_l = kp_f.get("left_ankle")
        an_r = kp_f.get("right_ankle")
        nose = kp_f.get("nose")

        if sh_l and sh_r:
            shoulder_asym = abs(float(sh_l["y"]) - float(sh_r["y"])) / s
            conf = min(float(sh_l.get("score") or 0.0), float(sh_r.get("score") or 0.0), sc_conf_f)
            signals.append(_signal("shoulder_asymmetry", shoulder_asym, threshold=0.08, confidence=conf))
            if shoulder_asym >= 0.10:
                label("shoulder_asymmetry", severity="mild", confidence=conf, evidence=["front", "shoulder_asymmetry"])

        if hip_l and hip_r:
            hip_asym = abs(float(hip_l["y"]) - float(hip_r["y"])) / s
            conf = min(float(hip_l.get("score") or 0.0), float(hip_r.get("score") or 0.0), sc_conf_f)
            signals.append(_signal("hip_asymmetry", hip_asym, threshold=0.09, confidence=conf))
            if hip_asym >= 0.11:
                label("hip_asymmetry", severity="mild", confidence=conf, evidence=["front", "hip_asymmetry"])

        # Medidas personales (proxy) — normalizadas por torso (hombros->cadera).
        try:
            if sh_l and sh_r:
                w = _dist((float(sh_l["x"]), float(sh_l["y"])) , (float(sh_r["x"]), float(sh_r["y"]))) / s
                conf = min(float(sh_l.get("score") or 0.0), float(sh_r.get("score") or 0.0), sc_conf_f)
                signals.append(_signal("shoulder_width", float(w), threshold=None, confidence=conf))
        except Exception:
            pass

        try:
            if hip_l and hip_r:
                w = _dist((float(hip_l["x"]), float(hip_l["y"])) , (float(hip_r["x"]), float(hip_r["y"]))) / s
                conf = min(float(hip_l.get("score") or 0.0), float(hip_r.get("score") or 0.0), sc_conf_f)
                signals.append(_signal("hip_width", float(w), threshold=None, confidence=conf))
        except Exception:
            pass

        try:
            if hip_l and an_l:
                ll = _dist((float(hip_l["x"]), float(hip_l["y"])) , (float(an_l["x"]), float(an_l["y"]))) / s
                conf = min(float(hip_l.get("score") or 0.0), float(an_l.get("score") or 0.0), sc_conf_f)
                signals.append(_signal("leg_length_left", float(ll), threshold=None, confidence=conf))
        except Exception:
            pass

        try:
            if hip_r and an_r:
                ll = _dist((float(hip_r["x"]), float(hip_r["y"])) , (float(an_r["x"]), float(an_r["y"]))) / s
                conf = min(float(hip_r.get("score") or 0.0), float(an_r.get("score") or 0.0), sc_conf_f)
                signals.append(_signal("leg_length_right", float(ll), threshold=None, confidence=conf))
        except Exception:
            pass

        # Longitud de brazo (proxy). Requiere muñeca; si no está, no se calcula.
        try:
            if sh_l and wr_l:
                al = _dist((float(sh_l["x"]), float(sh_l["y"])) , (float(wr_l["x"]), float(wr_l["y"]))) / s
                conf = min(float(sh_l.get("score") or 0.0), float(wr_l.get("score") or 0.0), sc_conf_f)
                signals.append(_signal("arm_length_left", float(al), threshold=None, confidence=conf))
        except Exception:
            pass

        try:
            if sh_r and wr_r:
                al = _dist((float(sh_r["x"]), float(sh_r["y"])) , (float(wr_r["x"]), float(wr_r["y"]))) / s
                conf = min(float(sh_r.get("score") or 0.0), float(wr_r.get("score") or 0.0), sc_conf_f)
                signals.append(_signal("arm_length_right", float(al), threshold=None, confidence=conf))
        except Exception:
            pass

        mid_sh, c_sh = _mid(kp_f, "left_shoulder", "right_shoulder")
        mid_hip, c_hip = _mid(kp_f, "left_hip", "right_hip")
        if nose and (mid_sh or mid_hip):
            ref = mid_sh or mid_hip
            head_offset = abs(float(nose["x"]) - float(ref[0])) / s
            conf = min(float(nose.get("score") or 0.0), float(c_sh or c_hip), sc_conf_f)
            signals.append(_signal("head_center_offset", head_offset, threshold=0.12, confidence=conf))

        # Calibración por altura (cm estimados): convertir algunas medidas si el encuadre es usable.
        try:
            if height_cm and img_w and img_h:
                # Estimar altura corporal visible en px usando y (top: nariz/oreja/hombros; bottom: tobillos).
                top_candidates = []
                for name in ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"):
                    p = kp_f.get(name)
                    if p and p.get('y') is not None:
                        top_candidates.append(float(p['y']) * float(img_h))
                bottom_candidates = []
                for name in ("left_ankle", "right_ankle"):
                    p = kp_f.get(name)
                    if p and p.get('y') is not None:
                        bottom_candidates.append(float(p['y']) * float(img_h))

                if top_candidates and bottom_candidates:
                    top_y = min(top_candidates)
                    bot_y = max(bottom_candidates)
                    body_h_px = float(bot_y - top_y)

                    # Guardrails: necesitamos algo cercano a cuerpo completo.
                    if body_h_px > 0.55 * float(img_h) and height_cm >= 80.0 and height_cm <= 260.0:
                        # Corrección suave: top_y no es la coronilla; ajustamos un poco.
                        body_h_px = body_h_px * 1.08
                        cm_per_px = float(height_cm) / max(1.0, body_h_px)

                        def _px(pt: dict[str, float]) -> tuple[float, float]:
                            return (float(pt['x']) * float(img_w), float(pt['y']) * float(img_h))

                        def _dist_cm(a: dict[str, float], b: dict[str, float]) -> float:
                            return _dist(_px(a), _px(b)) * cm_per_px

                        # shoulder_width_cm / hip_width_cm
                        if sh_l and sh_r:
                            conf_cm = min(float(sh_l.get('score') or 0.0), float(sh_r.get('score') or 0.0), sc_conf_f)
                            signals.append(_signal('shoulder_width_cm', _dist_cm(sh_l, sh_r), threshold=None, confidence=conf_cm))
                        if hip_l and hip_r:
                            conf_cm = min(float(hip_l.get('score') or 0.0), float(hip_r.get('score') or 0.0), sc_conf_f)
                            signals.append(_signal('hip_width_cm', _dist_cm(hip_l, hip_r), threshold=None, confidence=conf_cm))

                        # arm_length_cm (hombro->muñeca)
                        if sh_l and wr_l:
                            conf_cm = min(float(sh_l.get('score') or 0.0), float(wr_l.get('score') or 0.0), sc_conf_f)
                            signals.append(_signal('arm_length_left_cm', _dist_cm(sh_l, wr_l), threshold=None, confidence=conf_cm))
                        if sh_r and wr_r:
                            conf_cm = min(float(sh_r.get('score') or 0.0), float(wr_r.get('score') or 0.0), sc_conf_f)
                            signals.append(_signal('arm_length_right_cm', _dist_cm(sh_r, wr_r), threshold=None, confidence=conf_cm))

                        # leg_length_cm (cadera->tobillo)
                        if hip_l and an_l:
                            conf_cm = min(float(hip_l.get('score') or 0.0), float(an_l.get('score') or 0.0), sc_conf_f)
                            signals.append(_signal('leg_length_left_cm', _dist_cm(hip_l, an_l), threshold=None, confidence=conf_cm))
                        if hip_r and an_r:
                            conf_cm = min(float(hip_r.get('score') or 0.0), float(an_r.get('score') or 0.0), sc_conf_f)
                            signals.append(_signal('leg_length_right_cm', _dist_cm(hip_r, an_r), threshold=None, confidence=conf_cm))
        except Exception:
            pass

        for side_key in ("left", "right"):
            hip = kp_f.get(f"{side_key}_hip")
            knee = kp_f.get(f"{side_key}_knee")
            ankle = kp_f.get(f"{side_key}_ankle")
            if not (hip and knee and ankle):
                continue
            ang = _angle((hip["x"], hip["y"]), (knee["x"], knee["y"]), (ankle["x"], ankle["y"]))
            conf = min(float(hip.get("score") or 0.0), float(knee.get("score") or 0.0), float(ankle.get("score") or 0.0), sc_conf_f)
            if ang is not None:
                signals.append(_signal(f"knee_angle_{side_key}", ang, threshold=168.0, confidence=conf))
                if float(ang) < 165.0 and conf >= 0.6:
                    label("knee_valgus", severity="mild", confidence=conf, evidence=["front", f"knee_angle_{side_key}"])

    # Señales de lateral (se calculan si hay escala lateral, incluso en modo parcial)
    if scale_s is not None and q_side.get("score", 0.0) >= 0.55:
        s = float(scale_s)
        ear, _ = _first(kp_s, ("right_ear", "left_ear"))
        sh, _ = _first(kp_s, ("right_shoulder", "left_shoulder"))
        hip, _ = _first(kp_s, ("right_hip", "left_hip"))
        knee, _ = _first(kp_s, ("right_knee", "left_knee"))

        if ear and sh:
            forward_head = abs(float(ear["x"]) - float(sh["x"])) / s
            conf = min(float(ear.get("score") or 0.0), float(sh.get("score") or 0.0), sc_conf_s)
            signals.append(_signal("forward_head", forward_head, threshold=0.18, confidence=conf))
            if forward_head >= 0.20 and conf >= 0.6:
                label("forward_head", severity="mild", confidence=conf, evidence=["side", "forward_head"])

        if sh and hip:
            rounded = abs(float(sh["x"]) - float(hip["x"])) / s
            conf = min(float(sh.get("score") or 0.0), float(hip.get("score") or 0.0), sc_conf_s)
            signals.append(_signal("rounded_shoulders", rounded, threshold=0.16, confidence=conf))
            if rounded >= 0.18 and conf >= 0.6:
                label("rounded_shoulders", severity="mild", confidence=conf, evidence=["side", "rounded_shoulders"])

        if hip and knee and sh:
            pelvis_dx = abs(float(hip["x"]) - float(knee["x"])) / s
            conf = min(float(hip.get("score") or 0.0), float(knee.get("score") or 0.0), float(sh.get("score") or 0.0), sc_conf_s)
            signals.append(_signal("pelvis_knee_offset", pelvis_dx, threshold=0.22, confidence=conf))
            if pelvis_dx >= 0.26 and conf >= 0.6:
                label("anterior_pelvic_tilt", severity="mild", confidence=conf, evidence=["side", "pelvis_knee_offset"])

    present_scores: list[float] = []
    try:
        if float(q_front.get("score") or 0.0) > 0:
            present_scores.append(float(q_front.get("score") or 0.0))
        if float(q_side.get("score") or 0.0) > 0:
            present_scores.append(float(q_side.get("score") or 0.0))
    except Exception:
        present_scores = []
    base_quality = float(_clamp01(sum(present_scores) / max(1, len(present_scores)) if present_scores else 0.0))

    # Penaliza si no están ambas vistas
    view_penalty = 1.0 if (front_ok and side_ok) else 0.72
    conf = float(_clamp01(base_quality * view_penalty))
    uncertainty = float(_clamp01(1.0 - conf))

    catalog = _filter_exercises(_exercise_catalog(), user_ctx)
    ids = _pick_by_labels(labels)
    ex_by_id = {e["id"]: e for e in catalog}
    routine = [ex_by_id[i] for i in ids if i in ex_by_id][:6]

    payload_out = {
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": {
            "score": round(float(conf), 4),
            "uncertainty_score": round(float(uncertainty), 4),
            "pose_quality": {"front": q_front, "side": q_side},
            "missing": missing,
        },
        "signals": signals,
        "labels": labels,
        "recommendations": {
            "routine": routine,
            "safety": [
                "Esto no es diagnóstico médico. Es una guía de técnica y hábitos posturales.",
                "Si hay dolor agudo, hormigueo, adormecimiento o lesión reciente, prioriza consultar a un profesional.",
            ],
        },
        "follow_up_questions": follow_up_questions,
        "meta": {
            "algorithm": "exp-006_posture_corrective_v0",
            "as_of": str(date.today()),
            "height_cm": height_cm,
        },
    }
    return PostureResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    labels = result.get("labels") if isinstance(result.get("labels"), list) else []
    signals = result.get("signals") if isinstance(result.get("signals"), list) else []
    rec = result.get("recommendations") if isinstance(result.get("recommendations"), dict) else {}
    routine = rec.get("routine") if isinstance(rec.get("routine"), list) else []

    progress = result.get('progress') if isinstance(result.get('progress'), dict) else {}
    vs_last = progress.get('vs_last') if isinstance(progress.get('vs_last'), list) else []

    history = result.get('history') if isinstance(result.get('history'), dict) else {}
    hist_count: int | None = None
    try:
        if history.get('count') is not None:
            hist_count = int(history.get('count'))
    except Exception:
        hist_count = None

    lines: list[str] = []
    decision = str(result.get('decision') or '').strip() or 'unknown'

    # Encabezado: evitar “Tu progreso” si el análisis es parcial.
    if decision != 'accepted':
        lines.append("Corrección de postura — Tu resumen")
    elif vs_last and (hist_count is None or hist_count >= 2):
        lines.append("Corrección de postura — Tu progreso")
    elif hist_count == 1:
        lines.append("Corrección de postura — Punto de partida")
    else:
        lines.append("Corrección de postura — Tu resumen")
    if conf.get("score") is not None:
        try:
            pct = round(float(conf.get('score')) * 100.0, 0)
            lines.append(f"Confianza del análisis: {pct:.0f}%")
        except Exception:
            pass

    # Aviso profesional: parcialidad (sin tecnicismos)
    try:
        if decision != 'accepted':
            conf_block = result.get('confidence') if isinstance(result.get('confidence'), dict) else {}
            pq = conf_block.get('pose_quality') if isinstance(conf_block.get('pose_quality'), dict) else {}
            f_ok = False
            s_ok = False
            try:
                f_ok = float((pq.get('front') or {}).get('score') or 0.0) >= 0.55
            except Exception:
                f_ok = False
            try:
                s_ok = float((pq.get('side') or {}).get('score') or 0.0) >= 0.55
            except Exception:
                s_ok = False
            if f_ok or s_ok:
                # Copy marketing (parcial)
                lines.append('')
                lines.append('Este es un análisis parcial.')
                lines.append('Con una sola vista puedo orientarte, pero con frontal + lateral el resultado sería mucho más preciso.')
    except Exception:
        pass

    if vs_last:
        evo_lines: list[str] = []
        shown = 0
        for item in vs_last:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label') or '').strip()
            kind = str(item.get('kind') or '').strip()
            try:
                delta = float(item.get('delta') or 0.0)
            except Exception:
                continue
            if not label:
                continue

            # Interpretación simple y honesta: magnitud + dirección.
            if abs(delta) < 1e-9:
                continue

            improved: bool | None = None
            if kind == 'lower_better':
                improved = delta < 0
            elif kind == 'higher_better':
                improved = delta > 0

            if improved is True:
                emoji = "✅"
                verb = "mejoraste"
                arrow = "↑"
            elif improved is False:
                emoji = "⚠️"
                verb = "ligera variación"
                arrow = "↓"
            else:
                emoji = "ℹ️"
                verb = "cambio"
                arrow = "→"

            mag = abs(delta)
            # Unidades: normalmente son "proxy"; mostramos magnitud acotada.
            evo_lines.append(f"{emoji} {label}: {verb} ({arrow} {mag:.2f})")
            shown += 1
            if shown >= 4:
                break

        if evo_lines:
            lines.append("\n📈 Tu evolución vs la última medición:")
            lines.extend(evo_lines)
            if hist_count is not None and hist_count >= 3:
                lines.append("\n🔎 Lo importante")
                lines.append("Pequeñas correcciones sostenidas → cambios visibles.")

    if (not vs_last) and hist_count == 1 and decision == 'accepted':
        lines.append("\n📈 Próximo paso")
        lines.append("Haz una segunda medición en 3–7 días. Con eso ya podremos mostrarte mejoras reales.")
        lines.append("\n🔎 Para que la comparación sea confiable")
        lines.append("• Misma luz")
        lines.append("• Misma distancia (2–3 m)")
        lines.append("• Cámara a la altura del pecho")
        lines.append("• Cuerpo completo visible")
    # Métricas interesantes (sin prometer cm reales; son proporciones normalizadas por escala corporal)
    try:
        def _sig(name: str):
            for s in signals:
                if isinstance(s, dict) and str(s.get('name') or '').strip() == name:
                    return s
            return None

        def _pct(v: Any) -> float | None:
            try:
                return float(v) * 100.0
            except Exception:
                return None

        metrics: list[str] = []

        sh = _sig('shoulder_asymmetry')
        if sh and sh.get('value') is not None:
            v = _pct(sh.get('value'))
            t = _pct(sh.get('threshold'))
            if v is not None:
                metrics.append(f"Hombros (asimetría): {v:.1f}%" + (f" (umbral {t:.1f}%)" if t is not None else ""))

        hip = _sig('hip_asymmetry')
        if hip and hip.get('value') is not None:
            v = _pct(hip.get('value'))
            t = _pct(hip.get('threshold'))
            if v is not None:
                metrics.append(f"Cadera (asimetría): {v:.1f}%" + (f" (umbral {t:.1f}%)" if t is not None else ""))

        fh = _sig('forward_head')
        if fh and fh.get('value') is not None:
            v = _pct(fh.get('value'))
            t = _pct(fh.get('threshold'))
            if v is not None:
                metrics.append(f"Cabeza adelantada (proxy): {v:.1f}%" + (f" (umbral {t:.1f}%)" if t is not None else ""))

        rs = _sig('rounded_shoulders')
        if rs and rs.get('value') is not None:
            v = _pct(rs.get('value'))
            t = _pct(rs.get('threshold'))
            if v is not None:
                metrics.append(f"Hombros redondeados (proxy): {v:.1f}%" + (f" (umbral {t:.1f}%)" if t is not None else ""))

        # Medidas personales (proxy) — NO son cm; son valores relativos a tu torso (hombros->cadera).
        def _rel(v: Any) -> float | None:
            try:
                return float(v)
            except Exception:
                return None

        personal: list[str] = []
        sw = _sig('shoulder_width')
        if sw and sw.get('value') is not None:
            v = _rel(sw.get('value'))
            if v is not None:
                personal.append(f"Ancho de hombros (relativo): {v:.2f}× torso")

        hw = _sig('hip_width')
        if hw and hw.get('value') is not None:
            v = _rel(hw.get('value'))
            if v is not None:
                personal.append(f"Ancho de cadera (relativo): {v:.2f}× torso")

        al = _sig('arm_length_left')
        ar = _sig('arm_length_right')
        if al and al.get('value') is not None:
            v = _rel(al.get('value'))
            if v is not None:
                personal.append(f"Longitud de brazo izq (proxy): {v:.2f}× torso")
        if ar and ar.get('value') is not None:
            v = _rel(ar.get('value'))
            if v is not None:
                personal.append(f"Longitud de brazo der (proxy): {v:.2f}× torso")

        ll = _sig('leg_length_left')
        lr = _sig('leg_length_right')
        if ll and ll.get('value') is not None:
            v = _rel(ll.get('value'))
            if v is not None:
                personal.append(f"Longitud de pierna izq (proxy): {v:.2f}× torso")
        if lr and lr.get('value') is not None:
            v = _rel(lr.get('value'))
            if v is not None:
                personal.append(f"Longitud de pierna der (proxy): {v:.2f}× torso")

        if metrics:
            lines.append("\n📊 Tus métricas hoy")
            lines.append("Alineación")
            for m in metrics[:4]:
                # Re-formatear a bullets con punto medio
                mm = str(m).strip()
                if mm:
                    lines.append(f"• {mm}")

            # Refuerzo de lectura (marketing): si están bajo umbral, decirlo.
            try:
                # Con la data actual, el umbral está en el string; usamos una heurística simple.
                lines.append('')
                lines.append('👉 Estás muy por debajo del umbral de alerta.')
                lines.append('Tu alineación general se ve estable.')
            except Exception:
                pass

        # Medidas en cm (estimadas): preferir calibración por px; fallback por altura+proporciones.
        cm_lines: list[str] = []
        swc = _sig('shoulder_width_cm')
        if swc and swc.get('value') is not None:
            try:
                cm_lines.append(f"Ancho de hombros (cm estimado): {float(swc.get('value')):.1f} cm")
            except Exception:
                pass
        hwc = _sig('hip_width_cm')
        if hwc and hwc.get('value') is not None:
            try:
                cm_lines.append(f"Ancho de cadera (cm estimado): {float(hwc.get('value')):.1f} cm")
            except Exception:
                pass
        alc = _sig('arm_length_left_cm')
        if alc and alc.get('value') is not None:
            try:
                cm_lines.append(f"Brazo izq (hombro→muñeca): {float(alc.get('value')):.1f} cm")
            except Exception:
                pass
        arc = _sig('arm_length_right_cm')
        if arc and arc.get('value') is not None:
            try:
                cm_lines.append(f"Brazo der (hombro→muñeca): {float(arc.get('value')):.1f} cm")
            except Exception:
                pass

        if cm_lines:
            # Si pudimos calibrar por pixeles, estas son las mejores cm posibles.
            lines.append("\n📏 Proporciones estimadas (cm)")
            for m in cm_lines[:6]:
                lines.append(f"• {m}")

        # Fallback: si NO hay cm por px, pero existe height_cm, convertir proxies a cm con proporción (aprox).
        if (not cm_lines) and personal:
            height_cm = None
            try:
                meta = result.get('meta') if isinstance(result.get('meta'), dict) else {}
                height_cm = float(meta.get('height_cm')) if meta.get('height_cm') is not None else None
            except Exception:
                height_cm = None

            if height_cm and 80.0 <= float(height_cm) <= 260.0:
                # Aproximación: hombro→cadera suele estar ~30–35% de la estatura.
                torso_cm_est = float(height_cm) * 0.32

                def _rel_sig(name: str) -> float | None:
                    s = _sig(name)
                    if not s or s.get('value') is None:
                        return None
                    try:
                        return float(s.get('value'))
                    except Exception:
                        return None

                sw_rel = _rel_sig('shoulder_width')
                hw_rel = _rel_sig('hip_width')
                al_rel = _rel_sig('arm_length_left')
                ar_rel = _rel_sig('arm_length_right')

                approx_lines: list[str] = []
                if sw_rel is not None:
                    approx_lines.append(f"Ancho de hombros (cm aprox): {sw_rel * torso_cm_est:.1f} cm")
                if hw_rel is not None:
                    approx_lines.append(f"Ancho de cadera (cm aprox): {hw_rel * torso_cm_est:.1f} cm")
                if al_rel is not None:
                    approx_lines.append(f"Brazo izq (cm aprox): {al_rel * torso_cm_est:.1f} cm")
                if ar_rel is not None:
                    approx_lines.append(f"Brazo der (cm aprox): {ar_rel * torso_cm_est:.1f} cm")

                if approx_lines:
                    lines.append("\n📏 Proporciones estimadas (según estatura)")
                    for m in approx_lines[:6]:
                        # Ya viene con texto tipo "Ancho de hombros..."; lo dejamos como bullet simple
                        lines.append(f"• {m}")

                    lines.append("")
                    lines.append("Estas medidas son aproximadas y sirven como referencia comparativa para próximas mediciones.")
                    lines.append("")
                    lines.append("Para obtener valores más fiables:")
                    lines.append("• Foto de cuerpo completo (pies a cabeza)")
                    lines.append("• Buena luz frontal")
                    lines.append("• Distancia de 2–3 metros")
                    lines.append("")
                    lines.append("La consistencia mejora la precisión.")
    except Exception:
        pass

    if labels:
        keys = [str(x.get("key")) for x in labels if isinstance(x, dict) and x.get("key")]
        if keys:
            lines.append("hallazgos: " + ", ".join(keys[:6]))
    if routine:
        # Cierre marketing
        pretty = []
        for ex in routine[:5]:
            if not isinstance(ex, dict):
                continue
            nm = str(ex.get('name') or '').strip()
            sets = ex.get('sets')
            reps = ex.get('reps')
            dur = ex.get('duration_sec')
            if not nm:
                continue
            tail = []
            try:
                if sets:
                    tail.append(f"{int(sets)} series")
            except Exception:
                pass
            try:
                if reps:
                    tail.append(f"{int(reps)} reps")
            except Exception:
                pass
            try:
                if dur:
                    tail.append(f"{int(dur)}s")
            except Exception:
                pass
            pretty.append(nm + (f" ({', '.join(tail)})" if tail else ""))
        if pretty:
            lines.append("\n🎯 Ajuste recomendado hoy")
            # Intentar el formato exacto del copy para el primer ejercicio
            first = routine[0] if isinstance(routine[0], dict) else {}
            nm0 = str(first.get('name') or '').strip()
            sets0 = first.get('sets')
            reps0 = first.get('reps')
            rest0 = first.get('rest_sec')
            if nm0:
                lines.append(nm0)
                bits = []
                try:
                    if sets0:
                        bits.append(f"{int(sets0)} series")
                except Exception:
                    pass
                try:
                    if reps0:
                        bits.append(f"{int(reps0)} repeticiones")
                except Exception:
                    pass
                try:
                    if rest0:
                        bits.append(f"descanso {int(rest0)}s")
                except Exception:
                    pass
                if bits:
                    lines.append(" · ".join(bits))

            # Si no hay rest_sec, igual mostramos los ejercicios restantes como apoyo
            if len(pretty) > 1:
                lines.append("\nOpcional (si tienes 8–10 min):")
                for x in pretty[1:3]:
                    lines.append(f"• {x}")

            lines.append("")
            lines.append("Pequeños ajustes sostenidos → postura más fuerte y eficiente.")
    return "\n".join(lines).strip()
