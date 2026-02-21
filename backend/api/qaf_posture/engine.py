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

    kp_f = _kp_map(front)
    kp_s = _kp_map(side)
    q_front = _pose_quality(kp_f, _REQ_FRONT)
    q_side = _pose_quality_side(kp_s)
    scale_f, sc_conf_f = _scale_from_pose(kp_f)
    scale_s, sc_conf_s = _scale_from_side_pose(kp_s)

    missing: list[str] = []
    if q_front["ratio"] < 0.8:
        missing.append("front_keypoints")
    if q_side["ratio"] < 0.8:
        missing.append("side_keypoints")
    if scale_f is None or scale_s is None:
        missing.append("scale")

    decision = "accepted"
    decision_reason = "ok"
    follow_up_questions: list[dict[str, Any]] = []
    if missing:
        decision = "needs_confirmation"
        decision_reason = "missing_or_low_quality_pose"
        follow_up_questions.append({"type": "retake_photos", "prompt": "Necesito frontal + lateral con cuerpo completo y buena luz.", "options": []})

    signals: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    def label(key: str, *, severity: str, confidence: float, evidence: list[str]):
        labels.append({"key": key, "severity": severity, "confidence": round(float(_clamp01(confidence)), 4), "evidence": evidence})

    if decision == "accepted" and scale_f is not None:
        s = float(scale_f)
        sh_l = kp_f.get("left_shoulder")
        sh_r = kp_f.get("right_shoulder")
        hip_l = kp_f.get("left_hip")
        hip_r = kp_f.get("right_hip")
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

        mid_sh, c_sh = _mid(kp_f, "left_shoulder", "right_shoulder")
        mid_hip, c_hip = _mid(kp_f, "left_hip", "right_hip")
        if nose and (mid_sh or mid_hip):
            ref = mid_sh or mid_hip
            head_offset = abs(float(nose["x"]) - float(ref[0])) / s
            conf = min(float(nose.get("score") or 0.0), float(c_sh or c_hip), sc_conf_f)
            signals.append(_signal("head_center_offset", head_offset, threshold=0.12, confidence=conf))

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

    if decision == "accepted" and scale_s is not None:
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

    base_quality = float(_clamp01((float(q_front["score"]) + float(q_side["score"])) / 2.0))
    conf = base_quality if decision == "accepted" else float(_clamp01(base_quality * 0.65))
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
        "meta": {"algorithm": "exp-006_posture_corrective_v0", "as_of": str(date.today())},
    }
    return PostureResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    conf = result.get("confidence") if isinstance(result.get("confidence"), dict) else {}
    labels = result.get("labels") if isinstance(result.get("labels"), list) else []
    rec = result.get("recommendations") if isinstance(result.get("recommendations"), dict) else {}
    routine = rec.get("routine") if isinstance(rec.get("routine"), list) else []

    lines: list[str] = []
    lines.append(f"decision: {result.get('decision')}")
    if conf.get("score") is not None:
        try:
            lines.append(f"confidence: {round(float(conf.get('score')), 3)}")
        except Exception:
            pass
    if labels:
        keys = [str(x.get("key")) for x in labels if isinstance(x, dict) and x.get("key")]
        if keys:
            lines.append("señales: " + ", ".join(keys[:6]))
    if routine:
        names = [str(x.get("name")) for x in routine if isinstance(x, dict) and x.get("name")]
        if names:
            lines.append("rutina: " + "; ".join(names[:5]))
    return "\n".join(lines).strip()
