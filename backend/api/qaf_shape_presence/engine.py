from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import math


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
    """Exp-012: Alta Costura Inteligente™ (MVP).

    Antes: "Shape & Presence".

    Objetivo: asesor de estilo/alta costura basado en *proporciones ópticas* (ratios proxy) y *presencia*.
    Importante:
    - No entrega medidas reales en cm.
    - No es diagnóstico médico.
    - Si la calidad del pose es baja: `needs_confirmation`.

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

    proxies: dict[str, Any] = {
        "shoulder_hip_ratio": None,
        "torso_leg_ratio": None,
        "stance_ratio": None,
        "alignment_delta": None,
        "profile_delta": None,
    }

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

            try:
                if hip_w > 1e-6:
                    proxies["shoulder_hip_ratio"] = float(shoulder_w / hip_w)
            except Exception:
                pass

            # Alineación/simetría (proxy)
            head_offset = 0.0
            if nose and shoulder_w > 1e-6:
                mid_sh, _ = _mid(kp, "left_shoulder", "right_shoulder")
                if mid_sh:
                    head_offset = abs(float(nose["x"]) - float(mid_sh[0])) / shoulder_w
            delta_align = float(0.5 * shoulder_level + 0.35 * hip_level + 0.15 * head_offset)
            alignment_symmetry = _score_from_delta(delta_align, good_at=0.03, bad_at=0.14)
            try:
                proxies["alignment_delta"] = float(delta_align)
            except Exception:
                pass

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
                try:
                    proxies["torso_leg_ratio"] = float(tl)
                except Exception:
                    pass
                # ideal aproximado 0.50..0.75 (muy heurístico)
                torso_leg_balance = _score_from_ratio(float(tl), lo=0.45, hi=0.75)

            # Base/grounding: apertura de pies vs hombros
            if la and ra and shoulder_w > 1e-6:
                ankle_w = abs(float(la["x"]) - float(ra["x"]))
                stance = ankle_w / shoulder_w
                try:
                    proxies["stance_ratio"] = float(stance)
                except Exception:
                    pass
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
                try:
                    proxies["profile_delta"] = float(delta)
                except Exception:
                    pass

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

    # --- Alta Costura: motor de sugerencias (no médico; no cm reales) ---
    def _fmt_pct(v: int | None) -> str:
        try:
            if v is None:
                return "—"
            return f"{int(v)}%"
        except Exception:
            return "—"

    def _silhouette_signature(r: float | None) -> str:
        if r is None:
            return "Arquitectura visual: no disponible"
        if r >= 1.18:
            return "Arquitectura visual: hombro dominante (línea superior fuerte)"
        if r <= 0.92:
            return "Arquitectura visual: cadera dominante (base fuerte)"
        return "Arquitectura visual: balance hombro–cadera"

    def _vertical_signature(tl: float | None) -> str:
        if tl is None:
            return "Proporción vertical: no disponible"
        # tl = torso / pierna (proxy)
        if tl >= 0.66:
            return "Proporción vertical: torso visual largo"
        if tl <= 0.52:
            return "Proporción vertical: pierna visual larga"
        return "Proporción vertical: balanceada"

    def _pick_priorities(vars_: dict[str, int]) -> list[str]:
        mapping = {
            "alignment_symmetry": "alineación (línea limpia)",
            "silhouette_v_taper": "arquitectura de silueta (hombro–cintura)",
            "torso_leg_balance": "proporción vertical (tiro/largos)",
            "stance_grounding": "base y caída (presencia)",
            "profile_stack": "perfil (cuello/escote/solapa)",
        }
        items = []
        for k, label in mapping.items():
            v = vars_.get(k)
            if v is None:
                continue
            try:
                items.append((label, int(v)))
            except Exception:
                continue
        items.sort(key=lambda t: t[1])
        out = [name for name, _v in items[:3]]
        return out or ["alineación (línea limpia)", "proporción vertical (tiro/largos)", "arquitectura de silueta (hombro–cintura)"]

    def _couture_actions(*, r: float | None, tl: float | None, v_taper_score: int | None, align_score: int | None, profile_score: int | None) -> list[str]:
        actions_out: list[str] = []

        # 1) Verticalidad (tiro/largos)
        if tl is not None:
            if tl >= 0.66:
                actions_out.append("Pantalón: prioriza tiro medio‑alto y caída recta para alargar pierna visual.")
                actions_out.append("Chaqueta: largo corto a medio (a la altura de cadera alta) para subir cintura visual.")
            elif tl <= 0.52:
                actions_out.append("Pantalón: tiro medio (evita ultra‑alto) y pierna limpia para no acortar torso visual.")
                actions_out.append("Chaqueta: largo medio (un poco más abajo de cintura) para equilibrar verticalidad.")
            else:
                actions_out.append("Proporción vertical balanceada: puedes jugar con tiro medio‑alto según el look (día vs noche).")

        # 2) Silueta / V‑taper
        if (v_taper_score is not None) and (v_taper_score < 55):
            actions_out.append("Sastrería: estructura suave en hombro + cintura limpia (pinzas/entallado ligero) para definir arquitectura.")
        elif (v_taper_score is not None) and (v_taper_score >= 75):
            actions_out.append("Sastrería: hombro ya proyecta; enfócate en telas con caída y líneas limpias (quiet luxury).")

        # 3) Alineación / presencia
        if (align_score is not None) and (align_score < 70):
            actions_out.append("Presencia editorial: hombros ‘abiertos’ 2–3° + mentón neutro mejora la caída del look en foto.")

        # 4) Perfil
        if (profile_score is not None) and (profile_score < 70):
            actions_out.append("Perfil: escote en V o solapa en punta limpia la línea superior y estiliza cuello visual.")

        # 5) Si faltan, completar con recomendaciones universales de alta costura (sin inventar medidas)
        fillers = [
            "Color/forma: una silueta limpia (monocromo o 2 tonos) se ve más alta costura en cámara.",
            "Fit: evita exceso de tela en cintura y cadera; busca líneas continuas y costuras alineadas.",
        ]
        for f in fillers:
            if len(actions_out) >= 7:
                break
            if f not in actions_out:
                actions_out.append(f)

        return actions_out[:8]

    couture = {
        "name": "Alta Costura Inteligente",
        "modules": {
            "silhouette_sculpt": {
                "signature": _silhouette_signature(proxies.get("shoulder_hip_ratio")),
                "v_taper_index": silhouette_v_taper,
            },
            "balance_torso_leg": {
                "signature": _vertical_signature(proxies.get("torso_leg_ratio")),
                "torso_leg_balance": torso_leg_balance,
            },
            "presence_alignment": {
                "presence_alignment_score": alignment_symmetry,
                "stance_grounding": stance_grounding,
            },
            "stacking_profile": {
                "profile_stack": profile_stack,
            },
        },
        "proxies": proxies,
    }

    couture_plan = None
    try:
        if decision == "accepted":
            vars_for_prios = {
                "alignment_symmetry": int(alignment_symmetry) if alignment_symmetry is not None else None,
                "silhouette_v_taper": int(silhouette_v_taper) if silhouette_v_taper is not None else None,
                "torso_leg_balance": int(torso_leg_balance) if torso_leg_balance is not None else None,
                "stance_grounding": int(stance_grounding) if stance_grounding is not None else None,
                "profile_stack": int(profile_stack) if profile_stack is not None else None,
            }
            vars_for_prios = {k: v for k, v in vars_for_prios.items() if isinstance(v, int)}
            prios = _pick_priorities(vars_for_prios)
            couture_plan = {
                "priorities": prios,
                "actions": _couture_actions(
                    r=proxies.get("shoulder_hip_ratio"),
                    tl=proxies.get("torso_leg_ratio"),
                    v_taper_score=silhouette_v_taper,
                    align_score=alignment_symmetry,
                    profile_score=profile_stack,
                ),
                "horizon_weeks": 4,
                "plan_4_weeks": [
                    "Semana 1: base impecable (fit + largos).",
                    "Semana 2: arquitectura (hombro–cintura) con sastrería y capas.",
                    "Semana 3: firma (cuello/escote/solapa + calzado).",
                    "Semana 4: repetición inteligente (2–3 looks núcleo y variaciones).",
                ],
                "note": "Recomendaciones ópticas (ratios proxy). No son medidas de alta costura en cm.",
            }
    except Exception:
        couture_plan = None

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
        "couture": couture,
        "couture_plan": couture_plan,
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
        "meta": {"algorithm": "exp-012_alta_costura_inteligente_v0", "as_of": str(date.today()), "legacy": "exp-012_shape_presence_v0"},
    }

    if baseline_delta is not None:
        payload_out["baseline_delta"] = {"overall_presence": int(baseline_delta)}

    return ShapePresenceResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    decision = str(result.get('decision') or '').strip().lower()
    conf = result.get('confidence') if isinstance(result.get('confidence'), dict) else {}
    couture = result.get('couture') if isinstance(result.get('couture'), dict) else {}
    plan = result.get('couture_plan') if isinstance(result.get('couture_plan'), dict) else {}
    vars_ = result.get('variables') if isinstance(result.get('variables'), dict) else {}

    confidence_pct = None
    try:
        if conf.get('score') is not None:
            confidence_pct = int(round(float(conf.get('score')) * 100.0))
    except Exception:
        confidence_pct = None

    lines: list[str] = []
    lines.append("**Alta Costura Inteligente (beta)**")
    lines.append("(Proporciones ópticas por foto; **no son medidas en cm**.)")

    if decision != 'accepted':
        lines.append("\n**⚠️ Necesito una foto mejor para medir**")
        lines.append("- Cuerpo completo (pies a cabeza)")
        lines.append("- Luz uniforme, sin contraluz")
        lines.append("- Cámara a 2–3m, a la altura del pecho")
        lines.append("- Frente relajado (mínimo) y, si puedes, perfil derecho")
        return "\n".join(lines).strip()

    if confidence_pct is not None:
        lines.append(f"\n**✅ Listo** · Confianza de captura: {confidence_pct}%")
    else:
        lines.append("\n**✅ Listo**")

    # Mapa (wow, escaneable)
    try:
        proxies = couture.get('proxies') if isinstance(couture.get('proxies'), dict) else {}
        sh = proxies.get('shoulder_hip_ratio')
        tl = proxies.get('torso_leg_ratio')

        sig1 = (couture.get('modules') or {}).get('silhouette_sculpt', {}) if isinstance((couture.get('modules') or {}).get('silhouette_sculpt'), dict) else {}
        sig2 = (couture.get('modules') or {}).get('balance_torso_leg', {}) if isinstance((couture.get('modules') or {}).get('balance_torso_leg'), dict) else {}

        lines.append("\n**🧵 Arquitectura visual (alta costura)**")
        if sig1.get('signature'):
            lines.append(f"- {str(sig1.get('signature')).strip()}")
        if sig2.get('signature'):
            lines.append(f"- {str(sig2.get('signature')).strip()}")

        # Scores
        lines.append("\n**📌 Índices (0–100)**")
        lines.append(f"- Presencia global: {int(vars_.get('overall_presence') or 0)}")
        lines.append(f"- Alineación: {int(vars_.get('alignment_symmetry') or 0)}")
        lines.append(f"- Silueta (V‑taper proxy): {int(vars_.get('silhouette_v_taper') or 0)}")
        lines.append(f"- Verticalidad (torso/pierna): {int(vars_.get('torso_leg_balance') or 0)}")
        if vars_.get('profile_stack') is not None:
            lines.append(f"- Perfil (stacking): {int(vars_.get('profile_stack') or 0)}")
    except Exception:
        pass

    # Prioridades + acciones
    prios = plan.get('priorities') if isinstance(plan.get('priorities'), list) else []
    prios = [str(x).strip() for x in prios if str(x).strip()]
    actions = plan.get('actions') if isinstance(plan.get('actions'), list) else []
    actions = [str(x).strip() for x in actions if str(x).strip()]

    lines.append("\n**🎯 Prioridades (4 semanas)**")
    if prios:
        for i, p in enumerate(prios[:3], start=1):
            lines.append(f"- Prioridad {i}: {p}")
    else:
        lines.append("- Prioridad 1: proporción vertical (tiro/largos)")
        lines.append("- Prioridad 2: alineación (línea limpia)")
        lines.append("- Prioridad 3: arquitectura de silueta")

    lines.append("\n**✅ Sugerencias personalizadas (alta costura)**")
    for a in actions[:8]:
        lines.append(f"- {a}")

    # Cierre
    lines.append("\nSi quieres afinarlo, repite la foto con la misma luz y encuadre 1 vez por semana.")
    return "\n".join(lines).strip()
