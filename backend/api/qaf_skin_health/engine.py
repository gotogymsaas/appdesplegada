from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


@dataclass(frozen=True)
class SkinHealthResult:
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


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * float(p)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(vals[f])
    return float(vals[f] * (c - k) + vals[c] * (k - f))


def _entropy_from_hist(hist: list[int]) -> float:
    total = float(sum(hist) or 0.0)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in hist:
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * math.log(p + 1e-12)
    return float(h)


def _variance_of_laplacian(gray: "np.ndarray") -> float:
    # kernel: 4-neighbor laplacian
    k = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    g = gray.astype(np.float32)
    # convolution (valid)
    out = (
        k[0, 0] * g[:-2, :-2]
        + k[0, 1] * g[:-2, 1:-1]
        + k[0, 2] * g[:-2, 2:]
        + k[1, 0] * g[1:-1, :-2]
        + k[1, 1] * g[1:-1, 1:-1]
        + k[1, 2] * g[1:-1, 2:]
        + k[2, 0] * g[2:, :-2]
        + k[2, 1] * g[2:, 1:-1]
        + k[2, 2] * g[2:, 2:]
    )
    return float(np.var(out))


def _center_crop(im: Image.Image, frac_w: float = 0.66, frac_h: float = 0.72) -> Image.Image:
    w, h = im.size
    cw = int(max(1, min(w, round(w * float(frac_w)))))
    ch = int(max(1, min(h, round(h * float(frac_h)))))
    left = int(max(0, (w - cw) // 2))
    top = int(max(0, (h - ch) // 2))
    return im.crop((left, top, left + cw, top + ch))


def _roi(arr: "np.ndarray", x0: float, y0: float, x1: float, y1: float) -> "np.ndarray":
    h, w = arr.shape[:2]
    xa = int(_clamp(x0, 0.0, 1.0) * w)
    xb = int(_clamp(x1, 0.0, 1.0) * w)
    ya = int(_clamp(y0, 0.0, 1.0) * h)
    yb = int(_clamp(y1, 0.0, 1.0) * h)
    if xb <= xa or yb <= ya:
        return arr[0:0, 0:0]
    return arr[ya:yb, xa:xb]


def evaluate_skin_health(
    *,
    image_bytes: bytes,
    content_type: str | None = None,
    context: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> SkinHealthResult:
    """Skin Health Intelligence System (MVP) basado en una sola foto.

    - No hace diagnóstico médico.
    - No hace juicio estético.
    - Si la calidad es mala: `needs_confirmation`.

    Variables de observación (imagen):
    - tone_uniformity
    - texture_microcontrast
    - natural_luminance
    - under_eye_shadow
    - specular_glow (hidratación aparente)
    - patchiness
    +2 nuevas:
    - redness_signal (proxy de irritación/enrojecimiento)
    - contrast_clarity (proxy de “energía visual” por contraste controlado)
    """

    if Image is None or np is None:
        return SkinHealthResult(
            payload={
                "decision": "needs_confirmation",
                "decision_reason": "dependencies_missing",
                "confidence": {"score": 0.0, "uncertainty_score": 1.0, "missing": ["pillow_or_numpy"]},
                "observations": {},
                "scores": {},
                "insights": ["No puedo analizar imagen en este entorno. Intenta de nuevo."],
                "follow_up_questions": [],
                "meta": {"algorithm": "exp-011_skin_health_v0", "as_of": str(date.today())},
            }
        )

    ctx = context or {}

    # Decode
    try:
        im = Image.open(__import__("io").BytesIO(image_bytes))
        im = im.convert("RGB")
    except Exception:
        return SkinHealthResult(
            payload={
                "decision": "needs_confirmation",
                "decision_reason": "image_decode_failed",
                "confidence": {"score": 0.0, "uncertainty_score": 1.0, "missing": ["image_decode"]},
                "observations": {},
                "scores": {},
                "insights": ["No pude leer la foto. Intenta con otra imagen."],
                "follow_up_questions": [],
                "meta": {"algorithm": "exp-011_skin_health_v0", "as_of": str(date.today())},
            }
        )

    # Heurística de “cara centrada”: usamos crop central y medimos si hay suficiente detalle.
    im_c = _center_crop(im)

    # Convert to arrays
    rgb = np.asarray(im_c).astype(np.float32) / 255.0  # H,W,3

    # Luminance (Rec. 709-ish)
    y = (0.2126 * rgb[:, :, 0]) + (0.7152 * rgb[:, :, 1]) + (0.0722 * rgb[:, :, 2])

    # Saturation proxy
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)

    # Quality metrics
    p05 = float(np.quantile(y, 0.05))
    p50 = float(np.quantile(y, 0.50))
    p95 = float(np.quantile(y, 0.95))
    sat_hi = float(np.mean((y >= 0.98).astype(np.float32)))
    sat_lo = float(np.mean((y <= 0.03).astype(np.float32)))

    gray_u8 = (y * 255.0).clip(0, 255).astype(np.uint8)
    blur_var = float(_variance_of_laplacian(gray_u8))

    # Normalize quality
    q_focus = _clamp01((blur_var - 20.0) / (180.0 - 20.0))
    q_exposure = 1.0 - _clamp01((abs(p50 - 0.55) / 0.25))
    q_dynamic = _clamp01((p95 - p05 - 0.25) / (0.70 - 0.25))
    q_saturation = 1.0 - _clamp01((sat_hi + sat_lo - 0.02) / 0.18)

    # Filter suspicion: EXIF Software tag (heurística, no certeza)
    filter_suspected = False
    try:
        exif = im.getexif() if hasattr(im, "getexif") else None
        if exif:
            software = str(exif.get(305) or "").lower()  # 305=Software
            if any(k in software for k in ("instagram", "snap", "picsart", "beauty", "facetune")):
                filter_suspected = True
    except Exception:
        filter_suspected = False

    q_filter = 0.75 if filter_suspected else 1.0

    quality_score = _clamp01((0.30 * q_focus) + (0.28 * q_exposure) + (0.22 * q_dynamic) + (0.20 * q_saturation))
    quality_score = float(_clamp01(float(quality_score) * float(q_filter)))

    decision = "accepted" if quality_score >= 0.60 else "needs_confirmation"
    decision_reason = "ok" if decision == "accepted" else "low_quality_capture"

    follow_up_questions: list[dict[str, Any]] = []
    if decision != "accepted":
        follow_up_questions.append(
            {
                "type": "retake_face_photo",
                "prompt": "Para un análisis confiable necesito luz natural, sin contraluz y el rostro centrado. ¿Puedes repetir la foto?",
                "options": [],
            }
        )

    # Convert to LAB using PIL (0..255 channels). We'll normalize to 0..1.
    lab = np.asarray(im_c.convert("LAB")).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    A = lab[:, :, 1] / 255.0
    B = lab[:, :, 2] / 255.0

    # Regions (rough, face-centered assumption)
    # under-eye: mid-upper band; cheek: lower-mid band
    under = _roi(L, 0.22, 0.40, 0.78, 0.55)
    cheek = _roi(L, 0.22, 0.62, 0.78, 0.78)
    tzone = _roi(L, 0.35, 0.38, 0.65, 0.70)

    # Observations
    tone_uniformity = float(np.std(A) + np.std(B))

    # Texture: entropy + high-pass energy proxy
    hist = np.histogram(gray_u8.flatten(), bins=64, range=(0, 255))[0].tolist()
    ent = _entropy_from_hist(hist)  # ~0..4
    hp = gray_u8.astype(np.float32) - np.mean(gray_u8.astype(np.float32))
    microcontrast = float(np.std(hp) / 255.0)

    natural_luminance = float(np.mean(L))

    under_eye_shadow = None
    try:
        mu_under = float(np.mean(under)) if under.size else None
        mu_cheek = float(np.mean(cheek)) if cheek.size else None
        if mu_under is not None and mu_cheek is not None and mu_cheek > 1e-6:
            under_eye_shadow = float(mu_under / mu_cheek)
    except Exception:
        under_eye_shadow = None

    # Specular glow: high luminance but low saturation -> highlights
    specular_glow = float(np.mean(((y > 0.82) & (sat < 0.25)).astype(np.float32)))

    # Patchiness: grid std of L means
    patchiness = 0.0
    try:
        gh, gw = 6, 6
        hs, ws = L.shape[0] // gh, L.shape[1] // gw
        means = []
        for i in range(gh):
            for j in range(gw):
                block = L[i * hs : (i + 1) * hs, j * ws : (j + 1) * ws]
                if block.size:
                    means.append(float(np.mean(block)))
        if means:
            patchiness = float(np.std(means))
    except Exception:
        patchiness = 0.0

    # +2 variables nuevas (observación)
    # redness_signal: a* mean deviation (proxy)
    redness_signal = float(np.mean(A))

    # contrast_clarity: combine microcontrast and dynamic range
    contrast_clarity = float(_clamp01((microcontrast / 0.10) * 0.6 + (q_dynamic) * 0.4))

    # Normalize observations to 0..100 sub-scores (higher is "healthier")
    # Nota: son proxies, no clínica.
    # 1) Uniformity: lower tone std => better.
    # Antes era un clamp lineal muy agresivo que llevaba a 0% fácil. Ahora usamos una caída exponencial suave.
    try:
        s_uniformity = int(round(_clamp01(math.exp(-float(tone_uniformity) / 0.20)) * 100.0))
    except Exception:
        s_uniformity = int(round(_clamp01(1.0 - (tone_uniformity / 0.40)) * 100.0))

    # 2) Texture: buscamos microcontraste moderado.
    # Usar una caída exponencial evita "0%" por clamps agresivos.
    try:
        s_texture = int(round(_clamp01(math.exp(-abs(float(microcontrast) - 0.05) / 0.08)) * 100.0))
    except Exception:
        s_texture = int(round(_clamp01(1.0 - abs(microcontrast - 0.05) / 0.12) * 100.0))

    # 3) Hydration visible: lower specular (oil) but not zero; target ~0.03..0.08
    s_hydration = int(round(_clamp01(1.0 - abs(specular_glow - 0.05) / 0.08) * 100.0))

    # 4) Energy facial: combination of luminance and under-eye shadow
    s_energy = int(round(_clamp01((natural_luminance - 0.25) / 0.45) * 100.0))
    if under_eye_shadow is not None:
        # under_eye_shadow closer to 1 => less shadow
        s_energy = int(round(_clamp01((float(s_energy) / 100.0) * 0.65 + _clamp01((under_eye_shadow - 0.75) / 0.30) * 0.35) * 100.0))

    # redness: score máximo cerca de 0.52; caída exponencial para evitar 0% por ruido de iluminación.
    try:
        s_redness = int(round(_clamp01(math.exp(-abs(float(redness_signal) - 0.52) / 0.22)) * 100.0))
    except Exception:
        s_redness = int(round(_clamp01(1.0 - abs(redness_signal - 0.52) / 0.30) * 100.0))

    # Final Skin Health Score (quality-weighted)
    sub = {
        "hydration_visible": s_hydration,
        "uniformity": s_uniformity,
        "facial_energy": s_energy,
        "texture_quality": s_texture,
        "redness_balance": s_redness,
        "contrast_clarity": int(round(contrast_clarity * 100.0)),
    }

    # UX: si aceptamos la foto, evitar 0% duros por artefactos de la métrica.
    # No altera el score principal de forma importante, solo evita lecturas "0%" confusas.
    if decision == "accepted":
        for k in ("hydration_visible", "uniformity", "facial_energy", "texture_quality", "redness_balance", "contrast_clarity"):
            try:
                v = int(sub.get(k) or 0)
                if v <= 0:
                    sub[k] = 1
            except Exception:
                continue

    # Score principal usa 4 sub-scores (como en el texto) + guardrail por calidad
    base_score = (
        0.30 * float(s_hydration)
        + 0.26 * float(s_uniformity)
        + 0.24 * float(s_energy)
        + 0.20 * float(s_texture)
    )
    skin_health_score = int(round(_clamp01(float(quality_score)) * _clamp(float(base_score), 0.0, 100.0)))

    # Confidence
    conf = float(quality_score)
    if baseline and isinstance(baseline, dict):
        conf = float(_clamp01(conf * 0.85 + 0.15))
    confidence = float(_clamp01(conf))
    uncertainty = float(_clamp01(1.0 - confidence))

    # Insights tipo coach (prudentes)
    insights: list[str] = []

    if decision != "accepted":
        insights.append("La foto no tiene calidad suficiente para un análisis confiable. Con mejor luz natural el resultado será más preciso.")
    else:
        insights.append(f"Skin Health Score™: {skin_health_score}/100 (confianza: {int(round(confidence*100))}%).")
        insights.append(f"Sub-scores — Hidratación visible: {s_hydration}/100 · Uniformidad: {s_uniformity}/100 · Energía facial: {s_energy}/100 · Textura: {s_texture}/100")

        # Conexión con hábitos (solo como hipótesis; no diagnóstico)
        sleep_min = _safe_float(ctx.get("sleep_minutes"))
        stress = _safe_float(ctx.get("stress_1_5"))
        water = _safe_float(ctx.get("water_liters"))
        if sleep_min is not None and sleep_min > 0 and sleep_min < 420 and s_energy < 70:
            insights.append("La energía facial está algo baja. Esto a veces coincide con sueño < 7h. Si quieres, revisamos tu descanso esta semana.")
        if water is not None and water < 1.5 and s_hydration < 70:
            insights.append("Veo señales leves de hidratación aparente baja. Si tu ingesta de agua ha estado baja, subirla puede ayudar (junto con sueño).")
        if stress is not None and stress >= 4 and (s_energy < 75 or s_uniformity < 75):
            insights.append("El estrés alto suele reflejarse en energía visible y uniformidad. Si quieres, ajustamos micro-hábitos esta semana.")

    observations = {
        "tone_uniformity": round(float(tone_uniformity), 6),
        "texture_microcontrast": round(float(microcontrast), 6),
        "natural_luminance": round(float(natural_luminance), 6),
        "under_eye_shadow": (round(float(under_eye_shadow), 6) if under_eye_shadow is not None else None),
        "specular_glow": round(float(specular_glow), 6),
        "patchiness": round(float(patchiness), 6),
        "redness_signal": round(float(redness_signal), 6),
        "contrast_clarity": round(float(contrast_clarity), 6),
        "quality": {
            "score": round(float(quality_score), 6),
            "q_focus": round(float(q_focus), 6),
            "q_exposure": round(float(q_exposure), 6),
            "q_dynamic": round(float(q_dynamic), 6),
            "q_saturation": round(float(q_saturation), 6),
            "filter_suspected": bool(filter_suspected),
            "p05": round(float(p05), 6),
            "p50": round(float(p50), 6),
            "p95": round(float(p95), 6),
            "sat_hi": round(float(sat_hi), 6),
            "sat_lo": round(float(sat_lo), 6),
            "blur_var": round(float(blur_var), 3),
        },
    }

    payload_out = {
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": {
            "score": round(float(confidence), 4),
            "uncertainty_score": round(float(uncertainty), 4),
        },
        "skin_health_score": int(skin_health_score),
        "sub_scores": {
            "hydration_visible": int(s_hydration),
            "uniformity": int(s_uniformity),
            "facial_energy": int(s_energy),
            "texture_quality": int(s_texture),
        },
        "progress": None,
        "context_signals": {},
        "recommendation_plan": None,
        "observations": observations,
        "extra_observations": {
            "redness_balance": int(s_redness),
            "contrast_clarity": int(round(contrast_clarity * 100.0)),
        },
        "scores": sub,
        "insights": insights[:6],
        "follow_up_questions": follow_up_questions,
        "meta": {"algorithm": "exp-011_skin_health_v0", "as_of": str(date.today())},
    }

    # Señales de contexto (si existen). No son obligatorias.
    try:
        ctx_out: dict[str, Any] = {}
        for k in (
            'sleep_minutes',
            'stress_1_5',
            'steps',
            'movement_1_5',
            'water_liters',
            'sun_minutes',
        ):
            if ctx.get(k) is None:
                continue
            ctx_out[k] = ctx.get(k)
        payload_out['context_signals'] = ctx_out
    except Exception:
        payload_out['context_signals'] = {}

    # Motor de recomendación inteligente (no médico): prioridades + acciones simples.
    try:
        if decision == 'accepted':
            # Necesidades (0..1). Mientras más alto, más prioridad.
            need_hyd = _clamp01((70.0 - float(s_hydration)) / 70.0)
            need_sleep = 0.0
            try:
                sleep_min = _safe_float(ctx.get('sleep_minutes'))
                if sleep_min is not None and sleep_min > 0:
                    # <7h => sube necesidad
                    need_sleep = max(need_sleep, _clamp01((420.0 - float(sleep_min)) / 180.0))
            except Exception:
                pass
            need_sleep = max(need_sleep, _clamp01((70.0 - float(s_energy)) / 70.0))

            s_red = int(sub.get('redness_balance') or 0)
            need_infl = _clamp01((70.0 - float(s_red)) / 70.0)
            try:
                if float(patchiness) >= 0.06:
                    need_infl = max(need_infl, 0.55)
            except Exception:
                pass

            # Ajustes por contexto (si existe)
            water_l = _safe_float(ctx.get('water_liters'))
            if water_l is not None and water_l < 1.6:
                need_hyd = max(need_hyd, 0.60)

            stress_1_5 = _safe_float(ctx.get('stress_1_5'))
            if stress_1_5 is not None and stress_1_5 >= 4:
                need_infl = max(need_infl, 0.55)
                need_sleep = max(need_sleep, 0.40)

            steps = _safe_float(ctx.get('steps'))
            movement_1_5 = _safe_float(ctx.get('movement_1_5'))
            need_movement = 0.0
            if steps is not None and steps >= 0:
                need_movement = _clamp01((6000.0 - float(steps)) / 6000.0)
            elif movement_1_5 is not None:
                # 1 = bajo movimiento (alta necesidad), 5 = excelente (baja necesidad)
                m = max(1.0, min(5.0, float(movement_1_5)))
                need_movement = _clamp01((5.0 - m) / 4.0)

            sun_min = _safe_float(ctx.get('sun_minutes'))
            need_sun = 0.0
            if sun_min is not None and sun_min > 20:
                need_sun = _clamp01((float(sun_min) - 20.0) / 60.0)

            # Priorización: seleccionar top-3 por necesidad.
            # Importante: NO forzamos siempre las mismas 3, para evitar respuestas genéricas.
            candidates = [
                ('hidratación', need_hyd),
                ('sueño', need_sleep),
                ('reducción de inflamación visible', need_infl),
                ('protección solar', need_sun),
                ('movimiento diario', need_movement),
            ]
            sorted_all = sorted(candidates, key=lambda t: float(t[1]), reverse=True)
            ordered = [name for name, _w in sorted_all if name]
            ordered = ordered[:3]

            # Si todo está muy bajo, damos una guía “mantenimiento” pero no repetitiva.
            try:
                top_w = float(sorted_all[0][1]) if sorted_all else 0.0
            except Exception:
                top_w = 0.0
            if top_w < 0.18:
                ordered = ['protección solar', 'hidratación', 'sueño']

            # Acciones simples (sin activos médicos)
            actions_simple: list[str] = []
            for p in ordered:
                if p == 'hidratación' and '+500ml de agua hoy' not in actions_simple:
                    actions_simple.append('+500ml de agua hoy')
                elif p == 'sueño' and 'Rutina nocturna básica: 30 min sin pantallas + dormir 30 min antes' not in actions_simple:
                    actions_simple.append('Rutina nocturna básica: 30 min sin pantallas + dormir 30 min antes')
                elif p == 'reducción de inflamación visible':
                    # Preferir respiración si hay estrés/señal de inflamación; si no, cuidado gentil.
                    if (stress_1_5 is not None and stress_1_5 >= 3.2) or (need_infl >= 0.45):
                        actions_simple.append('5 min de respiración lenta (inhala 4, exhala 6)')
                    else:
                        actions_simple.append('Rutina mínima hoy: limpieza gentil + hidratación simple (sin frotar fuerte)')
                elif p == 'protección solar':
                    actions_simple.append('Si vas a salir: protector solar y reaplicar si hay sol directo')
                elif p == 'movimiento diario':
                    actions_simple.append('Caminata suave 10–15 min (mejora circulación y “energía” en tendencia)')

            # Guardrail: si por alguna razón quedaron <3, completar con alternativas seguras.
            fillers = [
                'Rutina mínima hoy: limpieza gentil + hidratación simple (sin frotar fuerte)',
                '5 min de respiración lenta (inhala 4, exhala 6)',
                'Si vas a salir: protector solar y reaplicar si hay sol directo',
                '+500ml de agua hoy',
                'Rutina nocturna básica: 30 min sin pantallas + dormir 30 min antes',
            ]
            for f in fillers:
                if len(actions_simple) >= 3:
                    break
                if f not in actions_simple:
                    actions_simple.append(f)

            payload_out['recommendation_plan'] = {
                'priorities': ordered,
                'actions': actions_simple[:3],
                'note': 'Plan no médico basado en tendencia visual + contexto disponible.'
            }
    except Exception:
        payload_out['recommendation_plan'] = None

    # Progreso vs baseline (semana pasada): deltas + % cuando sea posible
    try:
        if baseline and isinstance(baseline, dict):
            prev_score = baseline.get('skin_health_score')
            prev_sub = baseline.get('sub_scores') if isinstance(baseline.get('sub_scores'), dict) else {}

            deltas: dict[str, dict[str, Any]] = {}
            try:
                prev_i = int(prev_score) if prev_score is not None else None
                if prev_i is not None:
                    now_i = int(skin_health_score)
                    d = int(now_i - prev_i)
                    pct = (float(d) / float(prev_i) * 100.0) if prev_i not in (0, None) else None
                    deltas['skin_health_score'] = {
                        'prev': prev_i,
                        'now': now_i,
                        'delta': d,
                        'pct': (round(float(pct), 2) if pct is not None else None),
                    }
            except Exception:
                pass

            for k, cur in (
                ('hydration_visible', s_hydration),
                ('uniformity', s_uniformity),
                ('facial_energy', s_energy),
                ('texture_quality', s_texture),
            ):
                try:
                    prev_v = prev_sub.get(k)
                    prev_i = int(prev_v) if prev_v is not None else None
                    if prev_i is None:
                        continue
                    now_i = int(cur)
                    d = int(now_i - prev_i)
                    pct = (float(d) / float(prev_i) * 100.0) if prev_i not in (0, None) else None
                    deltas[k] = {
                        'prev': prev_i,
                        'now': now_i,
                        'delta': d,
                        'pct': (round(float(pct), 2) if pct is not None else None),
                    }
                except Exception:
                    continue

            if deltas:
                payload_out['progress'] = {
                    'baseline_source': 'last_week',
                    'vs_last_week': {
                        'available': True,
                        'deltas': deltas,
                    },
                }
    except Exception:
        pass

    return SkinHealthResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    user_display_name = str(result.get('user_display_name') or '').strip()
    hello = f"Hola {user_display_name}," if user_display_name else "Hola,"

    decision = str(result.get('decision') or '').strip().lower()
    conf = result.get('confidence') if isinstance(result.get('confidence'), dict) else {}
    obs = result.get('observations') if isinstance(result.get('observations'), dict) else {}
    q = obs.get('quality') if isinstance(obs.get('quality'), dict) else {}
    progress = result.get('progress') if isinstance(result.get('progress'), dict) else {}

    score = None
    try:
        score = int(result.get('skin_health_score')) if result.get('skin_health_score') is not None else None
    except Exception:
        score = None

    confidence_pct = None
    try:
        if conf.get('score') is not None:
            confidence_pct = int(round(float(conf.get('score')) * 100.0))
    except Exception:
        confidence_pct = None

    sub = result.get('sub_scores') if isinstance(result.get('sub_scores'), dict) else {}
    plan = result.get('recommendation_plan') if isinstance(result.get('recommendation_plan'), dict) else {}
    ctx_sig = result.get('context_signals') if isinstance(result.get('context_signals'), dict) else {}

    def _sub_int(key: str) -> int | None:
        v = sub.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    s_h = _sub_int('hydration_visible')
    s_u = _sub_int('uniformity')
    s_e = _sub_int('facial_energy')
    s_t = _sub_int('texture_quality')

    filter_suspected = bool(q.get('filter_suspected'))

    lines: list[str] = [hello]
    lines.append("**🔹 Vitalidad de la Piel (Skin Health · beta)**")
    lines.append("(Lectura visual de tendencia; **no es diagnóstico médico**.)")

    if decision != 'accepted':
        lines.append("\n**⚠️ Esta foto no funciona para un análisis confiable**")
        lines.append("No te voy a inventar resultados: con esta captura el sistema no puede medir tendencia con precisión.")
        lines.append("\n**📷 Cómo tomarla para que sí mida**")
        lines.append("- Luz natural de frente (sin contraluz)")
        lines.append("- Sin filtros y sin modo belleza")
        lines.append("- Rostro centrado (sin zoom extremo)")
        lines.append("- Si puedes: misma hora/luz cada semana (eso crea progreso real)")
        return "\n".join(lines).strip()

    lines.append("\n**✅ Listo**")
    if score is not None:
        if confidence_pct is not None:
            lines.append(f"- Vitalidad Score™: {score}% · Confianza: {confidence_pct}%")
        else:
            lines.append(f"- Vitalidad Score™: {score}%")

    # Mostrar TODAS las variables clave (wow) en formato escaneable.
    try:
        scores_all = result.get('scores') if isinstance(result.get('scores'), dict) else {}
        redness_balance = None
        contrast_clarity = None
        try:
            if scores_all.get('redness_balance') is not None:
                redness_balance = int(scores_all.get('redness_balance'))
        except Exception:
            redness_balance = None
        try:
            if scores_all.get('contrast_clarity') is not None:
                contrast_clarity = int(scores_all.get('contrast_clarity'))
        except Exception:
            contrast_clarity = None

        lines.append("\n**📌 Lectura visual (0–100)**")
        if s_h is not None:
            lines.append(f"- Hidratación visible: {s_h}%")
        if s_u is not None:
            lines.append(f"- Uniformidad de tono: {s_u}%")
        if s_e is not None:
            lines.append(f"- Energía facial: {s_e}%")
        if s_t is not None:
            lines.append(f"- Textura (micro‑detalle): {s_t}%")
        if redness_balance is not None:
            lines.append(f"- Balance de rojez (proxy): {redness_balance}%")
        if contrast_clarity is not None:
            lines.append(f"- Claridad/contraste (proxy): {contrast_clarity}%")
    except Exception:
        pass

    if filter_suspected:
        lines.append("- Nota: detecté indicios de filtro/edición; puede distorsionar la tendencia.")

    # Calidad de captura (para transparencia)
    try:
        q_score = q.get('score')
        q_focus = q.get('q_focus')
        q_exposure = q.get('q_exposure')
        q_dynamic = q.get('q_dynamic')
        q_saturation = q.get('q_saturation')

        parts = []
        if q_score is not None:
            try:
                parts.append(f"Calidad: {int(round(float(q_score) * 100.0))}%")
            except Exception:
                pass
        if q_focus is not None:
            try:
                parts.append(f"Enfoque: {int(round(float(q_focus) * 100.0))}%")
            except Exception:
                pass
        if q_exposure is not None:
            try:
                parts.append(f"Luz: {int(round(float(q_exposure) * 100.0))}%")
            except Exception:
                pass
        if q_dynamic is not None:
            try:
                parts.append(f"Rango: {int(round(float(q_dynamic) * 100.0))}%")
            except Exception:
                pass
        if q_saturation is not None:
            try:
                parts.append(f"Color: {int(round(float(q_saturation) * 100.0))}%")
            except Exception:
                pass
        if parts:
            lines.append("\n**📷 Calidad de captura (transparencia)**")
            lines.append("- " + " · ".join(parts[:5]))
    except Exception:
        pass

    # IA contextual (si hay señales)
    try:
        ctx_parts: list[str] = []
        ctx_insights: list[str] = []

        def _fmt_minutes_as_hm(minutes: float) -> str:
            m = int(round(float(minutes)))
            h = max(0, m) // 60
            mm = max(0, m) % 60
            return f"{h}h {mm:02d}m"
        sm = ctx_sig.get('sleep_minutes')
        if sm is not None:
            try:
                sm_i = float(sm)
                ctx_parts.append(f"Sueño: {_fmt_minutes_as_hm(sm_i)}")
                if sm_i < 420:
                    deficit = int(round(420 - sm_i))
                    ctx_insights.append(f"Sueño bajo hoy (déficit aprox: {_fmt_minutes_as_hm(deficit)}).")
                elif sm_i >= 480:
                    ctx_insights.append("Sueño sólido hoy (buena base para que la piel se vea más estable).")
            except Exception:
                pass
        st = ctx_sig.get('stress_1_5')
        if st is not None:
            try:
                st_f = float(st)
                ctx_parts.append(f"Estrés (1–5): {st_f:.1f}")
                if st_f >= 4.0:
                    ctx_insights.append("Estrés alto: suele reflejarse como menos energía visible y más ‘carga’ en el rostro.")
                elif st_f <= 2.0:
                    ctx_insights.append("Estrés bajo: buen punto para sostener uniformidad y claridad.")
            except Exception:
                pass
        steps = ctx_sig.get('steps')
        if steps is not None:
            try:
                steps_i = int(float(steps))
                ctx_parts.append(f"Pasos: {steps_i}")
                if steps_i < 5000:
                    ctx_insights.append("Movimiento bajo: subirlo un poco mejora circulación y ‘energía’ visual en tendencia.")
            except Exception:
                pass
        mv = ctx_sig.get('movement_1_5')
        if mv is not None:
            try:
                mv_f = float(mv)
                ctx_parts.append(f"Movimiento (1–5): {mv_f:.1f}")
                if mv_f <= 2.0:
                    ctx_insights.append("Movimiento bajo: una caminata corta + respiración lenta suele cambiar tu ‘vitalidad’ en el día.")
            except Exception:
                pass
        wl = ctx_sig.get('water_liters')
        if wl is not None:
            try:
                wl_f = float(wl)
                ctx_parts.append(f"Agua: {wl_f:.1f} L")
                if wl_f < 1.5:
                    ctx_insights.append("Agua baja: hoy tu piel tiende a verse menos ‘elástica’ y con menos brillo saludable.")
            except Exception:
                pass
        sunm = ctx_sig.get('sun_minutes')
        if sunm is not None:
            try:
                sun_i = int(float(sunm))
                ctx_parts.append(f"Sol: {sun_i} min")
                if sun_i >= 20:
                    ctx_insights.append("Si hubo sol directo: protector solar + reaplicar mantiene uniformidad en tendencia.")
            except Exception:
                pass

        if ctx_parts:
            lines.append("\n**🧠 IA contextual (tu piel refleja tu sistema)**")
            lines.append("- " + " · ".join(ctx_parts))
            if ctx_insights:
                # Máximo 2 para mantenerlo escaneable
                lines.append("- Correlación visible: " + " ".join(ctx_insights[:2]))
            lines.append("- Si ajustas 1–2 hábitos hoy, normalmente la piel lo refleja en tendencia (no es diagnóstico).")
    except Exception:
        pass

    # Cambios vs semana pasada (si existe)
    try:
        vs = progress.get('vs_last_week') if isinstance(progress.get('vs_last_week'), dict) else None
        if vs and isinstance(vs.get('deltas'), dict):
            deltas = vs.get('deltas')

            def _fmt(key: str, label: str) -> str | None:
                row = deltas.get(key) if isinstance(deltas.get(key), dict) else None
                if not row:
                    return None
                try:
                    d = int(row.get('delta') or 0)
                    prev = row.get('prev')
                    sign = "+" if d >= 0 else ""
                    pct = row.get('pct')
                    if pct is not None:
                        return f"- {label}: {sign}{d} pts ({sign}{float(pct):.1f}%)"
                    if prev is not None and int(prev) > 0:
                        pct2 = float(d) / float(int(prev)) * 100.0
                        return f"- {label}: {sign}{d} pts ({sign}{pct2:.1f}%)"
                    return f"- {label}: {sign}{d} pts"
                except Exception:
                    return None

            parts = [
                _fmt('skin_health_score', 'Score'),
                _fmt('hydration_visible', 'Hidratación'),
                _fmt('uniformity', 'Uniformidad'),
                _fmt('facial_energy', 'Energía'),
            ]
            parts = [p for p in parts if p]
            if parts:
                lines.append("\n**📈 Cambios vs semana pasada**")
                lines.extend(parts[:3])
    except Exception:
        pass

    # Recomendaciones seguras (no médicas)
    lines.append("\n**🎯 Prioridades de hoy**")
    try:
        prios = plan.get('priorities') if isinstance(plan.get('priorities'), list) else []
        if prios:
            for i, p in enumerate([str(x).strip() for x in prios if str(x).strip()][:3], start=1):
                lines.append(f"- Prioridad {i}: {p}")
        else:
            lines.append("- Prioridad 1: hidratación")
            lines.append("- Prioridad 2: sueño")
            lines.append("- Prioridad 3: reducción de inflamación visible")
    except Exception:
        lines.append("- Prioridad 1: hidratación")
        lines.append("- Prioridad 2: sueño")
        lines.append("- Prioridad 3: reducción de inflamación visible")

    lines.append("\n**✅ Acciones simples (hoy)**")
    try:
        acts = plan.get('actions') if isinstance(plan.get('actions'), list) else []
        acts = [str(x).strip() for x in acts if str(x).strip()]
        for a in acts[:3]:
            lines.append(f"- {a}")
    except Exception:
        lines.append("- +500ml de agua hoy")
        lines.append("- 5 min de respiración lenta")
        lines.append("- Rutina nocturna básica")

    # Nota final contextual (coherente con lo que medimos; sin diagnóstico)
    try:
        scores_all = result.get('scores') if isinstance(result.get('scores'), dict) else {}
        redness_balance = None
        try:
            if scores_all.get('redness_balance') is not None:
                redness_balance = int(scores_all.get('redness_balance'))
        except Exception:
            redness_balance = None

        patchiness = None
        try:
            patchiness = obs.get('patchiness')
            if patchiness is not None:
                patchiness = float(patchiness)
        except Exception:
            patchiness = None

        note_bits: list[str] = []
        if redness_balance is not None and redness_balance < 40:
            note_bits.append("Veo una señal alta de **rojez/irritación** (proxy). Si sientes **ardor o picazón** que no baja, pausa cambios y busca orientación profesional.")
        elif redness_balance is not None and redness_balance < 55:
            note_bits.append("Hay una señal moderada de **rojez** (proxy). Hoy prioriza rutina gentil; si hay ardor/picazón persistente, busca orientación profesional.")
        if s_h is not None and s_h < 55:
            note_bits.append("Si aparece **tirantez** o sensibilidad, prioriza suavidad (limpieza gentil + hidratación simple).")
        if s_t is not None and s_t < 55:
            note_bits.append("Si notas que la piel se ve más reactiva con cambios rápidos, vuelve a una rutina mínima por 3–5 días.")
        if patchiness is not None and patchiness >= 0.07:
            note_bits.append("Si ves zonas muy irregulares y se mantiene, repite la medición con la misma luz para confirmar tendencia.")

        # Si nada “alerta” aparece, cerrar con seguimiento.
        if not note_bits:
            note_bits.append("Si algo se siente fuera de lo normal o el cambio empeora de forma persistente, busca orientación profesional.")

        lines.append("\n**🧩 Nota final (coherente con tu lectura)**")
        for b in note_bits[:2]:
            lines.append(f"- {b}")
    except Exception:
        lines.append("\n**🧩 Nota final**")
        lines.append("- Si algo se siente fuera de lo normal o el cambio empeora de forma persistente, busca orientación profesional.")
    return "\n".join(lines).strip()
