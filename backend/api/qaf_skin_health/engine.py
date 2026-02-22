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
    # 1) Uniformity: lower tone std => better
    s_uniformity = int(round(_clamp01(1.0 - (tone_uniformity / 0.28)) * 100.0))

    # 2) Texture: we want moderate microcontrast + not too high entropy. We'll penalize extremes.
    s_texture = int(round(_clamp01(1.0 - abs(microcontrast - 0.05) / 0.07) * 100.0))

    # 3) Hydration visible: lower specular (oil) but not zero; target ~0.03..0.08
    s_hydration = int(round(_clamp01(1.0 - abs(specular_glow - 0.05) / 0.08) * 100.0))

    # 4) Energy facial: combination of luminance and under-eye shadow
    s_energy = int(round(_clamp01((natural_luminance - 0.25) / 0.45) * 100.0))
    if under_eye_shadow is not None:
        # under_eye_shadow closer to 1 => less shadow
        s_energy = int(round(_clamp01((float(s_energy) / 100.0) * 0.65 + _clamp01((under_eye_shadow - 0.75) / 0.30) * 0.35) * 100.0))

    # redness: treat as warning if high; score peaks around mid.
    s_redness = int(round(_clamp01(1.0 - abs(redness_signal - 0.52) / 0.18) * 100.0))

    # Final Skin Health Score (quality-weighted)
    sub = {
        "hydration_visible": s_hydration,
        "uniformity": s_uniformity,
        "facial_energy": s_energy,
        "texture_quality": s_texture,
        "redness_balance": s_redness,
        "contrast_clarity": int(round(contrast_clarity * 100.0)),
    }

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

    return SkinHealthResult(payload=payload_out)


def render_professional_summary(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""

    lines: list[str] = []
    lines.append(f"decision: {result.get('decision')}")

    conf = result.get('confidence') if isinstance(result.get('confidence'), dict) else {}
    if conf.get('score') is not None:
        try:
            lines.append(f"confidence: {round(float(conf.get('score')), 3)}")
        except Exception:
            pass

    score = result.get('skin_health_score')
    if score is not None:
        try:
            lines.append(f"Skin Health Score™: {int(score)}/100")
        except Exception:
            pass

    sub = result.get('sub_scores') if isinstance(result.get('sub_scores'), dict) else {}
    if sub:
        parts = []
        for k, label in (
            ('hydration_visible', 'Hidratación visible'),
            ('uniformity', 'Uniformidad'),
            ('facial_energy', 'Energía facial'),
            ('texture_quality', 'Textura'),
        ):
            if sub.get(k) is None:
                continue
            try:
                parts.append(f"{label}: {int(sub.get(k))}/100")
            except Exception:
                pass
        if parts:
            lines.append(' · '.join(parts))

    insights = result.get('insights')
    if isinstance(insights, list) and insights:
        for x in insights[:3]:
            if str(x).strip():
                lines.append(f"note: {str(x).strip()}")

    return "\n".join(lines).strip()
