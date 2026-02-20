from __future__ import annotations

from typing import Any


def build_ui_blocks(qaf: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    if not isinstance(qaf, dict):
        return blocks

    total = qaf.get("total_calories")
    rng = qaf.get("total_calories_range") or {}
    low = rng.get("low")
    high = rng.get("high")
    driver_name = qaf.get("range_driver_display") or qaf.get("range_driver")

    if total is not None and low is not None and high is not None:
        blocks.append(
            {
                "type": "summary",
                "text": f"Estimado: {round(float(total), 0):.0f} kcal (rango {round(float(low), 0):.0f}–{round(float(high), 0):.0f})",
            }
        )
    if driver_name:
        blocks.append({"type": "range_driver", "text": f"Lo que más mueve el rango: {driver_name}"})

    macros = (qaf.get("macros_total") or {})
    if macros:
        p = float(macros.get("protein_g") or 0.0)
        c = float(macros.get("carbs_g") or 0.0)
        f = float(macros.get("fat_g") or 0.0)
        blocks.append({"type": "macros", "text": f"Macros aprox: P {p:.0f}g | C {c:.0f}g | G {f:.0f}g"})

    micros = qaf.get("micros_highlights") or []
    if micros:
        blocks.append({"type": "micros", "text": "Micros probables: " + ", ".join(micros[:5])})

    return blocks


def build_professional_chat_text(qaf: dict[str, Any]) -> str:
    """Texto corto y profesional (fallback) sin recomendaciones."""
    if not isinstance(qaf, dict):
        return ""

    lines = []
    for b in build_ui_blocks(qaf):
        if b.get("type") in {"summary", "range_driver", "macros"}:
            lines.append(str(b.get("text") or "").strip())

    follow = (qaf.get("follow_up_questions") or [])
    if qaf.get("needs_confirmation") and follow:
        prompt = str((follow[0] or {}).get("prompt") or "").strip()
        if prompt:
            lines.append(prompt)

    return "\n".join([x for x in lines if x]).strip()
