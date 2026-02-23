from api.qaf_posture_proportion.engine import render_professional_summary


def test_posture_proportion_renderer_premium_has_header_and_disclaimer():
    text = render_professional_summary(
        {
            "decision": "accepted",
            "confidence": {"score": 0.9},
            "variables": {
                "postural_efficiency_score": 80,
                "posture_score": 78,
                "proportion_score": 70,
                "alignment_silhouette_index": 75,
            },
            "insights": ["Alineación estable."],
            "patterns": ["rounded_shoulders"],
            "immediate_corrections": [
                {"title": "Chin tucks", "cue": "Barbilla atrás", "duration_sec": 45},
                {"title": "Retracción escapular", "cue": "Hombros abajo y atrás", "duration_sec": 45},
            ],
            "weekly_adjustment": {"focus": ["core", "movilidad torácica"]},
        }
    )

    assert "**Arquitectura Corporal (QAF)**" in text
    assert "no son medidas en cm" in text.lower()
    assert "Lectura ejecutiva" in text
    assert "Micro" in text
    assert "%" in text


def test_posture_proportion_renderer_needs_photos_message():
    text = render_professional_summary({"decision": "needs_confirmation"})
    assert "Necesito 2 fotos" in text
