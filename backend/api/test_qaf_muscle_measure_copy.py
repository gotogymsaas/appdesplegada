from django.test import SimpleTestCase

from api.qaf_muscle_measure.engine import render_professional_summary


class QAFMuscleMeasureCopyTests(SimpleTestCase):
    def test_summary_is_human_and_starts_with_name(self):
        result = {
            "decision": "accepted",
            "user_display_name": "Juan Manuel",
            "confidence": {"score": 0.82, "n_views": 2, "views": {}},
            "variables": {
                "symmetry": 78,
                "v_taper": 64,
                "static_posture": 71,
                "definition": 60,
                "measurement_consistency": 74,
                "volume_by_group": {"arms": 55, "back": 62, "glutes": 70, "thigh": 58, "shoulders": 64},
            },
            "progress": {"vs_last_week": {"available": True, "deltas": {"v_taper": {"prev": 60, "now": 64, "delta": 4}}}},
            "insights": ["OK"],
        }

        text = render_professional_summary(result)
        self.assertTrue(text.startswith("Hola Juan Manuel,"))
        self.assertIn("Medición del progreso muscular".lower(), text.lower())
        self.assertNotIn("decision:", text.lower())
        self.assertNotIn("confidence:", text.lower())

    def test_summary_handles_needs_confirmation(self):
        result = {"decision": "needs_confirmation", "user_display_name": "Juan Manuel"}
        text = render_professional_summary(result)
        self.assertIn("No pude detectar", text)
