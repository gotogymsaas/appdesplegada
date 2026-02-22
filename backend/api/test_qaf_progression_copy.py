from django.test import SimpleTestCase

from api.qaf_progression.engine import render_professional_summary


class QAFProgressionCopyTests(SimpleTestCase):
    def test_summary_hides_technical_fields_and_is_ordered(self):
        result = {
            "decision": "needs_confirmation",
            "readiness": {"score": 55},
            "plateau": {"detected": False, "reason": ""},
            "decision_engine": {"action": "minimum_viable", "reason": "default_safe"},
            "micro_goal": "Hoy: cumplir el mínimo viable y proteger la constancia.",
            "confidence": {"score": 0.5, "missing": ["rpe_1_10", "completion_pct", "modality"]},
        }

        text = render_professional_summary(result)

        # Debe ser humano y sin claves internas
        self.assertIn("Tu estado para entrenar hoy", text)
        self.assertIn("Micro‑objetivo de hoy", text)
        self.assertNotIn("minimum_viable", text)
        self.assertNotIn("rpe_1_10", text)
        self.assertNotIn("completion_pct", text)
        self.assertNotIn("modality", text)
        self.assertNotIn("Readiness:", text)
        self.assertNotIn("Acción:", text)
        self.assertNotIn("Faltan:", text)

        # Debe explicar lo que pide
        self.assertIn("¿Hoy fue Fuerza o Cardio?", text)
        self.assertIn("RPE", text)
        self.assertIn("100%", text)
