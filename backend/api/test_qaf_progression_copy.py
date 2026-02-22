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
        self.assertIn("avance con control", text.lower())
        self.assertIn("55%", text)
        self.assertIn("Micro‑objetivo de hoy", text)
        self.assertNotIn("minimum_viable", text)
        self.assertNotIn("rpe_1_10", text)
        self.assertNotIn("completion_pct", text)
        self.assertNotIn("modality", text)
        self.assertNotIn("Readiness:", text)
        self.assertNotIn("Acción:", text)
        self.assertNotIn("Faltan:", text)

        # Debe explicar lo que pide
        self.assertIn("¿hoy fue Fuerza o Cardio?".lower(), text.lower())
        # Secuencial: en este paso (falta modalidad) no debería pedir aún RPE ni %
        self.assertNotIn("RPE", text)
        self.assertNotIn("100%", text)

    def test_summary_bucket_0_20_is_recovery_intelligent(self):
        result = {
            "decision": "needs_confirmation",
            "readiness": {"score": 20},
            "decision_engine": {"action": "minimum_viable", "reason": "default_safe"},
            "micro_goal": "Hoy: cumplir el mínimo viable y proteger la constancia.",
            "confidence": {"score": 0.5, "missing": ["modality"]},
        }

        text = render_professional_summary(result)
        self.assertIn("recuperación inteligente", text.lower())
        self.assertIn("20%", text)
        self.assertIn("¿hoy fue fuerza o cardio?".lower(), text.lower())
