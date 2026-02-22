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
        self.assertIn("optimizar", text.lower())
        self.assertIn("evolución de entrenamiento", text.lower())
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
        self.assertNotIn("al límite", text.lower())
        self.assertNotIn("100%", text)

    def test_summary_does_not_repeat_long_intro_when_show_intro_false(self):
        result = {
            "decision": "needs_confirmation",
            "readiness": {"score": 55},
            "plateau": {"detected": False, "reason": ""},
            "decision_engine": {"action": "minimum_viable", "reason": "default_safe"},
            "confidence": {"score": 0.5, "missing": ["modality"]},
            "ui": {"show_intro": False},
        }

        text = render_professional_summary(result)

        # Con show_intro False, debe ser corto (solo pregunta), sin el bloque largo.
        self.assertIn("¿hoy fue fuerza o cardio?".lower(), text.lower())
        self.assertNotIn("en menos de 30 segundos", text.lower())
