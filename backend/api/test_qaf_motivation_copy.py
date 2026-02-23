from django.test import SimpleTestCase

from api.qaf_motivation.engine import render_professional_summary


class QAFMotivationCopyTests(SimpleTestCase):
    def test_needs_confirmation_starts_with_name_and_explains_pressure(self):
        result = {
            "decision": "needs_confirmation",
            "user_display_name": "Juan Manuel",
            "profile": {"top": "disciplina"},
            "state": {"mood": "neutral"},
            "challenge": {"label": "Reto: hoy solo 6–10 min. No es intensidad, es constancia.", "type": "consistencia", "minutes": 8},
            "reward": {"label": "Cadena de constancia: 3 días", "note": "La racha protege tu identidad"},
        }

        text = render_professional_summary(result)

        self.assertTrue(text.startswith("Hola Juan Manuel,"))
        self.assertIn("¿Cómo quieres que te empuje hoy?".lower(), text.lower())
        self.assertIn("Suave".lower(), text.lower())
        self.assertIn("Medio".lower(), text.lower())
        self.assertIn("Firme".lower(), text.lower())

        # Debe evitar el formato técnico viejo
        self.assertNotIn("perfil dominante:", text.lower())

    def test_accepted_includes_min_plan_and_reward(self):
        result = {
            "decision": "accepted",
            "user_display_name": "Juan Manuel",
            "profile": {"top": "disciplina"},
            "state": {"mood": "neutral"},
            "challenge": {"label": "Reto: hoy solo 6–10 min. No es intensidad, es constancia.", "type": "minimo", "minutes": 10},
            "reward": {"label": "Cadena de constancia: 4 días", "note": "La racha protege tu identidad"},
        }

        text = render_professional_summary(result)

        self.assertTrue(text.startswith("Hola Juan Manuel,"))
        self.assertIn("plan mínimo".lower(), text.lower())
        self.assertIn("Guía rápida".lower(), text.lower())
        self.assertIn("Recompensa".lower(), text.lower())
        self.assertIn("Cadena de constancia".lower(), text.lower())
