from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class _DummyN8nResp:
    status_code = 200

    def json(self):
        # Simular salida de n8n para asegurar que el backend la reemplaza cuando lifestyle está activo.
        return {"output": "RESPUESTA_N8N"}


class ChatLifestyleActivationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_lifestyle_activates_on_natural_self_report_training_today(self, _mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Dormí 5 horas, caminé poco y estoy estresado. ¿Cómo debería entrenar hoy para avanzar sin quemarme?",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}
        out = (data or {}).get("output") or ""
        self.assertIn("Estado de hoy", out)
        self.assertIn("DHSS:", out)
        self.assertNotIn("RESPUESTA_N8N", out)

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_lifestyle_heuristic_activation_is_rate_limited_per_day(self, _mock_post):
        msg = "Dormí 5 horas, caminé poco y estoy estresado. ¿Cómo debería entrenar hoy para avanzar sin quemarme?"

        # 1ra vez: activa
        r1 = self.client.post(
            "/api/chat/",
            {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
            format="json",
        )
        self.assertEqual(r1.status_code, 200)
        out1 = (r1.json() or {}).get("output") or ""
        self.assertIn("Estado de hoy", out1)

        # 2da vez: activa
        r2 = self.client.post(
            "/api/chat/",
            {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
            format="json",
        )
        self.assertEqual(r2.status_code, 200)
        out2 = (r2.json() or {}).get("output") or ""
        self.assertIn("Estado de hoy", out2)

        # 3ra vez: NO debe activar (rate limit). En este caso, n8n responde "RESPUESTA_N8N".
        r3 = self.client.post(
            "/api/chat/",
            {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
            format="json",
        )
        self.assertEqual(r3.status_code, 200)
        out3 = (r3.json() or {}).get("output") or ""
        self.assertNotIn("Estado de hoy", out3)
        self.assertIn("RESPUESTA_N8N", out3)

        # Activación explícita debe seguir funcionando aunque el límite se haya alcanzado
        r4 = self.client.post(
            "/api/chat/",
            {
                "message": "Estado de hoy: ¿cómo está mi energía?",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )
        self.assertEqual(r4.status_code, 200)
        out4 = (r4.json() or {}).get("output") or ""
        self.assertIn("Estado de hoy", out4)

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_lifestyle_mark_habit_done_does_not_repeat_full_summary(self, _mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "✅ micro",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "lifestyle_habit_done": {"id": "mobility_3"},
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        out = (resp.json() or {}).get("output") or ""
        self.assertIn("Registré", out)
        self.assertNotIn("DHSS:", out)
