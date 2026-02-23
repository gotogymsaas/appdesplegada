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
