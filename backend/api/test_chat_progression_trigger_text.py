from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatProgressionTriggerTextTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_text_evolucion_entrenamiento_triggers_exp009(self):
        # Sin payload progression_request: debe activar Exp-009 por texto y responder directo (sin depender de n8n).
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Evolución de Entrenamiento",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("output", data)
        self.assertIn("qaf_progression", data)

        out = data.get("output") or ""
        self.assertIn("Tu estado para entrenar hoy", out)
        # Debe pedir inputs en lenguaje humano
        self.assertIn("¿Hoy fue Fuerza o Cardio?", out)
