from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatProgressionNoRepeatIntroTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
            full_name="Juan Manuel",
        )

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_second_step_does_not_repeat_long_intro(self, mock_post):
        # Hacemos que n8n no interfiera: el flujo Exp-009 debe responder directo.
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "ok"})

        # Paso 1: iniciar por texto del botón
        r1 = self.client.post(
            "/api/chat/",
            {"message": "Evolución de Entrenamiento", "sessionId": "s1", "attachment": "", "attachment_text": "", "username": "juan"},
            format="json",
        )
        self.assertEqual(r1.status_code, 200)
        out1 = (r1.json() or {}).get("output") or ""
        self.assertIn("modo", out1.lower())
        self.assertIn("%", out1)

        # Paso 2: click 'Cardio' (modalidad). Debe preguntar RPE sin repetir el bloque largo.
        r2 = self.client.post(
            "/api/chat/",
            {
                "message": "Cardio",
                "sessionId": "s1",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "progression_action": {"cardio": {"minutes": 20}},
            },
            format="json",
        )
        self.assertEqual(r2.status_code, 200)
        out2 = (r2.json() or {}).get("output") or ""

        self.assertIn("rpe", out2.lower())
        self.assertNotIn("%", out2)  # no repetir porcentaje
        self.assertNotIn("¿por qué", out2.lower())
        self.assertNotIn("tu nivel de preparación", out2.lower())
