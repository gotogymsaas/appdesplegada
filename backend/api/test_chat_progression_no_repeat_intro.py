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
        self.assertIn("optimizar", out1.lower())
        self.assertIn("en menos de 30 segundos", out1.lower())
        self.assertIn("fuerza o cardio", out1.lower())

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

        self.assertIn("qué tan duro", out2.lower())
        self.assertIn("al límite", out2.lower())
        self.assertNotIn("en menos de 30 segundos", out2.lower())
        self.assertNotIn("respóndeme solo esto", out2.lower())

        # Paso 3: set RPE 10
        r3 = self.client.post(
            "/api/chat/",
            {
                "message": "RPE 10",
                "sessionId": "s1",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "progression_action": {"session": {"rpe_1_10": 10}},
            },
            format="json",
        )
        self.assertEqual(r3.status_code, 200)
        out3 = (r3.json() or {}).get("output") or ""
        self.assertIn("cuánto del plan", out3.lower())

        # Paso 4: completion 100% => accepted => debe traer cierre con valor
        r4 = self.client.post(
            "/api/chat/",
            {
                "message": "Cumplí 100%",
                "sessionId": "s1",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "progression_action": {"session": {"completion_pct": 1.0}},
            },
            format="json",
        )
        self.assertEqual(r4.status_code, 200)
        out4 = (r4.json() or {}).get("output") or ""
        self.assertIn("próximo paso", out4.lower())
        self.assertIn("ajuste recomendado", out4.lower())
