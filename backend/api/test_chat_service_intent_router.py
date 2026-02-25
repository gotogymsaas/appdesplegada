from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatServiceIntentRouterTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="qauser",
            email="qauser@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_skin_message_returns_intent_confirmation(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Vitalidad de la Piel",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "qauser",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}
        out = str(data.get("output") or "").lower()
        self.assertIn("último resultado", out)

        qas = data.get("quick_actions") or []
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "message" for x in qas))
        self.assertFalse(mock_post.called)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_show_last_skin_returns_saved_result(self, mock_post):
        self.user.coach_state = {
            "skin_last_result": {
                "result": {
                    "decision": "accepted",
                    "skin_health_score": 81,
                    "sub_scores": {"texture": 80, "tone": 82},
                },
                "updated_at": "2026-02-25T10:00:00Z",
            }
        }
        self.user.save(update_fields=["coach_state"])

        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Ver último resultado",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "qauser",
                "service_intent": {
                    "experience": "exp-011_skin_health",
                    "action": "show_last",
                },
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}
        out = str(data.get("output") or "")
        self.assertIn("Tu último resultado", out)
        self.assertIn("skin_health_score", out)
        self.assertFalse(mock_post.called)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_start_new_muscle_returns_capture_actions(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Iniciar Progreso Muscular",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "qauser",
                "service_intent": {
                    "experience": "exp-010_muscle_measure",
                    "action": "start_new",
                },
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}
        qas = data.get("quick_actions") or []
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "muscle_capture" for x in qas))
        self.assertFalse(mock_post.called)
