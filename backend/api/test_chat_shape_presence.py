from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatShapePresenceTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_shape_presence_starts_flow_with_quick_actions(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Shape & Presence",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}

        self.assertIn("quick_actions", data)
        qas = data.get("quick_actions") or []
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "shape_capture" for x in qas))
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "shape_cancel" for x in qas))

        # Flujo inicia sin llamar a n8n
        self.assertFalse(mock_post.called)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_shape_presence_start_new_service_intent_keeps_shape_flow(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Iniciar evaluación nueva",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "service_intent": {
                    "experience": "exp-012_shape_presence",
                    "action": "start_new",
                },
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}
        qas = data.get("quick_actions") or []

        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "shape_capture" for x in qas))
        self.assertFalse(any(isinstance(x, dict) and x.get("type") == "pp_capture" for x in qas))
        self.assertFalse(mock_post.called)
