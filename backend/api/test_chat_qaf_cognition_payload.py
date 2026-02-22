from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatQAFCognitionPayloadTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_chat_sends_qaf_cognition_to_n8n(self, mock_post):
        # n8n responde OK
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "ok"})

        resp = self.client.post(
            "/api/chat/",
            {
                "message": "hazme un análisis QAF",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)

        _args, kwargs = mock_post.call_args
        sent = kwargs.get("json") or {}

        # Debe incluir el bloque del motor de cognición
        self.assertIn("qaf_cognition", sent)
        qc = sent.get("qaf_cognition")
        self.assertTrue(isinstance(qc, dict))

        # Debe incluir un resumen corto para modo quantum
        sys_rules = sent.get("system_rules") or {}
        self.assertTrue(isinstance(sys_rules, dict))
        summary = sys_rules.get("qaf_cognition_summary")
        self.assertTrue(isinstance(summary, str))
        self.assertIn("modo:", summary)

        # Debe anexar también al attachment_text (para que el LLM narre la decisión)
        at = sent.get("attachment_text") or ""
        self.assertIn("[QAF / COGNICIÓN]", at)
