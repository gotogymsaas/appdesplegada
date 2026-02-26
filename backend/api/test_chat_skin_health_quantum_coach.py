from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatSkinHealthQuantumCoachTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.qaf_skin_health.engine.render_professional_summary")
    @patch("api.qaf_skin_health.engine.evaluate_skin_health")
    @patch("api.views.requests.get")
    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_skin_analysis_response_comes_from_quantum_coach(
        self,
        mock_post,
        mock_get,
        mock_eval,
        mock_render,
    ):
        mock_get.return_value = Mock(
            status_code=200,
            content=b"fake-image-bytes",
            headers={"Content-Type": "image/jpeg"},
        )

        mock_eval.return_value = SimpleNamespace(
            payload={
                "decision": "accepted",
                "skin_health_score": 38,
                "confidence": {"score": 0.91, "uncertainty_score": 0.09},
                "recommendation_plan": {
                    "priorities": ["hidratación", "sueño"],
                    "actions": ["+500ml de agua hoy"],
                },
            }
        )

        mock_render.return_value = "RESUMEN TÉCNICO SKIN"
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "Respuesta completa del Quantum Coach"})

        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Vitalidad de la Piel",
                "sessionId": "skin-coach-1",
                "attachment": "https://test.blob.core.windows.net/attachments/foto.jpg?sig=1&se=1&sp=r",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}

        self.assertEqual(data.get("output"), "Respuesta completa del Quantum Coach")
        self.assertNotEqual(data.get("output"), "RESUMEN TÉCNICO SKIN")
        self.assertIn("qaf_skin_health", data)

        qas = data.get("quick_actions") or []
        labels = [str(x.get("label") or "") for x in qas if isinstance(x, dict)]
        self.assertTrue(any("finalizar" in l.lower() for l in labels))
        self.assertTrue(mock_post.called)
