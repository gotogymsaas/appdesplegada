import os
from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatVisionRouterTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post")
    @patch("api.views._describe_image_with_azure_openai")
    @patch("api.views.requests.get")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_training_route_does_not_trigger_calories(self, mock_get, mock_vision, mock_post):
        attachment_url = (
            "https://example.blob.core.windows.net/attachments/"
            "juan/persona.png?sv=2024-01-01&sig=fake"
        )

        mock_get.return_value = Mock(
            status_code=200,
            content=b"\x89PNG\r\n\x1a\nFAKE",
            headers={"Content-Type": "image/png"},
        )

        # Aunque el modelo devolviera 'items' por error, la ruta training debe bloquear QAF calorías.
        mock_vision.return_value = (
            '{"route":"training","route_confidence":0.9,"is_food":false,"items":["manzana"],"portion_estimate":"","notes":"persona entrenando","has_person":true,"has_nutrition_label":false,"is_closeup_skin_or_muscle":false}',
            "",
        )
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "ok"})

        # Para evitar dependencias externas en tests, deshabilitamos vision "real" vía env.
        with patch.dict(os.environ, {"CHAT_ATTACHMENT_VISION": "true"}):
            resp = self.client.post(
                "/api/chat/",
                {
                    "message": "",
                    "sessionId": "",
                    "attachment": attachment_url,
                    "attachment_text": "",
                    "username": "juan",
                },
                format="json",
            )

        self.assertEqual(resp.status_code, 200)

        _post_args, post_kwargs = mock_post.call_args
        sent = post_kwargs.get("json") or {}
        at = sent.get("attachment_text") or ""
        self.assertIn("[DESCRIPCIÓN DE IMAGEN]", at)
        self.assertIn("route: training", at)
        self.assertIn("[ENTRENAMIENTO / IMAGEN]", at)
        self.assertNotIn("[CALORÍAS ESTIMADAS]", at)
