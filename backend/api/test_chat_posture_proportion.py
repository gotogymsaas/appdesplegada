from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class ChatPostureProportionTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_posture_proportion_ambiguous_message_returns_confirmation(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Arquitectura Corporal",
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
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "message" for x in qas))
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "pp_cancel" for x in qas))
        self.assertIn("último resultado", (data.get("output") or "").lower())

        # Flujo inicia sin llamar a n8n
        self.assertFalse(mock_post.called)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_posture_proportion_show_last_returns_saved_result(self, mock_post):
        self.user.coach_weekly_state = {
            "posture_proportion": {
                "2026-W10": {
                    "result": {
                        "decision": "accepted",
                        "confidence": {"score": 0.91},
                        "variables": {
                            "postural_efficiency_score": 82,
                            "posture_score": 79,
                            "proportion_score": 76,
                            "alignment_silhouette_index": 78,
                        },
                        "patterns": ["forward_head"],
                        "insights": ["Buen potencial de mejora esta semana."],
                    },
                    "updated_at": "2026-03-01T10:00:00Z",
                }
            }
        }
        self.user.save(update_fields=["coach_weekly_state"])

        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Muéstrame mi último resultado de arquitectura corporal",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}

        self.assertIn("última evaluación", (data.get("output") or "").lower())
        self.assertIn("qaf_posture_proportion", data)
        qas = data.get("quick_actions") or []
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "pp_cancel" for x in qas))
        self.assertFalse(any(isinstance(x, dict) and x.get("type") == "pp_capture" for x in qas))
        self.assertFalse(mock_post.called)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_posture_proportion_start_new_starts_capture_flow(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Iniciar evaluación nueva de arquitectura corporal",
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
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "pp_capture" for x in qas))
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "pp_cancel" for x in qas))
        self.assertFalse(mock_post.called)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_posture_proportion_show_last_without_history_offers_start(self, mock_post):
        resp = self.client.post(
            "/api/chat/",
            {
                "message": "Ver último resultado de arquitectura corporal",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
            },
            format="json",
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json() if hasattr(resp, "json") else {}

        self.assertIn("no encuentro una evaluación previa", (data.get("output") or "").lower())
        qas = data.get("quick_actions") or []
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "message" for x in qas))
        self.assertTrue(any(isinstance(x, dict) and x.get("type") == "pp_capture" for x in qas))
        self.assertFalse(mock_post.called)
