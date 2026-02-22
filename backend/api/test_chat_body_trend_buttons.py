from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from api.views import _week_id


User = get_user_model()


class ChatBodyTrendButtonsTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

        wk = _week_id()
        # Aseguramos que hay peso semanal y kcal promedio para que aparezcan botones de simulación.
        self.user.coach_weekly_state = {
            "weekly_weights": {wk: {"avg_weight_kg": 80.0}},
            "kcal_avg_by_week": {wk: {"kcal_in_avg_day": 2000.0}},
            "metabolic_last": {"tdee_effective_kcal_day": 2400.0, "kcal_day": 2000.0},
        }
        self.user.save(update_fields=["coach_weekly_state"])

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_sim_buttons_turn_off_on_second_simulation(self, mock_post):
        # Evitar dependencia de n8n
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "ok"})

        # 1ra simulación: debe devolver botones (Simular...)
        r1 = self.client.post(
            "/api/chat/",
            {
                "message": "Simular -200 kcal",
                "sessionId": "sbt1",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "body_trend_request": {"scenario": "minus_200"},
            },
            format="json",
        )
        self.assertEqual(r1.status_code, 200)
        qa1 = (r1.json() or {}).get("quick_actions") or []
        labels1 = [str(x.get("label") or "") for x in qa1 if isinstance(x, dict)]
        self.assertTrue(any("simular" in (l or "").lower() for l in labels1))

        # 2da simulación: botones se apagan
        r2 = self.client.post(
            "/api/chat/",
            {
                "message": "Simular +200 kcal",
                "sessionId": "sbt1",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "body_trend_request": {"scenario": "plus_200"},
            },
            format="json",
        )
        self.assertEqual(r2.status_code, 200)
        qa2 = (r2.json() or {}).get("quick_actions") or []
        labels2 = [str(x.get("label") or "") for x in qa2 if isinstance(x, dict)]
        self.assertFalse(any("simular" in (l or "").lower() for l in labels2))
