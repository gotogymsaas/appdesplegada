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
    def test_sim_buttons_turn_off_on_first_simulation_and_show_finalize(self, mock_post):
        # Evitar dependencia de n8n
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "ok"})

        # 1ra simulación: no debe reinyectar Simular..., solo finalizar.
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
        self.assertFalse(any("simular" in (l or "").lower() for l in labels1))
        self.assertIn("Finalizar", labels1)

        # 2da simulación: se mantiene sin botones de simulación.
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
        self.assertIn("Finalizar", labels2)

    @patch("api.views.requests.post")
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_start_new_trend_reenables_sim_buttons(self, mock_post):
        mock_post.return_value = Mock(status_code=200, json=lambda: {"output": "ok"})

        # Bloquea simulaciones al ejecutar una.
        self.client.post(
            "/api/chat/",
            {
                "message": "Simular recomendación",
                "sessionId": "sbt2",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "body_trend_request": {"scenario": "follow_plan"},
            },
            format="json",
        )

        # Nueva evaluación (sin escenario) vuelve a habilitar los 3 botones + finalizar.
        r = self.client.post(
            "/api/chat/",
            {
                "message": "Tendencia 6 semanas",
                "sessionId": "sbt2",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "body_trend_request": {},
            },
            format="json",
        )
        self.assertEqual(r.status_code, 200)
        qa = (r.json() or {}).get("quick_actions") or []
        labels = [str(x.get("label") or "") for x in qa if isinstance(x, dict)]
        self.assertTrue(any("simular" in (l or "").lower() for l in labels))
        self.assertIn("Finalizar", labels)
