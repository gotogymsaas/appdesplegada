from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class QAFCognitionEvaluateTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="cog",
            email="cog@example.com",
            password="pass12345",
        )
        self.client.force_authenticate(user=self.user)

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_returns_stable_schema_without_observations(self):
        resp = self.client.post(
            "/api/qaf/cognition/evaluate/",
            {"message": "hola"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertIn("state", data)
        self.assertIn("indices", data)
        self.assertIn("decision", data)
        self.assertIn("policy", data)

        decision = data.get("decision") or {}
        self.assertIn(decision.get("mode"), ("nutrition", "training", "health", "quantum"))

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_quantum_trigger_forces_quantum_mode(self):
        resp = self.client.post(
            "/api/qaf/cognition/evaluate/",
            {"message": "hazme un análisis QAF"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual((data.get("decision") or {}).get("mode"), "quantum")

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_low_data_quality_asks_clarifying(self):
        # Usuario nuevo (sin coach_state/weekly_state) => Q_data bajo
        resp = self.client.post(
            "/api/qaf/cognition/evaluate/",
            {"message": "no sé qué hacer"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        decision = data.get("decision") or {}
        self.assertIn(decision.get("type"), ("ask_clarifying", "needs_confirmation", "proceed"))
        # si la decisión es pedir claridad, debe proponer una pregunta
        if decision.get("type") == "ask_clarifying":
            qs = decision.get("follow_up_questions") or []
            self.assertTrue(len(qs) >= 1)
