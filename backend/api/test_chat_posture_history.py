from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


class _DummyN8nResp:
    status_code = 200

    def json(self):
        # Output vacío para que el backend preprenda el resumen de postura.
        return {"output": ""}


def _posture_request(front_delta: float, side_delta: float):
    # front_delta: cambia asimetría de hombros
    # side_delta: cambia ear-vs-shoulder (forward head proxy)
    return {
        "poses": {
            "front": {
                "keypoints": [
                    {"name": "left_shoulder", "x": 0.40, "y": 0.30, "score": 0.95},
                    {"name": "right_shoulder", "x": 0.60, "y": 0.30 + float(front_delta), "score": 0.95},
                    {"name": "left_hip", "x": 0.45, "y": 0.60, "score": 0.95},
                    {"name": "right_hip", "x": 0.55, "y": 0.60 + (float(front_delta) / 2.0), "score": 0.95},
                    {"name": "left_knee", "x": 0.46, "y": 0.80, "score": 0.95},
                    {"name": "right_knee", "x": 0.54, "y": 0.81, "score": 0.95},
                    {"name": "left_ankle", "x": 0.46, "y": 0.95, "score": 0.95},
                    {"name": "right_ankle", "x": 0.54, "y": 0.96, "score": 0.95},
                    {"name": "nose", "x": 0.50, "y": 0.15, "score": 0.95},
                ]
            },
            "side": {
                "keypoints": [
                    {"name": "right_ear", "x": 0.52 + float(side_delta), "y": 0.14, "score": 0.92},
                    {"name": "right_shoulder", "x": 0.50, "y": 0.30, "score": 0.92},
                    {"name": "right_hip", "x": 0.50, "y": 0.60, "score": 0.92},
                    {"name": "right_knee", "x": 0.52, "y": 0.80, "score": 0.92},
                    {"name": "right_ankle", "x": 0.53, "y": 0.96, "score": 0.92},
                ]
            },
        },
        "user_context": {"injury_recent": False, "pain_neck": False, "pain_low_back": False, "level": "beginner"},
        "locale": "es-CO",
    }


class ChatPostureHistoryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_posture_persists_history_and_shows_progress_from_second(self, _mock_post):
        r1 = self.client.post(
            "/api/chat/",
            {
                "message": "Postura",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "posture_request": _posture_request(front_delta=0.02, side_delta=0.00),
            },
            format="json",
        )
        self.assertEqual(r1.status_code, 200)
        out1 = (r1.json() or {}).get("output") or ""
        self.assertIn("Corrección de postura", out1)
        self.assertNotIn("Cambios vs tu última medición", out1)

        self.user.refresh_from_db()
        cs = self.user.coach_state or {}
        hist = cs.get("posture_measurements") or []
        self.assertEqual(len(hist), 1)

        r2 = self.client.post(
            "/api/chat/",
            {
                "message": "Postura",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "posture_request": _posture_request(front_delta=0.00, side_delta=-0.01),
            },
            format="json",
        )
        self.assertEqual(r2.status_code, 200)
        out2 = (r2.json() or {}).get("output") or ""
        self.assertIn("Tu evolución vs la última medición", out2)

        self.user.refresh_from_db()
        cs2 = self.user.coach_state or {}
        hist2 = cs2.get("posture_measurements") or []
        self.assertEqual(len(hist2), 2)

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_posture_history_is_capped_to_last_4(self, _mock_post):
        first_shoulder_asym = None
        for i in range(5):
            front_delta = 0.01 * float(i + 1)
            resp = self.client.post(
                "/api/chat/",
                {
                    "message": "Postura",
                    "sessionId": "",
                    "attachment": "",
                    "attachment_text": "",
                    "username": "juan",
                    "posture_request": _posture_request(front_delta=front_delta, side_delta=0.0),
                },
                format="json",
            )
            self.assertEqual(resp.status_code, 200)

            self.user.refresh_from_db()
            cs = self.user.coach_state or {}
            hist = cs.get("posture_measurements") or []
            self.assertLessEqual(len(hist), 4)
            if i == 0:
                try:
                    first_shoulder_asym = float((hist[0].get("metrics") or {}).get("shoulder_asymmetry"))
                except Exception:
                    first_shoulder_asym = None

        self.user.refresh_from_db()
        cs_end = self.user.coach_state or {}
        hist_end = cs_end.get("posture_measurements") or []
        self.assertEqual(len(hist_end), 4)
        if first_shoulder_asym is not None:
            vals = []
            for h in hist_end:
                try:
                    vals.append(float((h.get("metrics") or {}).get("shoulder_asymmetry")))
                except Exception:
                    pass
            self.assertNotIn(first_shoulder_asym, vals)
