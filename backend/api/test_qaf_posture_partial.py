from django.test import SimpleTestCase

from api.qaf_posture.engine import evaluate_posture, render_professional_summary


class QafPosturePartialTests(SimpleTestCase):
    def test_posture_partial_front_only_returns_partial_notice(self):
        payload = {
            "poses": {
                "front": {
                    "keypoints": [
                        {"name": "left_shoulder", "x": 0.4, "y": 0.3, "score": 0.9},
                        {"name": "right_shoulder", "x": 0.6, "y": 0.31, "score": 0.9},
                        {"name": "left_hip", "x": 0.45, "y": 0.6, "score": 0.9},
                        {"name": "right_hip", "x": 0.55, "y": 0.61, "score": 0.9},
                        {"name": "left_knee", "x": 0.46, "y": 0.8, "score": 0.9},
                        {"name": "right_knee", "x": 0.54, "y": 0.81, "score": 0.9},
                        {"name": "left_ankle", "x": 0.46, "y": 0.95, "score": 0.9},
                        {"name": "right_ankle", "x": 0.54, "y": 0.96, "score": 0.9},
                        {"name": "nose", "x": 0.5, "y": 0.15, "score": 0.9},
                    ]
                },
                "side": None,
            },
            "user_context": {"injury_recent": False, "pain_neck": False, "pain_low_back": False, "level": "beginner"},
            "locale": "es-CO",
        }

        res = evaluate_posture(payload).payload
        self.assertEqual(res.get("decision"), "needs_confirmation")

        text = render_professional_summary(res)
        self.assertIn("resultado parcial", (text or "").lower())
