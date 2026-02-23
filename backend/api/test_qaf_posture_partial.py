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
        self.assertIn("análisis parcial", (text or "").lower())

    def test_posture_partial_includes_cm_approx_from_height(self):
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
            "user_context": {"height_cm": 175, "injury_recent": False, "pain_neck": False, "pain_low_back": False, "level": "beginner"},
            "locale": "es-CO",
        }

        res = evaluate_posture(payload).payload
        text = render_professional_summary(res) or ""
        self.assertTrue(("cm aprox" in text.lower()) or ("cm estimad" in text.lower()))
        self.assertTrue(("hombros" in text.lower()) or ("cadera" in text.lower()) or ("brazo" in text.lower()))

    def test_posture_height_calibration_adds_cm_estimates_when_full_body(self):
        payload = {
            "poses": {
                "front": {
                    "keypoints": [
                        {"name": "left_shoulder", "x": 0.35, "y": 0.30, "score": 0.95},
                        {"name": "right_shoulder", "x": 0.65, "y": 0.30, "score": 0.95},
                        {"name": "left_hip", "x": 0.45, "y": 0.60, "score": 0.95},
                        {"name": "right_hip", "x": 0.55, "y": 0.60, "score": 0.95},
                        {"name": "left_knee", "x": 0.46, "y": 0.80, "score": 0.95},
                        {"name": "right_knee", "x": 0.54, "y": 0.80, "score": 0.95},
                        {"name": "left_ankle", "x": 0.46, "y": 0.96, "score": 0.95},
                        {"name": "right_ankle", "x": 0.54, "y": 0.96, "score": 0.95},
                        {"name": "nose", "x": 0.50, "y": 0.12, "score": 0.95},
                        {"name": "left_wrist", "x": 0.30, "y": 0.52, "score": 0.90},
                        {"name": "right_wrist", "x": 0.70, "y": 0.52, "score": 0.90},
                    ],
                    "image": {"width": 1000, "height": 2000},
                },
                "side": None,
            },
            "user_context": {"height_cm": 175, "injury_recent": False, "pain_neck": False, "pain_low_back": False, "level": "beginner"},
            "locale": "es-CO",
        }

        res = evaluate_posture(payload).payload
        sigs = res.get('signals') if isinstance(res.get('signals'), list) else []
        names = {str(s.get('name')) for s in sigs if isinstance(s, dict) and s.get('name')}
        self.assertIn('shoulder_width_cm', names)
