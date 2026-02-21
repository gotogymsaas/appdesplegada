import unittest
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_posture_corrective import evaluate_posture


class TestQAFPostureCorrective(unittest.TestCase):
    def test_low_quality_needs_confirmation(self):
        payload = {
            "poses": {
                "front": {"keypoints": [{"name": "nose", "x": 0.5, "y": 0.1, "score": 0.2}]},
                "side": {"keypoints": [{"name": "right_ear", "x": 0.6, "y": 0.2, "score": 0.2}]},
            },
            "user_context": {"pain_neck": False},
        }
        r = evaluate_posture(payload).payload
        self.assertEqual(r.get("decision"), "needs_confirmation")

    def test_forward_head_label_when_large_offset(self):
        payload = {
            "poses": {
                "front": {
                    "keypoints": [
                        {"name": "nose", "x": 0.5, "y": 0.12, "score": 0.95},
                        {"name": "left_shoulder", "x": 0.43, "y": 0.26, "score": 0.92},
                        {"name": "right_shoulder", "x": 0.57, "y": 0.27, "score": 0.91},
                        {"name": "left_hip", "x": 0.46, "y": 0.52, "score": 0.9},
                        {"name": "right_hip", "x": 0.54, "y": 0.52, "score": 0.9},
                        {"name": "left_knee", "x": 0.47, "y": 0.74, "score": 0.88},
                        {"name": "right_knee", "x": 0.53, "y": 0.74, "score": 0.88},
                        {"name": "left_ankle", "x": 0.47, "y": 0.94, "score": 0.9},
                        {"name": "right_ankle", "x": 0.53, "y": 0.94, "score": 0.9},
                    ]
                },
                "side": {
                    "keypoints": [
                        {"name": "right_ear", "x": 0.66, "y": 0.14, "score": 0.92},
                        {"name": "right_shoulder", "x": 0.56, "y": 0.28, "score": 0.9},
                        {"name": "right_hip", "x": 0.54, "y": 0.52, "score": 0.9},
                        {"name": "right_knee", "x": 0.55, "y": 0.74, "score": 0.87},
                        {"name": "right_ankle", "x": 0.55, "y": 0.94, "score": 0.9},
                    ]
                },
            },
            "user_context": {"pain_neck": False, "injury_recent": False},
        }
        r = evaluate_posture(payload).payload
        self.assertEqual(r.get("decision"), "accepted")
        keys = {x.get("key") for x in (r.get("labels") or [])}
        self.assertIn("forward_head", keys)


if __name__ == "__main__":
    unittest.main()
