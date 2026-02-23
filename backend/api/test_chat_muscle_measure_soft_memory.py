from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient


User = get_user_model()


def _pose_front(shoulder_width=0.25, hip_width=0.20):
    # Keypoints con coords normalizadas 0..1 y score alto
    # shoulders centered at y=0.25, hips y=0.55
    cx = 0.5
    lsx = cx - shoulder_width / 2
    rsx = cx + shoulder_width / 2
    lhx = cx - hip_width / 2
    rhx = cx + hip_width / 2
    return {
        "keypoints": [
            {"name": "nose", "x": cx, "y": 0.12, "score": 0.95},
            {"name": "left_shoulder", "x": lsx, "y": 0.25, "score": 0.95},
            {"name": "right_shoulder", "x": rsx, "y": 0.25, "score": 0.95},
            {"name": "left_hip", "x": lhx, "y": 0.55, "score": 0.95},
            {"name": "right_hip", "x": rhx, "y": 0.55, "score": 0.95},
        ],
        "image": {"width": 1000, "height": 1800},
    }


class ChatMuscleMeasureSoftMemoryTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
            height=175,
        )

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_soft_memory_persists_and_is_used_as_baseline_for_same_focus(self):
        # 1) Primera medición con foco bíceps (baseline no existe)
        r1 = self.client.post(
            "/api/chat/",
            {
                "message": "Medición del progreso muscular",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "muscle_measure_request": {
                    "focus": "biceps",
                    "poses": {"front_relaxed": _pose_front(shoulder_width=0.24, hip_width=0.20)},
                },
            },
            format="json",
        )
        self.assertEqual(r1.status_code, 200)

        self.user.refresh_from_db()
        cs = getattr(self.user, "coach_state", {}) or {}
        mem = cs.get("muscle_measure_memory") if isinstance(cs.get("muscle_measure_memory"), dict) else {}
        self.assertIn("last_by_focus", mem)
        self.assertIn("biceps", (mem.get("last_by_focus") or {}))

        # 2) Segunda medición mismo foco, diferente ratio -> debe marcar baseline_source=last_same_focus
        r2 = self.client.post(
            "/api/chat/",
            {
                "message": "Medición del progreso muscular",
                "sessionId": "",
                "attachment": "",
                "attachment_text": "",
                "username": "juan",
                "muscle_measure_request": {
                    "focus": "biceps",
                    "poses": {"front_relaxed": _pose_front(shoulder_width=0.28, hip_width=0.20)},
                },
            },
            format="json",
        )
        self.assertEqual(r2.status_code, 200)
        data2 = r2.json() if hasattr(r2, "json") else {}
        qaf = (data2 or {}).get("qaf_muscle_measure") or {}
        prog = qaf.get("progress") if isinstance(qaf.get("progress"), dict) else {}
        self.assertEqual(prog.get("baseline_source"), "last_same_focus")
