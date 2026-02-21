import unittest
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_lifestyle_intelligence import evaluate_lifestyle


class TestQAFLifestyleIntelligence(unittest.TestCase):
    def test_missing_sleep_needs_confirmation(self):
        r = evaluate_lifestyle({"daily_metrics": [{"date": "2026-02-21", "steps": 4000}]}).payload
        self.assertEqual(r.get("decision"), "needs_confirmation")
        qs = r.get("follow_up_questions") or []
        self.assertTrue(qs)

    def test_microhabits_max_3(self):
        payload = {
            "daily_metrics": [
                {"date": "2026-02-20", "steps": 2000, "sleep_minutes": 300, "calories": 900},
                {"date": "2026-02-21", "steps": 1800, "sleep_minutes": 280, "calories": 850},
            ],
            "memory": {"last_ids": ["breath_5", "mobility_3"]},
        }
        r = evaluate_lifestyle(payload).payload
        self.assertLessEqual(len(r.get("microhabits") or []), 3)

    def test_dhss_in_range(self):
        payload = {
            "daily_metrics": [
                {"date": "2026-02-21", "steps": 9000, "sleep_minutes": 450, "calories": 2100, "resting_heart_rate_bpm": 62},
            ],
        }
        r = evaluate_lifestyle(payload).payload
        s = (r.get("dhss") or {}).get("score")
        self.assertIsInstance(s, int)
        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 100)


if __name__ == "__main__":
    unittest.main()
