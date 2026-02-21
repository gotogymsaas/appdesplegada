import unittest
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_body_trend import evaluate_body_trend


class TestQAFBodyTrend(unittest.TestCase):
    def test_6_weeks_trajectory(self):
        prof = {"tdee_kcal_day": 2500, "recommended_kcal_day": 2100}
        obs = {"weight_current_week_avg_kg": 80.0, "weight_previous_week_avg_kg": 80.5, "kcal_in_avg_day": 2300}
        r = evaluate_body_trend(prof, obs, horizon_weeks=6)
        self.assertEqual(r.payload.get('decision'), 'accepted')
        baseline = (r.payload.get('scenarios') or {}).get('baseline')
        self.assertIsInstance(baseline, dict)
        traj = baseline.get('trajectory')
        self.assertEqual(len(traj), 6)

    def test_missing_intake_needs_confirmation(self):
        prof = {"tdee_kcal_day": 2300, "recommended_kcal_day": 1900}
        obs = {"weight_current_week_avg_kg": 70.0, "weight_previous_week_avg_kg": 70.2}
        r = evaluate_body_trend(prof, obs, horizon_weeks=6)
        self.assertEqual(r.payload.get('decision'), 'needs_confirmation')
        self.assertTrue((r.payload.get('follow_up_questions') or []))


if __name__ == '__main__':
    unittest.main()
