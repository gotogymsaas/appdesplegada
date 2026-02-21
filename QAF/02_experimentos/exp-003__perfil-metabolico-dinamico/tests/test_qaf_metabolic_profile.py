import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
import sys


sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_metabolic_profile import evaluate_weekly_metabolic_profile, robust_center


class TestQAFMetabolicProfile(unittest.TestCase):
    def test_robust_center_removes_outlier(self):
        center, meta = robust_center([80.0, 80.1, 80.2, 95.0])
        self.assertIsNotNone(center)
        self.assertGreaterEqual(meta.get("outliers_removed"), 1)
        self.assertLess(float(center), 82.0)

    def test_missing_weights_needs_confirmation(self):
        profile = {"sex": "male", "age": 30, "height_cm": 175, "weight_kg": 80, "goal_type": "deficit", "activity_level": "moderate"}
        weights = {"current_week": [], "previous_week": []}
        r = evaluate_weekly_metabolic_profile(profile, weights)
        self.assertEqual(r.payload.get("decision"), "needs_confirmation")

    def test_deficit_adjustment_is_capped(self):
        profile = {"sex": "female", "age": 29, "height_cm": 165, "weight_kg": 70, "goal_type": "deficit", "activity_level": "low"}
        # Caso extremo: subió mucho vs objetivo deficit
        weights = {"current_week": [71.0, 71.2, 71.1, 71.0], "previous_week": [70.0, 70.1, 70.0, 70.2]}
        r = evaluate_weekly_metabolic_profile(profile, weights, max_weekly_kcal_adjust=200.0)
        adj = float(((r.payload.get("recommendation") or {}).get("weekly_adjustment_kcal_day") or 0.0))
        self.assertLessEqual(abs(adj), 200.0)

    def test_output_has_core_fields(self):
        profile = {"sex": "male", "age": 30, "height_cm": 175, "weight_kg": 80, "goal_type": "maintenance", "activity_level": "moderate"}
        weights = {"current_week": [80.0, 79.9, 80.1, 80.0], "previous_week": [80.1, 80.2, 80.0, 80.1]}
        r = evaluate_weekly_metabolic_profile(profile, weights)
        self.assertIn("metabolic", r.payload)
        self.assertIn("recommendation", r.payload)
        self.assertIsNotNone((r.payload.get("metabolic") or {}).get("tmb_kcal_day"))


if __name__ == "__main__":
    unittest.main()
