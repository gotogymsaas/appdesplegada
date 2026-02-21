import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
import sys


sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_goal_coherence import evaluate_meal, infer_goal


class TestQAFGoalCoherence(unittest.TestCase):
    def test_infer_goal_explicit(self):
        g = infer_goal({"goal_type": "deficit"})
        self.assertEqual(g.goal_type, "deficit")
        self.assertEqual(g.source, "explicit")

    def test_infer_goal_text_deficit(self):
        g = infer_goal({"goal_text": "quiero bajar grasa y definir"})
        self.assertEqual(g.goal_type, "deficit")
        self.assertGreaterEqual(g.confidence, 0.7)

    def test_evaluate_meal_missing_weight_triggers_info_alert(self):
        r = evaluate_meal({"goal_type": "maintenance"}, {"total_calories": 500, "meal_slot": "lunch"})
        codes = {a.get("code") for a in (r.get("alerts") or [])}
        self.assertIn("missing_weight", codes)

    def test_high_uncertainty_forces_needs_confirmation(self):
        r = evaluate_meal(
            {"weight_kg": 70, "goal_type": "deficit"},
            {"total_calories": 700, "uncertainty_score": 0.8, "meal_slot": "dinner"},
        )
        self.assertEqual(r.get("decision"), "needs_confirmation")

    def test_deficit_over_meal_generates_over_target_alert(self):
        r = evaluate_meal(
            {"weight_kg": 80, "goal_type": "deficit", "activity_level": "moderate"},
            {"total_calories": 1200, "uncertainty_score": 0.1, "meal_slot": "lunch"},
        )
        codes = {a.get("code") for a in (r.get("alerts") or [])}
        self.assertIn("over_target_meal", codes)


if __name__ == "__main__":
    unittest.main()
