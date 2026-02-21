import unittest
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_meal_planner import generate_week_plan


class TestQAFMealPlanner(unittest.TestCase):
    def test_generates_week(self):
        r = generate_week_plan(kcal_day=2000, meals_per_day=3, variety_level="normal", exclude=[], seed=1)
        days = ((r.get("plan") or {}).get("days") or [])
        self.assertEqual(len(days), 7)

    def test_exclude_respected(self):
        r = generate_week_plan(kcal_day=1800, meals_per_day=4, variety_level="high", exclude=["arroz_blanco_cocido"], seed=2)
        plan = (r.get("plan") or {})
        for d in plan.get("days") or []:
            for m in d.get("meals") or []:
                for it in m.get("items") or []:
                    self.assertNotEqual(it.get("item_id"), "arroz_blanco_cocido")


if __name__ == "__main__":
    unittest.main()
