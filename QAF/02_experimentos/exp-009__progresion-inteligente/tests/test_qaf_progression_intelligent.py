import unittest
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_progression_intelligent import evaluate_progression, _parse_strength_line


class TestQAFProgressionIntelligent(unittest.TestCase):
    def test_missing_requires_confirmation(self):
        r = evaluate_progression({"session": {}, "signals": {}}).payload
        self.assertEqual(r.get('decision'), 'needs_confirmation')

    def test_plateau_strength_detects(self):
        payload = {
            "session": {"rpe_1_10": 8, "completion_pct": 1.0},
            "strength": {"name": "sentadilla", "sets": 3, "reps": 5, "load_kg": 80},
            "history": {
                "strength:sentadilla": [
                    {"est_1rm": 100.0, "rpe": 7.0, "tonnage": 1200},
                    {"est_1rm": 100.2, "rpe": 7.8, "tonnage": 1200},
                    {"est_1rm": 100.1, "rpe": 8.4, "tonnage": 1200},
                ]
            },
            "signals": {"sleep_minutes": 420, "steps": 6000},
        }
        r = evaluate_progression(payload).payload
        self.assertTrue((r.get('plateau') or {}).get('detected'))

    def test_parse_strength_line(self):
        ex = _parse_strength_line('press banca 4x8x60kg')
        self.assertEqual(ex.get('sets'), 4)
        self.assertEqual(ex.get('reps'), 8)
        self.assertEqual(ex.get('load_kg'), 60.0)


if __name__ == '__main__':
    unittest.main()
