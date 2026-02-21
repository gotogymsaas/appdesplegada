import unittest
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_motivation_psych import evaluate_motivation


class TestQAFMotivationPsych(unittest.TestCase):
    def test_needs_confirmation_without_pressure(self):
        r = evaluate_motivation({"message": "Necesito motivación"}).payload
        self.assertEqual(r.get("decision"), "needs_confirmation")
        fu = r.get("follow_up_questions") or []
        self.assertTrue(fu)

    def test_fatigue_proposes_recovery(self):
        r = evaluate_motivation({
            "message": "Estoy cansado y estresado",
            "preferences": {"pressure": "suave"},
            "lifestyle": {"dhss": {"band": "recovery"}},
        }).payload
        chall = r.get("challenge") or {}
        self.assertIn("movilidad", str(chall.get("label") or "").lower())

    def test_vector_is_normalized(self):
        r = evaluate_motivation({
            "message": "Quiero verme mejor y definir abdomen",
            "preferences": {"pressure": "medio"},
        }).payload
        vec = (r.get("profile") or {}).get("vector") or {}
        s = sum(float(v) for v in vec.values())
        self.assertAlmostEqual(s, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
