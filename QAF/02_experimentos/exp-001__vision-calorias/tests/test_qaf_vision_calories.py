import json
import tempfile
import unittest
from pathlib import Path


# Permite importar desde scripts/ sin empaquetar.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
import sys

sys.path.insert(0, str(SCRIPTS_DIR))

from qaf_vision_calories import build_portion_candidates, load_calorie_db, qaf_infer_from_vision


class TestQAFVisionCalories(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_dir = Path(__file__).resolve().parents[1]
        cls.calorie_db = load_calorie_db(cls.base_dir / "data" / "calorie_db.csv")

    def test_build_portion_candidates_exactly_three(self):
        cands = build_portion_candidates("huevo", calorie_db=self.calorie_db, locale="es-CO")
        self.assertIsInstance(cands, list)
        self.assertEqual(len(cands), 3)
        for c in cands:
            self.assertIn("label", c)
            self.assertIn("grams", c)
            self.assertIn("confidence_hint", c)

    def test_build_portion_candidates_preserves_default_serving_when_exists(self):
        # huevo tiene default_serving_grams=50 en calorie_db.csv
        cands = build_portion_candidates("huevo", calorie_db=self.calorie_db, locale="es-CO")
        grams = [float(c.get("grams") or 0.0) for c in cands]
        self.assertIn(50.0, grams)

    def test_range_driver_prefers_uncertain_item(self):
        vision = {
            "items": ["huevo", "pan"],
            # Porción explícita para huevo (estrecha rango), pan queda con más spread.
            "portion_estimate": "1 unidad de huevo",
        }
        result = qaf_infer_from_vision(vision, self.calorie_db, locale="es-CO")
        self.assertIn(result.get("range_driver"), {"pan_rebanada", "huevo"})
        # Esperado: pan tiene mayor spread si huevo está centrado explícitamente.
        self.assertEqual(result.get("range_driver"), "pan_rebanada")

    def test_confirmed_portions_sets_selected_and_persists_soft_memory(self):
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "soft_memory.json"
            vision = {
                "items": ["huevo"],
                "confirmed_portions": [{"item_id": "huevo", "grams": 100}],
            }
            result = qaf_infer_from_vision(
                vision,
                self.calorie_db,
                user_id="u1",
                locale="es-CO",
                soft_memory_path=mp,
            )

            items = result.get("items") or []
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].get("item_id"), "huevo")
            self.assertEqual(float(items[0].get("grams") or 0.0), 100.0)
            self.assertEqual(items[0].get("selected_portion"), {"grams": 100.0, "source": "user"})

            self.assertTrue(mp.exists())
            saved = json.loads(mp.read_text(encoding="utf-8"))
            self.assertIn("u1", saved)
            self.assertIn("huevo", saved["u1"])
            self.assertAlmostEqual(float(saved["u1"]["huevo"]["grams"]), 100.0)

    def test_ambiguous_items_high_entropy_triggers_confirmation(self):
        # Dos hipótesis con masa similar -> H_norm alta -> colapso humano
        vision = {
            "items": ["arroz", "pollo"],
            "portion_estimate": "",
        }
        result = qaf_infer_from_vision(vision, self.calorie_db, locale="es-CO")
        unc = result.get("uncertainty") or {}
        self.assertGreaterEqual(float(unc.get("entropy_norm") or 0.0), 0.75)
        self.assertTrue(bool(result.get("needs_confirmation")))
        self.assertIn("follow_up_questions", result)
        self.assertTrue(isinstance(result.get("follow_up_questions"), list))

    def test_dominant_hypothesis_low_entropy_auto_selects_portion(self):
        # Una hipótesis dominante (solo 1 item exacto) -> H_norm ~0
        vision = {
            "items": ["arroz"],
            "portion_estimate": "160 g de arroz",
        }
        result = qaf_infer_from_vision(vision, self.calorie_db, locale="es-CO")
        unc = result.get("uncertainty") or {}
        self.assertIn("entropy_norm", unc)
        self.assertLessEqual(float(unc.get("entropy_norm")), 0.05)
        self.assertEqual(result.get("decision"), "accepted")

        items = result.get("items") or []
        self.assertTrue(items)
        # Debe haber una selección automática (optimized) cuando no hay colapso humano.
        sp = items[0].get("selected_portion") or {}
        self.assertEqual(sp.get("source"), "optimized")
        self.assertAlmostEqual(float(sp.get("grams") or 0.0), 160.0, places=1)


if __name__ == "__main__":
    unittest.main()
