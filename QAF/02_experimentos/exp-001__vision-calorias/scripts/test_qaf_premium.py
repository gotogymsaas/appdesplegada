from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Permite importar módulos desde esta carpeta cuando se ejecuta desde la raíz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qaf_vision_calories import build_portion_candidates, load_calorie_db, qaf_infer_from_vision


class QAFPremiumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calorie_db = load_calorie_db(
            Path(__file__).resolve().parents[1] / "data" / "calorie_db.csv"
        )

    def test_build_portion_candidates_exactly_three(self):
        c = build_portion_candidates("arroz_blanco_cocido", calorie_db=self.calorie_db)
        self.assertEqual(len(c), 3)
        self.assertTrue(all("label" in x and "grams" in x and "confidence_hint" in x for x in c))

    def test_range_driver_arroz_pollo(self):
        vision = {
            "is_food": True,
            "items": ["arroz con pollo"],
            "portion_estimate": "1 taza de arroz y 150g de pollo",
        }
        r = qaf_infer_from_vision(vision, self.calorie_db, user_id="u", locale="es-CO")
        self.assertIn(r.get("range_driver"), ("arroz_blanco_cocido", "pollo_pechuga_asada"))
        self.assertIn(r.get("decision"), ("accepted", "partial", "needs_confirmation"))
        self.assertIsInstance(r.get("items"), list)

    def test_decision_needs_confirmation_when_missing_items(self):
        vision = {"is_food": True, "items": ["caviar"], "portion_estimate": "100g"}
        r = qaf_infer_from_vision(vision, self.calorie_db, user_id="u", locale="es-CO")
        # si normaliza a None, items podría quedar vacío; en ambos casos no debe explotar.
        if r.get("missing_items"):
            self.assertEqual(r.get("decision"), "needs_confirmation")

    def test_parse_zero_units_does_not_crash(self):
        vision = {"is_food": True, "items": ["huevo"], "portion_estimate": "0 unidades"}
        r = qaf_infer_from_vision(vision, self.calorie_db, user_id="u", locale="es-CO")
        self.assertIsInstance(r, dict)


if __name__ == "__main__":
    unittest.main()
