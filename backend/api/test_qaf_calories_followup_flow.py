from django.test import SimpleTestCase

from api.qaf_calories.engine import load_aliases, load_calorie_db, load_items_meta, load_micros_db, load_nutrition_db, paths, qaf_estimate_v2


class QAFCaloriesFollowUpFlowTests(SimpleTestCase):
    def test_confirmed_portion_stops_repeating_follow_up_buttons(self):
        cat = paths()
        aliases = load_aliases(cat["aliases"])
        calorie_db = load_calorie_db(cat["calorie_db"])
        nutrition_db = load_nutrition_db(cat["nutrition_db"])
        micros_db = load_micros_db(cat["micros_db"])
        items_meta = load_items_meta(cat["items_meta"])

        # Usamos "huevo" porque en el dataset tiene unidad (1/2/3) y por defecto genera incertidumbre.
        vision = {
            "is_food": True,
            "items": ["huevo"],
            "portion_estimate": "",
            "notes": "",
        }

        out1 = qaf_estimate_v2(
            vision,
            calorie_db=calorie_db,
            nutrition_db=nutrition_db,
            micros_db=micros_db,
            aliases=aliases,
            items_meta=items_meta,
            locale="es-CO",
            confirmed_portions=None,
            goal_kcal_meal=None,
        )

        self.assertTrue(out1.get("needs_confirmation"))
        fu1 = out1.get("follow_up_questions") or []
        self.assertTrue(isinstance(fu1, list) and len(fu1) >= 1)
        self.assertEqual((fu1[0] or {}).get("type"), "confirm_portion")

        # Confirmamos porción (ej. 2 unidades = 100g). Con esto, el rango de ese ítem debe fijarse
        # y no debe seguir apareciendo la misma pregunta/botones.
        out2 = qaf_estimate_v2(
            vision,
            calorie_db=calorie_db,
            nutrition_db=nutrition_db,
            micros_db=micros_db,
            aliases=aliases,
            items_meta=items_meta,
            locale="es-CO",
            confirmed_portions=[{"item_id": "huevo", "grams": 100}],
            goal_kcal_meal=None,
        )

        self.assertFalse(bool(out2.get("needs_confirmation")))
        fu2 = out2.get("follow_up_questions") or []
        self.assertEqual(fu2, [])
