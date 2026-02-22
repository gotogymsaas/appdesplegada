from django.test import SimpleTestCase

from api.qaf_meal_planner.engine import build_quick_actions_for_menu


class QAFMealPlannerQuickActionsTests(SimpleTestCase):
    def test_apply_button_hidden_when_plan_already_applied(self):
        qas = build_quick_actions_for_menu(variety_level="normal", is_applied=True)
        labels = [str(x.get("label") or "") for x in qas if isinstance(x, dict)]
        self.assertNotIn("Aplicar menú", labels)
        self.assertIn("Lista de compras", labels)
        self.assertIn("Cambiar cena (mañana)", labels)

    def test_apply_button_present_when_not_applied(self):
        qas = build_quick_actions_for_menu(variety_level="normal", is_applied=False)
        labels = [str(x.get("label") or "") for x in qas if isinstance(x, dict)]
        self.assertIn("Aplicar menú", labels)
