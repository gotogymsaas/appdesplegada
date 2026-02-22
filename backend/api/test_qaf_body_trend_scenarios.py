from django.test import SimpleTestCase

from api.qaf_body_trend.engine import render_professional_summary


class QAFBodyTrendScenarioSummaryTests(SimpleTestCase):
    def test_summary_includes_plus_200_when_present(self):
        payload = {
            "decision": "accepted",
            "confidence": {"score": 0.8},
            "scenarios": {
                "baseline": {"trajectory": [{"weight_kg": 80.0, "weight_kg_min": 79.5, "weight_kg_max": 80.5}]},
                "follow_plan": {"trajectory": [{"weight_kg": 79.0, "weight_kg_min": 78.4, "weight_kg_max": 79.6}]},
                "minus_200": {"trajectory": [{"weight_kg": 78.6, "weight_kg_min": 78.0, "weight_kg_max": 79.2}]},
                "plus_200": {"trajectory": [{"weight_kg": 81.2, "weight_kg_min": 80.6, "weight_kg_max": 81.8}]},
            },
        }
        text = render_professional_summary(payload)
        self.assertIn("plus_200", text)

    def test_preferred_scenario_is_reflected(self):
        payload = {
            "decision": "accepted",
            "confidence": {"score": 0.8},
            "scenarios": {
                "baseline": {"trajectory": [{"weight_kg": 80.0, "weight_kg_min": 79.5, "weight_kg_max": 80.5}]},
                "minus_200": {"trajectory": [{"weight_kg": 78.6, "weight_kg_min": 78.0, "weight_kg_max": 79.2}]},
            },
        }
        text = render_professional_summary(payload, preferred_scenario="minus_200")
        self.assertIn("escenario: minus_200", text)
