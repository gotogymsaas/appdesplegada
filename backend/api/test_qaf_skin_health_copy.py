from __future__ import annotations

import unittest

from django.test import SimpleTestCase


class QAFSkinHealthCopyTests(SimpleTestCase):
    def _make_test_image_bytes(self) -> bytes:
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"dependencies missing: {exc}")

        # Imagen sintética con variación suficiente (evita low-quality por exposición/dinámico)
        h, w = 240, 240
        x = np.linspace(0.15, 0.85, w, dtype=np.float32)
        y = np.linspace(0.20, 0.80, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        r = (0.65 * xv + 0.15).clip(0, 1)
        g = (0.55 * yv + 0.20).clip(0, 1)
        b = (0.35 * (1 - xv) + 0.25).clip(0, 1)
        rgb = (np.stack([r, g, b], axis=2) * 255.0).astype(np.uint8)

        im = Image.fromarray(rgb, mode="RGB")
        import io

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    def test_renderer_is_human_and_no_debug_keys(self):
        from api.qaf_skin_health.engine import evaluate_skin_health, render_professional_summary

        res = evaluate_skin_health(
            image_bytes=self._make_test_image_bytes(),
            content_type="image/jpeg",
            context={
                'sleep_minutes': 360,
                'stress_1_5': 4,
                'movement_1_5': 2,
                'water_liters': 1.0,
                'sun_minutes': 30,
            },
            baseline=None,
        ).payload
        res = dict(res)
        res["user_display_name"] = "Juan Manuel"

        text = render_professional_summary(res)
        self.assertTrue(text.startswith("Hola"))
        self.assertIn("Skin Health", text)
        self.assertIn("no es diagnóstico", text.lower())
        self.assertNotIn("decision:", text.lower())
        self.assertNotIn("confidence:", text.lower())
        self.assertIn("Prioridad 1", text)
        self.assertIn("Acciones simples", text)
        self.assertIn("IA contextual", text)
        self.assertIn("tu piel refleja tu sistema", text.lower())

        # No activos médicos / tratamientos
        banned = [
            'tretino',
            'retinol',
            'hidroquinona',
            'isotretino',
            'antibió',
            'cortico',
            'esteroide',
        ]
        low = text.lower()
        for b in banned:
            self.assertNotIn(b, low)

    def test_renderer_handles_needs_confirmation(self):
        from api.qaf_skin_health.engine import render_professional_summary

        res = {
            "decision": "needs_confirmation",
            "confidence": {"score": 0.1, "uncertainty_score": 0.9},
            "user_display_name": "Juan Manuel",
        }
        text = render_professional_summary(res)
        self.assertIn("No pude", text)
        self.assertIn("Luz natural", text)
