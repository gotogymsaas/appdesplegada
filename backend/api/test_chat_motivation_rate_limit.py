from datetime import date, datetime
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient


User = get_user_model()


class _DummyN8nResp:
    status_code = 200

    def json(self):
        return {"output": "RESPUESTA_N8N"}


class ChatMotivationRateLimitTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="juan",
            email="juan@example.com",
            password="pass12345",
        )

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_motivation_heuristic_is_rate_limited_per_8h_window(self, _mock_post):
        fixed_day = date(2026, 2, 23)

        hour_holder = {"h": 10}  # ventana 1 (08-15)

        def _fake_localtime(_dt=None):
            tz = timezone.get_current_timezone()
            return timezone.make_aware(datetime(2026, 2, 23, int(hour_holder["h"]), 0, 0), tz)

        msg = "No tengo ganas de entrenar. Me cuesta arrancar, pero quiero 10 minutos para mantener constancia."

        with patch("api.views.timezone.localdate", return_value=fixed_day), patch(
            "api.views.timezone.localtime", side_effect=_fake_localtime
        ):
            # 1ra vez en la ventana: activa (reemplaza n8n)
            r1 = self.client.post(
                "/api/chat/",
                {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
                format="json",
            )
            self.assertEqual(r1.status_code, 200)
            out1 = (r1.json() or {}).get("output") or ""
            self.assertIn("Hoy estoy leyendo", out1)
            self.assertNotIn("RESPUESTA_N8N", out1)

            # 2da vez misma ventana: NO debe activar; cae a n8n
            r2 = self.client.post(
                "/api/chat/",
                {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
                format="json",
            )
            self.assertEqual(r2.status_code, 200)
            out2 = (r2.json() or {}).get("output") or ""
            self.assertNotIn("Hoy estoy leyendo", out2)
            self.assertIn("RESPUESTA_N8N", out2)

            # Cambiamos de ventana (16-23): vuelve a activar
            hour_holder["h"] = 18
            r3 = self.client.post(
                "/api/chat/",
                {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
                format="json",
            )
            self.assertEqual(r3.status_code, 200)
            out3 = (r3.json() or {}).get("output") or ""
            self.assertIn("Hoy estoy leyendo", out3)
            self.assertNotIn("RESPUESTA_N8N", out3)

    @patch("api.views.requests.post", return_value=_DummyN8nResp())
    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_motivation_explicit_text_bypasses_rate_limit(self, _mock_post):
        fixed_day = date(2026, 2, 23)

        hour_holder = {"h": 10}

        def _fake_localtime(_dt=None):
            tz = timezone.get_current_timezone()
            return timezone.make_aware(datetime(2026, 2, 23, int(hour_holder["h"]), 0, 0), tz)

        # Mensaje que activa y consume la ventana
        msg = "No tengo ganas de entrenar. Me cuesta arrancar."

        with patch("api.views.timezone.localdate", return_value=fixed_day), patch(
            "api.views.timezone.localtime", side_effect=_fake_localtime
        ):
            r1 = self.client.post(
                "/api/chat/",
                {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
                format="json",
            )
            self.assertEqual(r1.status_code, 200)
            out1 = (r1.json() or {}).get("output") or ""
            self.assertIn("Hoy estoy leyendo", out1)

            # Segunda vez misma ventana: heurístico cae a n8n
            r2 = self.client.post(
                "/api/chat/",
                {"message": msg, "sessionId": "", "attachment": "", "attachment_text": "", "username": "juan"},
                format="json",
            )
            self.assertEqual(r2.status_code, 200)
            out2 = (r2.json() or {}).get("output") or ""
            self.assertIn("RESPUESTA_N8N", out2)

            # Intento explícito por texto: debe activar aunque esté en la misma ventana
            r3 = self.client.post(
                "/api/chat/",
                {
                    "message": "Necesito motivación ahora",
                    "sessionId": "",
                    "attachment": "",
                    "attachment_text": "",
                    "username": "juan",
                },
                format="json",
            )
            self.assertEqual(r3.status_code, 200)
            out3 = (r3.json() or {}).get("output") or ""
            self.assertIn("Hoy estoy leyendo", out3)
            self.assertNotIn("RESPUESTA_N8N", out3)
