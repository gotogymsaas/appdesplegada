from django.contrib.auth import get_user_model
from django.test import TestCase

from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from .models import AuditLog


class AdminSecurityTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User = get_user_model()

        self.admin = User.objects.create_user(
            username="admin",
            email="admin@example.com",
            password="Admin1234",
        )
        self.admin.is_superuser = True
        self.admin.is_staff = True
        self.admin.save()

        self.user = User.objects.create_user(
            username="u1",
            email="u1@example.com",
            password="User12345",
            plan="Gratis",
        )

        self.normal = User.objects.create_user(
            username="n1",
            email="n1@example.com",
            password="User12345",
        )

    def _auth(self, user):
        token = str(RefreshToken.for_user(user).access_token)
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")

    def test_users_list_forbidden_for_non_admin(self):
        self._auth(self.normal)
        res = self.client.get("/api/users/")
        self.assertEqual(res.status_code, 403)

    def test_soft_delete_requires_reason_and_audits(self):
        self._auth(self.admin)

        res = self.client.delete(f"/api/users/delete/{self.user.id}/", data={}, format="json")
        self.assertEqual(res.status_code, 400)

        res = self.client.delete(
            f"/api/users/delete/{self.user.id}/",
            data={"reason": "solicitud del titular"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        self.user.refresh_from_db()
        self.assertFalse(self.user.is_active)
        self.assertTrue(self.user.email.endswith("@example.invalid"))
        self.assertTrue(AuditLog.objects.filter(action="users.soft_delete", entity_id=str(self.user.id)).exists())

    def test_plan_change_requires_reason_and_audits(self):
        self._auth(self.admin)

        res = self.client.put(
            f"/api/users/update_admin/{self.user.id}/",
            data={"plan": "Premium", "username": self.user.username, "email": self.user.email},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

        res = self.client.put(
            f"/api/users/update_admin/{self.user.id}/",
            data={"plan": "Premium", "username": self.user.username, "email": self.user.email, "reason": "soporte"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.user.refresh_from_db()
        self.assertEqual(self.user.plan, "Premium")
        self.assertTrue(AuditLog.objects.filter(action="users.update_admin", entity_id=str(self.user.id)).exists())

    def test_push_broadcast_requires_reason_and_audits(self):
        self._auth(self.admin)

        res = self.client.post("/api/push/admin/broadcast/", data={"title": "t", "body": "b"}, format="json")
        self.assertEqual(res.status_code, 400)

        res = self.client.post(
            "/api/push/admin/broadcast/",
            data={"title": "t", "body": "b", "reason": "campaña"},
            format="json",
        )
        self.assertIn(res.status_code, (200, 400))
        self.assertTrue(AuditLog.objects.filter(action="push.broadcast").exists())
