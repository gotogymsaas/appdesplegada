from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from .models import AuditLog, UserDocument, PushToken


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

    def test_admin_set_premium_enables_14d_trial_when_not_active_billing(self):
        self._auth(self.admin)

        self.user.plan = "Gratis"
        self.user.billing_status = "free"
        self.user.trial_active = False
        self.user.trial_started_at = None
        self.user.trial_ends_at = None
        self.user.save(update_fields=["plan", "billing_status", "trial_active", "trial_started_at", "trial_ends_at"])

        before_call = timezone.now()
        res = self.client.put(
            f"/api/users/update_admin/{self.user.id}/",
            data={"plan": "Premium", "username": self.user.username, "email": self.user.email, "reason": "soporte"},
            format="json",
        )
        after_call = timezone.now()

        self.assertEqual(res.status_code, 200)

        self.user.refresh_from_db()
        self.assertEqual(self.user.plan, "Premium")
        self.assertEqual(self.user.billing_status, "trial")
        self.assertTrue(self.user.trial_active)
        self.assertIsNotNone(self.user.trial_started_at)
        self.assertIsNotNone(self.user.trial_ends_at)
        self.assertGreaterEqual(self.user.trial_started_at, before_call)
        self.assertLessEqual(self.user.trial_started_at, after_call)

        trial_delta = self.user.trial_ends_at - self.user.trial_started_at
        self.assertGreaterEqual(trial_delta.total_seconds(), 14 * 24 * 60 * 60 - 5)
        self.assertLessEqual(trial_delta.total_seconds(), 14 * 24 * 60 * 60 + 5)

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


class SelfDeleteAccountTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User = get_user_model()

        self.user = User.objects.create_user(
            username="self1",
            email="self1@example.com",
            password="User12345",
            plan="Premium",
            full_name="Self User",
            profession="Developer",
        )

    def _auth(self, user):
        token = str(RefreshToken.for_user(user).access_token)
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")

    def test_self_delete_requires_valid_confirmation_text(self):
        self._auth(self.user)
        res = self.client.post(
            "/api/users/delete_my_account/",
            data={
                "username": self.user.username,
                "confirm_text": "BORRAR",
                "current_password": "User12345",
                "reason": "solicitud titular",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_self_delete_requires_current_password(self):
        self._auth(self.user)
        res = self.client.post(
            "/api/users/delete_my_account/",
            data={
                "username": self.user.username,
                "confirm_text": "ELIMINAR",
                "current_password": "WrongPass123",
                "reason": "solicitud titular",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_self_delete_success_anonymizes_and_cleans_related_data(self):
        self._auth(self.user)

        UserDocument.objects.create(
            user=self.user,
            doc_type="nutrition_plan",
            file_name="plan.pdf",
            file_url="https://api.gotogym.store/media/nutrition_plans/self1/plan.pdf",
            extracted_text="texto",
        )
        PushToken.objects.create(user=self.user, token="tok-self-delete-1", platform="android")

        res = self.client.post(
            "/api/users/delete_my_account/",
            data={
                "username": self.user.username,
                "confirm_text": "ELIMINAR",
                "current_password": "User12345",
                "reason": "solicitud titular",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)

        self.user.refresh_from_db()
        self.assertFalse(self.user.is_active)
        self.assertTrue(self.user.email.endswith("@example.invalid"))
        self.assertTrue(self.user.username.startswith("deleted_"))
        self.assertIsNone(self.user.full_name)
        self.assertEqual(UserDocument.objects.filter(user=self.user).count(), 0)
        self.assertEqual(PushToken.objects.filter(user=self.user).count(), 0)
        self.assertTrue(AuditLog.objects.filter(action="users.self_delete", entity_id=str(self.user.id)).exists())
