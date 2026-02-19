from django.db import models
from django.conf import settings
class DeviceConnection(models.Model):
    PROVIDERS = [
        ('apple_health', 'Apple Health'),
        ('google_fit', 'Google Fit'),
        ('fitbit', 'Fitbit'),
        ('garmin', 'Garmin'),
        ('whoop', 'WHOOP'),
    ]

    # Línea clave: relación con el usuario
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='device_connections'
    )

    provider = models.CharField(
        max_length=40,
        choices=PROVIDERS
    )

    status = models.CharField(
        max_length=20,
        default='disconnected'
    )  # disconnected | connected | pending | error

    last_sync_at = models.DateTimeField(
        null=True,
        blank=True
    )

    # Tokens OAuth (se usarán más adelante)
    access_token = models.TextField(blank=True, default="")
    refresh_token = models.TextField(blank=True, default="")
    token_expires_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'provider')

    def __str__(self):
        return f'{self.user} - {self.provider} ({self.status})'


class FitnessSync(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='fitness_syncs'
    )
    provider = models.CharField(max_length=40)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    metrics = models.JSONField(default=dict, blank=True)
    raw = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "provider", "created_at"]),
        ]

    def __str__(self):
        return f'{self.user} - {self.provider} ({self.created_at:%Y-%m-%d %H:%M})'


class UserSyncCheckpoint(models.Model):
    """Marca la última ejecución de una tarea de sync por usuario.

    Ejemplos de key:
    - daily_reset
    - sleep_analysis
    - midday_review
    - evening_review
    - weekly_qaf
    - invisible_3h
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sync_checkpoints'
    )
    key = models.CharField(max_length=40)
    last_run_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'key')
        indexes = [
            models.Index(fields=["user", "key"]),
        ]

    def __str__(self):
        return f"{self.user} - {self.key}"


class SyncRequest(models.Model):
    """Cola simple en DB para sync por evento o por apertura de app."""

    STATUS_CHOICES = [
        ("pending", "pending"),
        ("running", "running"),
        ("done", "done"),
        ("error", "error"),
        ("skipped", "skipped"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sync_requests'
    )
    provider = models.CharField(max_length=40, blank=True, default="")
    reason = models.CharField(max_length=80, blank=True, default="")
    priority = models.IntegerField(default=5)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="pending")

    requested_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    error = models.TextField(blank=True, default="")
    result = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["status", "priority", "requested_at"]),
            models.Index(fields=["user", "status"]),
        ]

    def __str__(self):
        return f"{self.user} - {self.provider or 'any'} ({self.status})"


class WhoopRawRecord(models.Model):
    """Almacena payloads crudos de WHOOP por recurso (sleep/workout/cycle/recovery/profile/body).

    Esta capa permite ingestar "todo" inicialmente sin forzar normalización inmediata.
    """

    RESOURCE_TYPES = [
        ("cycle", "cycle"),
        ("sleep", "sleep"),
        ("recovery", "recovery"),
        ("workout", "workout"),
        ("profile", "profile"),
        ("body", "body"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="whoop_raw_records",
    )
    resource_type = models.CharField(max_length=20, choices=RESOURCE_TYPES)
    resource_id = models.CharField(max_length=100)

    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    created_at_remote = models.DateTimeField(null=True, blank=True)
    updated_at_remote = models.DateTimeField(null=True, blank=True)

    timezone_offset = models.CharField(max_length=8, blank=True, default="")
    score_state = models.CharField(max_length=20, blank=True, default="")

    payload = models.JSONField(default=dict, blank=True)

    fetched_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "resource_type", "resource_id")
        indexes = [
            models.Index(fields=["user", "resource_type"], name="devices_whoop_user_type_idx"),
            models.Index(fields=["resource_type", "resource_id"], name="devices_whoop_type_id_idx"),
        ]

    def __str__(self):
        return f"{self.user} - whoop:{self.resource_type}:{self.resource_id}"
