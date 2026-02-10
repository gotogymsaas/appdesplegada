from django.db import models
from django.conf import settings
class DeviceConnection(models.Model):
    PROVIDERS = [
        ('apple_health', 'Apple Health'),
        ('google_fit', 'Google Fit'),
        ('fitbit', 'Fitbit'),
        ('garmin', 'Garmin'),
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
